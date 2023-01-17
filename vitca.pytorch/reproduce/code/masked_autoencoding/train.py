"""
Imports and utility functions
"""
import os
import random
import shutil
import logging
import time
from pathlib import Path

import numpy as np
import wandb

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.types import TaskFunction
import submitit

import cloudpickle
import src

# Otherwise pickle.load from submitit launcher won't find src since cloudpickle registers by reference by default
cloudpickle.register_pickle_by_value(src)

from masked_autoencoding.src.utils import norm_grad
from masked_autoencoding.src.datasets import get_train_val_datasets
from masked_autoencoding.src.model_and_trainer import ModelAndTrainer

LOGGER = logging.getLogger(__name__)


class Train(TaskFunction):
    def __init__(self):
        self.model_and_trainer = None
        self.run = None
        self.cfg = None
        self.src = src
        self.ready_to_checkpoint = True
        self.train_iter = 0
        self.save = None

    # Main train loop code
    def __call__(self, cfg: DictConfig) -> float:
        self.cfg = cfg
        run = self.run = wandb.run
        within_wandb_context = False

        try:
            env = submitit.JobEnvironment()
            LOGGER.info(
                "Process ID %s executing experiment %s, with %s.",
                os.getpid(),
                cfg.experiment.name,
                env,
            )
        except RuntimeError:
            LOGGER.info(
                "Process ID %s executing experiment %s",
                os.getpid(),
                cfg.experiment.name,
            )

        save = self.save = Path(os.getcwd())

        LOGGER.info("Current working/save directory: %s", save)

        # W&B setup
        runtime_cfg = hydra.core.hydra_config.HydraConfig.get()
        if run is None:
            LOGGER.info("Initializing wandb...")
            if "BasicSweeper" in runtime_cfg.sweeper._target_:
                experiment_group = save.name
                wandb_run_id = experiment_group
            else:
                experiment_group = save.parent.stem
                wandb_run_id = experiment_group + "-" + save.name
            if cfg.wandb.run_id is not None:
                wandb_run_id = cfg.wandb.run_id
            run = self.run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                settings=wandb.Settings(start_method="thread"),
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                group=experiment_group,
                name=save.name,
                notes=OmegaConf.to_yaml(runtime_cfg.overrides.task),
                resume="allow",
                id=wandb_run_id,
                tags=[cfg.experiment.name, cfg.model._target_, cfg.dataset.name],
            )
        else:
            within_wandb_context = True
        # Defining custom x axis metrics
        run.define_metric("train/step")
        run.define_metric("validation/step")
        # Defining metrics to plot against the above x axes
        run.define_metric("train/*", step_metric="train/step")
        run.define_metric("validation/*", step_metric="train/step")
        run.define_metric(
            "validation/reconstruction_loss", step_metric="train/step", summary="best"
        )
        run.define_metric("validation/examples", step_metric="validation/step")
        LOGGER.info("Logged wandb metrics available at url: %s", run.get_url())

        run.summary["workdir"] = str(save)

        # Lazy importing for pickling purposes with submitit and cloudpickle so that cuda.deterministic=True and
        # cudnn.benchmark=False don't break the pickling process (they change the torch backend, so importing
        # them outside would mean pickle wouldn't pickle the modified torch)
        import torch
        from torch.utils.data import DataLoader
        import torch.backends.cudnn as cudnn

        # Setting deterministic behaviour for RNG for results reproducibility purposes
        torch.manual_seed(cfg.experiment.random_seed)
        torch.cuda.manual_seed(cfg.experiment.random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(cfg.experiment.random_seed)
        random.seed(cfg.experiment.random_seed)

        # Displaying device info
        if torch.cuda.is_available and "cuda" in cfg.experiment.device:
            device = torch.device(cfg.experiment.device)
            devices = list(range(torch.cuda.device_count()))
            devices_properties = [
                torch.cuda.get_device_properties(torch.device(f"cuda:{d}"))
                for d in devices
            ]
            LOGGER.info("CUDA available.")
            for i in range(len(devices)):
                LOGGER.info("Using GPU/device %s: %s", i, devices_properties[i])
            mode = "gpu"
        else:
            device = torch.device("cpu")
            LOGGER.info("CUDA not available. Using CPU mode.")
            mode = "cpu"

        # Setup model, loss, opt, and train & val loss arrays
        LOGGER.info("Setting up model and loss...")
        model_and_trainer = self.model_and_trainer = ModelAndTrainer(cfg, device)
        model = model_and_trainer.model
        opt = model_and_trainer.opt
        loss = model_and_trainer.loss
        lr_sched = model_and_trainer.lr_sched

        # Log gradients and model params on wandb
        run.watch(model, log_freq=cfg.experiment.log_frequency * 10)

        # Load latest checkpoint in save dir (cwd) if requested, otherwise randomly initialize
        model_and_trainer.load_latest_checkpoint(cfg, save)

        # Log model and number of params
        LOGGER.info(
            "Using model: %s", model
        )  # NOTE: logging multi-line logs is a no-no
        num_train_params = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        num_nontrain_params = sum(
            param.numel() for param in model.parameters() if not param.requires_grad
        )
        LOGGER.info("Trainable params: %s", num_train_params)
        LOGGER.info("Untrainable params: %s", num_nontrain_params)
        LOGGER.info(
            "Total number of params: %s", num_train_params + num_nontrain_params
        )
        run.summary["trainable_params"] = num_train_params
        run.summary["untrainable_params"] = num_nontrain_params
        run.summary["total_params"] = num_train_params + num_nontrain_params

        # Setup dataset and dataloader
        train_size = list(cfg.experiment.input_size.train)
        train_batch_size = cfg.experiment.batch_size.train
        validation_size = list(cfg.experiment.input_size.val)
        val_batch_size = cfg.experiment.batch_size.val
        LOGGER.info(
            "Training on dataset: %s, train batch size: %s, train image size: %s, "
            "val batch size: %s, val image size: %s",
            cfg.dataset.name,
            train_batch_size,
            train_size,
            val_batch_size,
            validation_size,
        )
        train_dataset, validation_dataset = get_train_val_datasets(
            cfg.dataset.name, cfg.dataset.dataset_root, train_size, validation_size
        )
        gpuargs = (
            {
                "num_workers": cfg.experiment.num_workers,
                "pin_memory": True,
                "drop_last": True,
            }
            if mode == "gpu"
            else {}
        )  # NOTE: drop_last should be false for val
        sampler = torch.utils.data.RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=(
                cfg.experiment.iter.train.total + 1 - cfg.experiment.iter.train.start
            )
            * train_batch_size,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=train_batch_size, sampler=sampler, **gpuargs
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=val_batch_size, shuffle=False, **gpuargs
        )

        # Training loop
        LOGGER.info(
            "Starting training at iter %s. Training for %s iterations.",
            cfg.experiment.iter.train.start,
            len(train_loader),
        )
        best_model_at_train_iter = 1
        model_and_trainer.train()
        for i, (x, y) in enumerate(train_loader, start=cfg.experiment.iter.train.start):
            self.ready_to_checkpoint = False
            x = x.to(model.device)
            y = y.to(model.device)

            # Forward pass
            forward_start = time.perf_counter()
            results = model_and_trainer(cfg, i, x, y)
            forward_time = time.perf_counter() - forward_start

            # Compute losses
            losses = loss(cfg, model, results, phase="train")
            total_loss = losses["rec_loss"] + losses["overflow_loss"]

            # Backward pass and logging
            with torch.no_grad():
                opt.zero_grad()

                backward_start = time.perf_counter()
                total_loss.backward()
                backward_time = time.perf_counter() - backward_start

                # Normalize gradients
                # TODO: don't normalize gradients for non-CA models
                if cfg.experiment.normalize_gradients:
                    norm_grad(model)

                opt.step()
                lr_sched.step()

                # Add new states to nca pool, shuffle, and retain first {pool_size} states
                if "CA" in cfg.model._target_:
                    model_and_trainer.update_pools(
                        cfg,
                        results["ground_truth"]["x"],
                        results["ground_truth"]["y"],
                        results["output_cells"].detach(),
                        results["was_sampled"],
                    )

                # Track losses and scaling factors
                train_scalars = {
                    "reconstruction_loss": losses["rec_loss"].item(),
                    "reconstruction_factor": loss.rec_factor,
                    "overflow_loss": losses["overflow_loss"].item(),
                    "overflow_factor": loss.overflow_factor,
                    "total_loss": total_loss.item(),
                }
                model_and_trainer.update_tracked_scalars(
                    train_scalars, i, phase="train"
                )

                # Logging to wandb and logger
                if i % cfg.experiment.log_frequency == 0:
                    train_results = {
                        "masked_input": results["masked_input"],
                        "output_img": results["output_img"],
                        "ground_truth": results["ground_truth"]["x"],
                        "learning_rate": opt.param_groups[0]["lr"],
                        "pool_size": len(model_and_trainer.z_pool)
                        if hasattr(model_and_trainer, "z_pool")
                        else 0,
                        "forward_time": forward_time,
                        "backward_time": backward_time,
                        **train_scalars,
                    }
                    model_and_trainer.log_to_wandb(
                        run, i, train_results, prefix="train"
                    )

                    # Logging train results to logger
                    LOGGER.info(
                        "Training results at train iter %s/%s: "
                        "total loss: %f, reconstruction loss: %f, overflow loss: %f, lr: %f, pool size: %d, "
                        "f-time: %fs, b-time: %fs",
                        i,
                        cfg.experiment.iter.train.total,
                        train_results["total_loss"],
                        train_results["reconstruction_loss"],
                        train_results["overflow_loss"],
                        train_results["learning_rate"],
                        len(model_and_trainer.z_pool)
                        if hasattr(model_and_trainer, "z_pool")
                        else 0,
                        forward_time,
                        backward_time,
                    )

            # For pre-emption purposes
            self.ready_to_checkpoint = True
            self.train_iter = i

            if i % cfg.experiment.val_frequency == 0:
                avg_val_loss = self.validate(
                    torch, cfg, run, model_and_trainer, i, validation_loader
                )

                # Track avg validation losses and scaling factors
                val_scalars = {
                    "reconstruction_loss": avg_val_loss["rec"].item(),
                    "reconstruction_factor": loss.rec_factor,
                    "overflow_loss": avg_val_loss["overflow"].item(),
                    "overflow_factor": loss.overflow_factor,
                    "total_loss": avg_val_loss["total"].item(),
                    "psnr": avg_val_loss["psnr"].item(),
                }
                model_and_trainer.update_tracked_scalars(
                    val_scalars, i, phase="validation"
                )

                # Log scalars to wandb. Other results such as images are logged to wandb within self.validate(...)
                model_and_trainer.log_to_wandb(
                    run,
                    i,
                    val_scalars,
                    prefix="validation",
                    step_prefix="train",
                    scalars_only=True,
                )

                # Logging validation results to logger
                LOGGER.info(
                    "Validation avg results at train iter %s: "
                    "avg reconstruction loss: %.3f, avg overflow loss: %.3f, avg total loss: %.3f, avg psnr: %.3f",
                    i,
                    val_scalars["reconstruction_loss"],
                    val_scalars["overflow_loss"],
                    val_scalars["total_loss"],
                    val_scalars["psnr"],
                )

                # Keeping track of training iteration with lowest validation reconstruction error
                if (
                    val_scalars["reconstruction_loss"]
                    < model_and_trainer.best_avg_val_rec_err
                ):
                    model_and_trainer.best_avg_val_rec_err = val_scalars[
                        "reconstruction_loss"
                    ]
                    best_model_at_train_iter = i
                    run.summary[
                        "best_avg_val_rec_err"
                    ] = model_and_trainer.best_avg_val_rec_err
                    run.summary["best_model_at_train_iter"] = i

                    # Saving then copying model to nca_best.pth.tar
                    self.model_and_trainer.save_checkpoint(
                        cfg, best_model_at_train_iter, save
                    )
                    chk_fname = save / f"nca_{best_model_at_train_iter}.pth.tar"
                    best_chk_fname = save / "nca_best.pth.tar"
                    run.summary["best_model_filepath"] = str(best_chk_fname)
                    LOGGER.info(
                        "Best model (train iter %s). Copying to %s.",
                        best_model_at_train_iter,
                        best_chk_fname,
                    )
                    try:
                        shutil.copyfile(chk_fname, best_chk_fname)
                    except FileNotFoundError:
                        LOGGER.exception("Best model file not found. Skipping save.")

            if i % cfg.experiment.save_frequency == 0:
                self.model_and_trainer.save_checkpoint(cfg, i, save)

        run.alert(
            title="Job Finished",
            text=f"Job {run.name} in group {run.group} finished with best validation error "
            f"{model_and_trainer.best_avg_val_rec_err:.4f} at train iteration {best_model_at_train_iter}"
            f"\nOverrides:\n{OmegaConf.to_yaml(runtime_cfg.overrides.task)}\nWorking directory: {save}",
            level=wandb.AlertLevel.INFO,
        )

        if not within_wandb_context:
            run.finish()

        return model_and_trainer.best_avg_val_rec_err

    def validate(
        self, torch, cfg, run, model_and_trainer, train_step, validation_loader
    ):
        model = model_and_trainer.model
        loss = model_and_trainer.loss
        model_and_trainer.eval()
        LOGGER.info("Validating at iter %s.", train_step)
        with torch.no_grad():
            avg_val_loss = {"rec": 0, "overflow": 0, "total": 0, "psnr": 0}
            num_batches = len(validation_loader)
            j = 1  # or else /= float(j) below will complain
            for j, (x, y) in enumerate(validation_loader, start=1):
                x = x.to(model.device)
                y = y.to(model.device)

                # Forward pass
                forward_start = time.perf_counter()
                results = model_and_trainer(cfg, j, x, y, phase="validation")
                forward_time = time.perf_counter() - forward_start

                # Compute losses
                losses = loss(cfg, model, results, phase="validation")
                total_loss = losses["rec_loss"] + losses["overflow_loss"]

                # Accumulate them
                avg_val_loss["rec"] += losses["rec_loss"]
                avg_val_loss["overflow"] += losses["overflow_loss"]
                avg_val_loss["total"] += total_loss
                avg_val_loss["psnr"] += losses["psnr"]

                if j % cfg.experiment.log_frequency == 0:
                    # Log output images, masked inputs, and ground truths
                    val_img_results = {
                        "masked_input": results["masked_input"],
                        "output_img": results["output_img"],
                        "ground_truth": results["ground_truth"]["x"],
                    }
                    model_and_trainer.log_to_wandb(
                        run,
                        ((train_step - 1) * num_batches) + j,
                        val_img_results,
                        prefix="validation",
                        step_prefix="validation",
                        images_only=True,
                    )

                    # Logging val results to logger
                    LOGGER.info(
                        "Validation results at val iter %s/%s: "
                        "psnr: %f, total loss: %f, reconstruction loss: %f, overflow loss: %f, f-time: %fs",
                        j,
                        len(validation_loader),
                        losses["psnr"],
                        total_loss,
                        losses["rec_loss"],
                        losses["overflow_loss"],
                        forward_time,
                    )

            # Average losses over validation set
            avg_val_loss["rec"] /= float(j)
            avg_val_loss["overflow"] /= float(j)
            avg_val_loss["total"] /= float(j)
            avg_val_loss["psnr"] /= float(j)
        model_and_trainer.train()
        return avg_val_loss


"""
Main program
"""
if __name__ == "__main__":
    train = Train()
    app = hydra.main(config_path="conf", config_name="train_config")(train.__call__)
    app()
