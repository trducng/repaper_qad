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

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.types import TaskFunction
import submitit

import cloudpickle
import src

# Otherwise pickle.load from submitit launcher won't find src since cloudpickle registers by reference by default
cloudpickle.register_pickle_by_value(src)

from image_classification.src.utils import norm_grad
from masked_autoencoding.src.datasets import get_train_val_datasets
from image_classification.src.samplers import BalancedBatchSampler
from image_classification.src.model_and_trainer import ModelAndTrainer

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
            tags = [
                cfg.experiment.name,
                cfg.pretrained_model._target_,
                cfg.dataset.name,
            ]
            if not cfg.experiment.trainer.linear_probing.use_pretrained_model:
                tags[1] = cfg.classifier._target_
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
                tags=tags,
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
            "validation/accuracy_top1",
            step_metric="train/step",
            summary="best",
            goal="maximize",
        )
        run.define_metric("validation/examples", step_metric="validation/step")
        LOGGER.info("Logged wandb metrics available at url: %s", run.get_url())

        run.summary["workdir"] = str(save)

        # azy importing for pickling purposes with submitit and cloudpickle so that cuda.deterministic=True and
        # cudnn.benchmark=False don't break the pickling process (they change the torch backend, so importing
        # them outside would mean pickle wouldn't pickle the modified torch)
        import torch
        from torch.utils.data import DataLoader
        import torch.backends.cudnn as cudnn

        # Setting deterministic behaviour for RNG for results reproducibility purposes
        torch.manual_seed(cfg.experiment.random_seed)
        torch.cuda.manual_seed(cfg.experiment.random_seed)
        if cfg.experiment.deterministic:
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            cudnn.benchmark = True
            LOGGER.warn("Using non-deterministic cudnn backend")
        np.random.seed(cfg.experiment.random_seed)
        random.seed(cfg.experiment.random_seed)

        # Displaying device info
        if torch.cuda.is_available and "cuda" in cfg.experiment.device:
            self.device = device = torch.device(cfg.experiment.device)
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
            self.device = device = torch.device("cpu")
            LOGGER.info("CUDA not available. Using CPU mode.")
            mode = "cpu"

        # Setup model, loss, opt, and train & val loss arrays
        LOGGER.info("Setting up model and loss...")
        model_and_trainer = self.model_and_trainer = ModelAndTrainer(cfg, device)
        model = model_and_trainer.model
        classifier = model_and_trainer.classifier
        opt = model_and_trainer.opt
        loss = model_and_trainer.loss
        lr_sched = model_and_trainer.lr_sched

        # Log gradients and classifier params on wandb
        if model is not None:
            run.watch(model, log_freq=cfg.experiment.log_frequency * 10)
        run.watch(classifier, log_freq=cfg.experiment.log_frequency * 10)

        # Load latest checkpoint for both model and classifier in save dir (cwd) if requested,
        # otherwise randomly initialize
        if not model_and_trainer.load_latest_checkpoint(cfg, save):
            # Load user-defined checkpoint for pre-trained model if no latest checkpoint found above
            if model is not None and not model_and_trainer.load_pretrained_model(cfg):
                raise RuntimeError("Pretrained model checkpoint must be provided.")

        # Log model, classifier, and number of params for each
        if model is not None:
            LOGGER.info(
                "Using model: %s", model
            )  # NOTE: logging multi-line logs is a no-no
            num_model_train_params = sum(
                param.numel() for param in model.parameters() if param.requires_grad
            )
            num_model_nontrain_params = sum(
                param.numel() for param in model.parameters() if not param.requires_grad
            )
        else:
            num_model_train_params = num_model_nontrain_params = 0
        LOGGER.info(
            "Using classifier: %s", classifier
        )  # NOTE: logging multi-line logs is a no-no
        num_classifier_train_params = sum(
            param.numel() for param in classifier.parameters() if param.requires_grad
        )
        num_classifier_nontrain_params = sum(
            param.numel()
            for param in classifier.parameters()
            if not param.requires_grad
        )
        num_train_params = num_model_train_params + num_classifier_train_params
        num_nontrain_params = num_model_nontrain_params + num_classifier_nontrain_params
        LOGGER.info("Trainable params (model): %s", num_model_train_params)
        LOGGER.info("Untrainable params (model): %s", num_model_nontrain_params)
        LOGGER.info(
            "Total number of params (model): %s",
            num_model_train_params + num_model_nontrain_params,
        )
        LOGGER.info("Trainable params (classifier): %s", num_classifier_train_params)
        LOGGER.info(
            "Untrainable params (classifier): %s", num_classifier_nontrain_params
        )
        LOGGER.info(
            "Total number of params (classifier): %s",
            num_classifier_train_params + num_classifier_nontrain_params,
        )
        LOGGER.info("Trainable params (all): %s", num_train_params)
        LOGGER.info("Untrainable params (all): %s", num_nontrain_params)
        LOGGER.info(
            "Total number of params (all): %s", num_train_params + num_nontrain_params
        )
        run.summary["model_trainable_params"] = num_model_train_params
        run.summary["model_untrainable_params"] = num_model_nontrain_params
        run.summary["model_total_params"] = (
            num_model_train_params + num_model_nontrain_params
        )
        run.summary["classifier_trainable_params"] = num_classifier_train_params
        run.summary["classifier_untrainable_params"] = num_classifier_nontrain_params
        run.summary["classifier_total_params"] = (
            num_classifier_train_params + num_classifier_nontrain_params
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
            {"num_workers": cfg.experiment.num_workers, "pin_memory": True}
            if mode == "gpu"
            else {}
        )
        if not cfg.experiment.trainer.fewshot.enabled:
            sampler = torch.utils.data.RandomSampler(
                train_dataset,
                replacement=True,
                num_samples=(
                    cfg.experiment.iter.train.total
                    + 1
                    - cfg.experiment.iter.train.start
                )
                * train_batch_size,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                sampler=sampler,
                drop_last=True,
                **gpuargs,
            )
            iter_train_total = cfg.experiment.iter.train.total
        else:
            batch_sampler = BalancedBatchSampler(
                train_dataset,
                classes=train_dataset.classes,
                samples_per_class=cfg.experiment.trainer.fewshot.samples_per_class,
                episodes=cfg.experiment.iter.train.total
                + 1
                - cfg.experiment.iter.train.start,
                episode_length=cfg.experiment.trainer.fewshot.episode_length,
            )
            train_batch_size = batch_sampler.batch_size
            train_loader = DataLoader(
                train_dataset, batch_sampler=batch_sampler, **gpuargs
            )
            with open_dict(cfg.experiment.batch_size):
                OmegaConf.update(cfg.experiment.batch_size, "train", train_batch_size)
            iter_train_total = len(batch_sampler)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
            **gpuargs,
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
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            # Forward pass
            forward_start = time.perf_counter()
            results = model_and_trainer(cfg, i, x, y)
            forward_time = time.perf_counter() - forward_start

            # Compute losses
            losses = loss(cfg, model, results, phase="train")
            total_loss = losses["ce_loss"]

            # Include denoising autoencoding loss if fine-tuning
            if model_and_trainer.method == "fine_tuning":
                total_loss = total_loss + losses["rec_loss"] + losses["overflow_loss"]

            # Backward pass and logging
            with torch.no_grad():
                opt.zero_grad()

                backward_start = time.perf_counter()
                total_loss.backward()
                backward_time = time.perf_counter() - backward_start

                # Normalize model gradients
                # TODO: don't normalize gradients for non-CA models
                if cfg.experiment.normalize_gradients and model is not None:
                    norm_grad(model)

                opt.step()
                if lr_sched is not None:
                    lr_sched.step()

                # Add new states to nca pool, shuffle, and retain first {pool_size} states
                if (
                    "CA" in cfg.pretrained_model._target_
                    and model_and_trainer.method == "fine_tuning"
                    and cfg.experiment.trainer.fine_tuning.masking_enabled
                ):
                    model_and_trainer.update_pools(
                        cfg,
                        results["ground_truth"]["x"],
                        results["ground_truth"]["y"],
                        results["output_cells"].detach(),
                        results["was_sampled"],
                    )

                # Track losses and scaling factors
                train_scalars = {
                    "cross_entropy_loss": losses["ce_loss"].item(),
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
                        "input_img": results["masked_input"],
                        "output_img": results["output_img"],
                        "output_img_from_unmasked": results.get(
                            "output_img_from_unmasked", None
                        ),
                        "ground_truth_img": results["ground_truth"]["x"],
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
                        "total loss: %f, cross entropy loss: %f, reconstruction loss: %f, overflow loss: %f, "
                        "lr: %f, pool size: %d, f-time: %fs, b-time: %fs",
                        i,
                        iter_train_total,
                        train_results["total_loss"],
                        train_results["cross_entropy_loss"],
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
                    "cross_entropy_loss": avg_val_loss["ce"].item(),
                    "reconstruction_loss": avg_val_loss["rec"].item(),
                    "reconstruction_factor": loss.rec_factor,
                    "overflow_loss": avg_val_loss["overflow"].item(),
                    "overflow_factor": loss.overflow_factor,
                    "total_loss": avg_val_loss["total"].item(),
                    "accuracy_top1": avg_val_loss["acc_top1"].item(),
                    "accuracy_top3": avg_val_loss["acc_top3"].item(),
                    "accuracy_top5": avg_val_loss["acc_top5"].item(),
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
                    "avg cross entropy loss: %.3f, avg reconstruction loss: %.3f, avg overflow loss: %.3f, "
                    "avg total loss: %.3f, avg acc (top-1): %.3f, avg acc (top-3): %.3f, avg acc (top-5): %.3f, "
                    "avg psnr: %.3f",
                    i,
                    val_scalars["cross_entropy_loss"],
                    val_scalars["reconstruction_loss"],
                    val_scalars["overflow_loss"],
                    val_scalars["total_loss"],
                    val_scalars["accuracy_top1"],
                    val_scalars["accuracy_top3"],
                    val_scalars["accuracy_top5"],
                    val_scalars["psnr"],
                )

                # Keeping track of training iteration with lowest validation reconstruction error
                if val_scalars["accuracy_top1"] > model_and_trainer.best_avg_val_acc:
                    model_and_trainer.best_avg_val_acc = val_scalars["accuracy_top1"]
                    best_model_at_train_iter = i
                    run.summary["best_avg_val_acc"] = model_and_trainer.best_avg_val_acc
                    run.summary["best_model_at_train_iter"] = i

                    # Saving then copying model to nca_best.pth.tar
                    self.model_and_trainer.save_checkpoint(
                        cfg, best_model_at_train_iter, save
                    )
                    chk_fname = save / f"ckpt_{best_model_at_train_iter}.pth.tar"
                    best_chk_fname = save / "ckpt_best.pth.tar"
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
            text=f"Job {run.name} in group {run.group} finished with best validation accuracy (top-1) "
            f"{model_and_trainer.best_avg_val_acc:.4f} at train iteration {best_model_at_train_iter}"
            f"\nOverrides:\n{OmegaConf.to_yaml(runtime_cfg.overrides.task)}\nWorking directory: {save}",
            level=wandb.AlertLevel.INFO,
        )

        if not within_wandb_context:
            run.finish()

        return model_and_trainer.best_avg_val_acc

    def validate(
        self, torch, cfg, run, model_and_trainer, train_step, validation_loader
    ):
        model = model_and_trainer.model
        loss = model_and_trainer.loss
        model_and_trainer.eval()
        LOGGER.info("Validating at iter %s.", train_step)
        with torch.no_grad():
            avg_val_loss = {
                "ce": 0,
                "rec": 0,
                "overflow": 0,
                "total": 0,
                "acc_top1": 0,
                "acc_top3": 0,
                "acc_top5": 0,
                "psnr": 0,
            }
            num_batches = len(validation_loader)
            j = 1  # or else /= float(j) below will complain
            for j, (x, y) in enumerate(validation_loader, start=1):
                x = x.to(self.device)
                y = y.to(self.device).unsqueeze(1)

                # Forward pass
                forward_start = time.perf_counter()
                results = model_and_trainer(cfg, j, x, y, phase="validation")
                forward_time = time.perf_counter() - forward_start

                # Compute losses
                losses = loss(cfg, model, results, phase="validation")
                total_loss = losses["ce_loss"]

                if model_and_trainer.method == "fine_tuning":
                    total_loss = (
                        total_loss + losses["rec_loss"] + losses["overflow_loss"]
                    )

                # Accumulate them
                avg_val_loss["ce"] += losses["ce_loss"]
                avg_val_loss["rec"] += losses["rec_loss"]
                avg_val_loss["overflow"] += losses["overflow_loss"]
                avg_val_loss["total"] += total_loss
                avg_val_loss["acc_top1"] += losses["acc_top1"]
                avg_val_loss["acc_top3"] += losses["acc_top3"]
                avg_val_loss["acc_top5"] += losses["acc_top5"]
                avg_val_loss["psnr"] += losses["psnr"]

                if j % cfg.experiment.log_frequency == 0:
                    # Log output labels, inputs, and ground truths
                    val_img_results = {
                        "input_img": results["masked_input"],
                        "output_img": results["output_img"],
                        "output_img_from_unmasked": results.get(
                            "output_img_from_unmasked", None
                        ),
                        "ground_truth_img": results["ground_truth"]["x"],
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
                        "accuracy (top-1): %f, accuracy (top-3): %f, accuracy (top-5): %f, psnr: %f, total loss: %f, "
                        "cross entropy loss: %f, reconstruction loss: %f, "
                        "overflow loss: %f, f-time: %fs",
                        j,
                        len(validation_loader),
                        losses["acc_top1"],
                        losses["acc_top3"],
                        losses["acc_top5"],
                        losses["psnr"],
                        total_loss,
                        losses["ce_loss"],
                        losses["rec_loss"],
                        losses["overflow_loss"],
                        forward_time,
                    )

            # Average losses over validation set
            avg_val_loss["ce"] /= float(j)
            avg_val_loss["rec"] /= float(j)
            avg_val_loss["overflow"] /= float(j)
            avg_val_loss["total"] /= float(j)
            avg_val_loss["acc_top1"] /= float(j)
            avg_val_loss["acc_top3"] /= float(j)
            avg_val_loss["acc_top5"] /= float(j)
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
