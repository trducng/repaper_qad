import os
import random
import time
import logging
from pathlib import Path

import numpy as np

import hydra
from hydra.types import TaskFunction

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from masked_autoencoding.src.utils import norm_grad
from masked_autoencoding.src.datasets import get_train_val_datasets
from masked_autoencoding.src.model_and_trainer import ModelAndTrainer

LOGGER = logging.getLogger(__name__)


class Train(TaskFunction):

    # Main train loop code
    def __call__(self, cfg) -> float:
        LOGGER.info(
            "Process ID %s executing experiment %s",
            os.getpid(),
            cfg.experiment.name,
        )

        save = Path(os.getcwd())

        LOGGER.info("Current working/save directory: %s", save)

        # Setting deterministic behaviour for RNG for results reproducibility purposes
        torch.manual_seed(cfg.experiment.random_seed)
        torch.cuda.manual_seed(cfg.experiment.random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
        np.random.seed(cfg.experiment.random_seed)
        random.seed(cfg.experiment.random_seed)

        # Displaying device info
        device = torch.device(cfg.experiment.device)
        devices = list(range(torch.cuda.device_count()))
        devices_properties = [
            torch.cuda.get_device_properties(torch.device(f"cuda:{d}"))
            for d in devices
        ]
        LOGGER.info("CUDA available.")
        for i in range(len(devices)):
            LOGGER.info("Using GPU/device %s: %s", i, devices_properties[i])

        # Setup model, loss, opt, and train & val loss arrays
        LOGGER.info("Setting up model and loss...")
        model_and_trainer = ModelAndTrainer(cfg, device)
        model = model_and_trainer.model
        opt = model_and_trainer.opt
        loss = model_and_trainer.loss
        lr_sched = model_and_trainer.lr_sched

        # Log model and number of params
        LOGGER.info(
            "Using model: %s", model
        )  # NOTE: logging multi-line logs is a no-no

        # Setup dataset and dataloader
        train_size = list(cfg.experiment.input_size.train)
        train_batch_size = cfg.experiment.batch_size.train
        validation_size = list(cfg.experiment.input_size.val)

        train_dataset, _ = get_train_val_datasets(
            cfg.dataset.name, cfg.dataset.dataset_root, train_size, validation_size
        )
        sampler = torch.utils.data.RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=(
                cfg.experiment.iter.train.total + 1 - cfg.experiment.iter.train.start
            )
            * train_batch_size,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=train_batch_size, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True
        )

        # Training loop
        LOGGER.info(
            "Starting training at iter %s. Training for %s iterations.",
            cfg.experiment.iter.train.start,
            len(train_loader),
        )
        model_and_trainer.train()
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        start = time.time()
        for i, (x, y) in enumerate(train_loader, start=cfg.experiment.iter.train.start):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                # Forward pass
                results = model_and_trainer(cfg, i, x, y)

                # Compute losses
                losses = loss(cfg, model, results, phase="train")
                total_loss = losses["rec_loss"] + losses["overflow_loss"]

            # Backward pass
            opt.zero_grad()

            scaler.scale(total_loss).backward()

            # Normalize gradients
            # TODO: don't normalize gradients for non-CA models
            with torch.no_grad():
                norm_grad(model)

            # opt.step()
            scaler.step(opt)
            lr_sched.step()
            scaler.update()

            # Add new states to nca pool, shuffle, and retain first {pool_size} states
            model_and_trainer.update_pools(
                cfg,
                results["ground_truth"]["x"],
                results["ground_truth"]["y"],
                results["output_cells"].detach(),
                results["was_sampled"],
            )

            if i % cfg.experiment.save_frequency == 0:
                model_and_trainer.save_checkpoint(cfg, i, save)
            if i == 2000:
                break

        LOGGER.info(f"Last: {time.time() - start}")
        return model_and_trainer.best_avg_val_rec_err


if __name__ == "__main__":
    train = Train()
    app = hydra.main(config_path="masked_autoencoding/conf", config_name="train_config")(train.__call__)
    app()

