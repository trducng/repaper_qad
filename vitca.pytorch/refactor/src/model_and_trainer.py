import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
import __init__
from src.utils import masking_schedule
from src.datasets import random_mask
import logging
import re
import dill  # so that torch.save can save lambda functions
import numpy as np
import random
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class ModelAndTrainer(torch.nn.Module):
    def __init__(self, cfg, device, phase="train"):
        super(ModelAndTrainer, self).__init__()
        self.device = device
        self.model = self.model_class(cfg, device, phase)

        if phase == "train":
            # Replacing CA-based configs with non-CA equivalents
            if "CA" not in cfg.model._target_:
                with open_dict(cfg.experiment.trainer.loss):
                    cfg.experiment.trainer.loss.pop("_target_", None)
                    OmegaConf.update(
                        cfg.experiment.trainer.loss,
                        "_target_",
                        "src.losses.Loss",
                    )

            self.loss = instantiate(cfg.experiment.trainer.loss).to(device)
            self.opt = instantiate(
                cfg.experiment.trainer.opt, params=self.model.parameters()
            )
            self.lr_sched = instantiate(
                cfg.experiment.trainer.lr_sched, optimizer=self.opt
            )
        elif phase == "test":
            # Replacing CA-based configs with non-CA equivalents
            if "CA" not in cfg.model._target_:
                with open_dict(cfg.experiment.trainer.loss):
                    cfg.experiment.trainer.loss.pop("_target_", None)
                    OmegaConf.update(
                        cfg.experiment.trainer.loss,
                        "_target_",
                        "src.losses.Loss",
                    )

            self.loss = instantiate(cfg.experiment.trainer.loss).to(device)
        elif phase == "inference":
            # Replacing CA-based configs with non-CA equivalents
            if "CA" not in cfg.model._target_:
                with open_dict(cfg.experiment.trainer.loss):
                    cfg.experiment.trainer.loss.pop("_target_", None)
                    OmegaConf.update(
                        cfg.experiment.trainer.loss,
                        "_target_",
                        "src.losses.Loss",
                    )

            self.loss = instantiate(cfg.experiment.trainer.loss).to(device)

        self.train_losses = {
            "reconstruction_loss": [],
            "reconstruction_factor": [],
            "overflow_loss": [],
            "overflow_factor": [],
            "total_loss": [],
        }
        self.avg_val_losses = {
            "reconstruction_loss": [],
            "reconstruction_factor": [],
            "overflow_loss": [],
            "overflow_factor": [],
            "total_loss": [],
            "psnr": [],
        }
        self.best_avg_val_rec_err = 1e8

    def model_class(self, cfg, device, phase="train"):
        if cfg.model._target_ == "src.models.vitca.ViTCA":
            if phase == "train":
                input_size = list(cfg.experiment.input_size.train)
            elif phase == "test":
                input_size = list(cfg.experiment.input_size.test)
            elif phase == "inference":
                input_size = list(cfg.experiment.input_size.inf)

            patch_size = cfg.model.patch_size
            grid_size = (input_size[0] // patch_size, input_size[1] // patch_size)
            num_patches = grid_size[0] * grid_size[1]

            model = instantiate(cfg.model, num_patches=num_patches, device=device).to(
                device
            )

            self.x_pool = []
            self.y_pool = []
            self.z_pool = []
        elif cfg.model._target_ == "src.models.vitca_dynattn.ViTCA":
            if phase == "train":
                input_size = list(cfg.experiment.input_size.train)
            elif phase == "test":
                input_size = list(cfg.experiment.input_size.test)
            elif phase == "inference":
                input_size = list(cfg.experiment.input_size.inf)

            patch_size = cfg.model.patch_size
            grid_size = (input_size[0] // patch_size, input_size[1] // patch_size)
            num_patches = grid_size[0] * grid_size[1]

            model = instantiate(cfg.model, num_patches=num_patches, device=device).to(
                device
            )

            self.x_pool = []
            self.y_pool = []
            self.z_pool = []
        elif cfg.model._target_ == "src.models.vitca_deq.ViTCA":
            if phase == "train":
                input_size = list(cfg.experiment.input_size.train)
            elif phase == "test":
                input_size = list(cfg.experiment.input_size.test)
            elif phase == "inference":
                input_size = list(cfg.experiment.input_size.inf)

            patch_size = cfg.model.patch_size
            grid_size = (input_size[0] // patch_size, input_size[1] // patch_size)
            num_patches = grid_size[0] * grid_size[1]

            model = instantiate(cfg.model, num_patches=num_patches, device=device).to(
                device
            )

            self.x_pool = []
            self.y_pool = []
            self.z_pool = []
        elif cfg.model._target_ == "src.models.vit.ViT":
            if phase == "train":
                input_size = list(cfg.experiment.input_size.train)
            elif phase == "test":
                input_size = list(cfg.experiment.input_size.test)
            elif phase == "inference":
                input_size = list(cfg.experiment.input_size.inf)

            patch_size = cfg.model.patch_size
            grid_size = (input_size[0] // patch_size, input_size[1] // patch_size)
            num_patches = grid_size[0] * grid_size[1]

            # Replacing CA-based configs with non-CA equivalents
            if "cell_in_chns" in cfg.model and "cell_out_chns" in cfg.model:
                with open_dict(cfg.model):
                    in_chns = cfg.model.pop("cell_in_chns", None)
                    out_chns = cfg.model.pop("cell_out_chns", None)
                    OmegaConf.update(cfg.model, "in_chns", in_chns)
                    OmegaConf.update(cfg.model, "out_chns", out_chns)

            model = instantiate(cfg.model, num_patches=num_patches, device=device).to(
                device
            )
        elif cfg.model._target_ == "src.models.tnca.TNCA":
            model = instantiate(cfg.model, device=device).to(device)

            self.x_pool = []
            self.y_pool = []
            self.z_pool = []
        elif cfg.model._target_ == "src.models.unetca.UNetCA":
            model = instantiate(cfg.model, device=device).to(device)

            self.x_pool = []
            self.y_pool = []
            self.z_pool = []
        elif cfg.model._target_ == "src.models.unet.UNet":
            # Replacing CA-based configs with non-CA equivalents
            if "cell_in_chns" in cfg.model and "cell_out_chns" in cfg.model:
                with open_dict(cfg.model):
                    in_chns = cfg.model.pop("cell_in_chns", None)
                    out_chns = cfg.model.pop("cell_out_chns", None)
                    OmegaConf.update(cfg.model, "in_chns", in_chns)
                    OmegaConf.update(cfg.model, "out_chns", out_chns)

            model = instantiate(cfg.model, device=device).to(device)
        return model

    def load_pretrained_model(self, cfg, ckpt_pth=None):
        pretrained_model_path = cfg.experiment.pretrained_model_path.get(
            str(cfg.dataset.name).lower(), None
        )
        if pretrained_model_path is None and ckpt_pth is None:
            LOGGER.info("Checkpoint not defined.")
            return False
        if ckpt_pth is not None:
            ckpt_pth = Path(ckpt_pth)
        else:
            ckpt_pth = Path(pretrained_model_path)
        if ckpt_pth.is_file():
            LOGGER.info("Loading pretrained model checkpoint at '%s'...", ckpt_pth)
            checkpoint = torch.load(str(ckpt_pth), pickle_module=dill)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            LOGGER.info(
                "Loaded pretrained model checkpoint '%s' (at iter %s)",
                ckpt_pth,
                checkpoint["iter"],
            )
            return True
        else:
            LOGGER.info("Checkpoint '%s' not found.", ckpt_pth)
            return False

    def load_latest_checkpoint(self, cfg, ckpt_dirname):
        # Load latest checkpoint in save dir (cwd) if requested, otherwise randomly initialize
        # check for ckpt_{%d}.pth.tar in current working save dir and pick largest number
        # meant for dealing with preemption and timeouts
        chk_num = lambda pth: int(
            re.search(r".*/ckpt_([0-9]+).pth.tar$", str(pth)).group(1)
        )
        chk_pths = sorted(ckpt_dirname.glob("ckpt_[0-9]*.pth.tar"), key=chk_num)
        if bool(chk_pths) and chk_pths[-1].is_file():
            latest_chk_pth = chk_pths[-1]
        else:
            latest_chk_pth = None
        if cfg.experiment.resume_from_latest and latest_chk_pth is not None:
            LOGGER.info(
                "Found checkpoint '%s', loading checkpoint and resuming...",
                latest_chk_pth,
            )
            checkpoint = torch.load(str(latest_chk_pth), pickle_module=dill)
            cfg.experiment.iter.train.start = checkpoint["iter"] + 1
            self.opt.load_state_dict(checkpoint["opt"])
            self.lr_sched.load_state_dict(checkpoint["lr_sched"])
            self.train_losses = checkpoint["train_losses"]
            self.avg_val_losses = checkpoint["avg_val_losses"]
            self.best_avg_val_rec_err = checkpoint["best_avg_val_rec_err"]
            if "CA" in cfg.model._target_:
                self.x_pool = checkpoint["pools"]["x"]
                self.y_pool = checkpoint["pools"]["y"]
                self.z_pool = checkpoint["pools"]["z"]
            self.model.load_state_dict(checkpoint["state_dict"])
            LOGGER.info(
                "Loaded checkpoint '%s' (at iter %s)",
                latest_chk_pth,
                checkpoint["iter"],
            )
        elif cfg.experiment.resume_from_latest:
            LOGGER.info(
                "No checkpoints found at '%s' (not including ckpt_best). Randomly initializing model.",
                ckpt_dirname,
            )
        else:
            LOGGER.info("Randomly initializing model.")

    def save_checkpoint(self, cfg, i, ckpt_dirname):
        chk_fname = ckpt_dirname / f"ckpt_{i}.pth.tar"
        LOGGER.info("Saving checkpoint at iter %s at %s.", i, chk_fname)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "iter": i,
                "opt": self.opt.state_dict(),
                "lr_sched": self.lr_sched.state_dict(),
                "last_lr": self.opt.param_groups[0]["lr"],
                "train_losses": self.train_losses,
                "avg_val_losses": self.avg_val_losses,
                "best_avg_val_rec_err": self.best_avg_val_rec_err,
                "pools": {"x": self.x_pool, "y": self.y_pool, "z": self.z_pool}
                if "CA" in cfg.model._target_
                else None,
            },
            chk_fname,
            pickle_module=dill,
        )

    def forward(self, cfg, step, x, y, phase="train"):
        if phase == "train":
            if "CA" in cfg.model._target_:
                return self.ca_train_forward(cfg, step, x, y)
            else:
                return self.reg_train_forward(cfg, step, x, y)
        elif phase == "validation":
            if "CA" in cfg.model._target_:
                return self.ca_val_forward(cfg, step, x, y)
            else:
                return self.reg_val_forward(cfg, step, x, y)
        elif phase == "test":
            if "CA" in cfg.model._target_:
                return self.ca_test_forward(cfg, step, x, y)
            else:
                return self.reg_test_forward(cfg, step, x, y)
        elif phase == "inference":
            if "CA" in cfg.model._target_:
                return self.ca_inf_forward(cfg, step, x, y)
            else:
                return self.reg_inf_forward(cfg, step, x, y)

    """
	CA-centric forward functions
	"""

    def ca_train_forward(self, cfg, step, x, y):
        train_size = list(cfg.experiment.input_size.train)
        train_batch_size = cfg.experiment.batch_size.train
        was_sampled = False
        if len(self.z_pool) > train_batch_size and step % 2 == 0:
            # sample from the nca pool, which includes cell states and associated ground truths, every 2nd iter
            x = torch.cat(self.x_pool[:train_batch_size]).to(self.device)
            y = torch.cat(self.y_pool[:train_batch_size]).to(self.device)
            with torch.no_grad():
                z_0 = torch.cat(self.z_pool[:train_batch_size]).to(self.device)
            was_sampled = True
        else:
            x, y = x.to(self.device), y.to(self.device)
            p, p_s = masking_schedule(
                step,
                schedule_start=cfg.experiment.masking.train.schedule_start,
                schedule_end=cfg.experiment.masking.train.schedule_end,
                max_prob=cfg.experiment.masking.train.max_prob,
                prob_stages=cfg.experiment.masking.train.prob_stages,
                max_patch_shape=cfg.experiment.masking.train.max_patch_shape,
                patch_shape_stages=cfg.experiment.masking.train.patch_shape_stages,
            )
            masked_x = random_mask(
                x,
                p=p,
                mask_type=cfg.experiment.masking.train.type,
                patch_shape=p_s,
                device=self.device,
            )
            with torch.no_grad():
                z_0 = self.model.seed(rgb_in=masked_x, sz=train_size)

        # iterate nca
        T = np.random.randint(
            cfg.experiment.iter.train.ca.min, cfg.experiment.iter.train.ca.max + 1
        )
        if cfg.experiment.trainer.checkpointing.enabled:
            segs = T // 2 if T > 1 else 1
            if segs > 1:
                z_0.requires_grad_(True)  # an in-place operation
        else:
            segs = 1
        z_T = self.model(
            z_0,
            step_n=T,
            update_rate=cfg.experiment.iter.train.ca.update_rate,
            chkpt_segments=segs,
            attn_size=list(cfg.experiment.attn_size.train),
        )

        return {
            "output_cells": z_T,
            "output_img": self.model.get_rgb_out(z_T),
            "ground_truth": {"x": x, "y": y},
            "was_sampled": was_sampled,
        }

    def ca_val_forward(self, cfg, step, x, y):
        validation_size = list(cfg.experiment.input_size.val)

        # use all available kinds of patch shapes and probs
        # TODO: come up with random seed that's not dependent on step since val batch size can affect it
        p, p_s = masking_schedule(
            -1,
            max_prob=cfg.experiment.masking.val.max_prob,
            prob_stages=cfg.experiment.masking.val.prob_stages,
            max_patch_shape=cfg.experiment.masking.val.max_patch_shape,
            patch_shape_stages=cfg.experiment.masking.val.patch_shape_stages,
            random_seed=step,
        )
        masked_x = random_mask(
            x,
            p=p,
            mask_type=cfg.experiment.masking.val.type,
            patch_shape=p_s,
            random_seed=cfg.experiment.random_seed,
            device=self.model.device,
        )
        z_0 = self.model.seed(rgb_in=masked_x, sz=validation_size)

        # iterate nca
        z_T = self.model(
            z_0,
            step_n=cfg.experiment.iter.val.ca.value,
            update_rate=cfg.experiment.iter.val.ca.update_rate,
            attn_size=list(cfg.experiment.attn_size.val),
        )

        return {
            "output_cells": z_T,
            "masked_input": self.model.get_rgb_in(z_0),
            "output_img": self.model.get_rgb_out(z_T),
            "ground_truth": {"x": x, "y": y},
        }

    def ca_test_forward(self, cfg, step, x, y):
        input_size = list(cfg.experiment.input_size.test)
        if cfg.no_noise:
            masked_x = x
        elif cfg.use_pretrained_cfg:
            # Use all available kinds of patch shapes and probs
            p, p_s = masking_schedule(
                -1,
                max_prob=cfg.experiment.masking.val.max_prob,
                prob_stages=cfg.experiment.masking.val.prob_stages,
                max_patch_shape=cfg.experiment.masking.val.max_patch_shape,
                patch_shape_stages=cfg.experiment.masking.val.patch_shape_stages,
            )
            masked_x = random_mask(
                x,
                p=p,
                mask_type=cfg.experiment.masking.val.type,
                patch_shape=p_s,
                device=self.model.device,
            )
        else:
            # Use all available kinds of patch shapes and probs
            p, p_s = masking_schedule(
                -1,
                max_prob=cfg.experiment.masking.test.max_prob,
                prob_stages=cfg.experiment.masking.test.prob_stages,
                max_patch_shape=cfg.experiment.masking.test.max_patch_shape,
                patch_shape_stages=cfg.experiment.masking.test.patch_shape_stages,
            )
            masked_x = random_mask(
                x,
                p=p,
                mask_type=cfg.experiment.masking.test.type,
                patch_shape=p_s,
                device=self.model.device,
            )

        z_0 = self.model.seed(rgb_in=masked_x, sz=input_size)

        import pdb; pdb.set_trace()
        # iterate nca
        z_T = self.model(
            z_0,
            step_n=cfg.experiment.iter.test.ca.value,
            update_rate=cfg.experiment.iter.test.ca.update_rate,
            attn_size=list(cfg.experiment.attn_size.test),
        )

        return {
            "output_cells": z_T,
            "masked_input": self.model.get_rgb_in(z_0),
            "output_img": self.model.get_rgb_out(z_T),
            "ground_truth": {"x": x, "y": y},
        }

    def ca_inf_forward(self, cfg, step, x, y):
        input_size = list(cfg.experiment.input_size.inf)
        p = cfg.experiment.masking.inf.prob
        p_s = cfg.experiment.masking.inf.patch_shape
        mask_type = cfg.experiment.masking.inf.type
        masked_x = random_mask(
            x, p=p, mask_type=mask_type, patch_shape=p_s, device=self.model.device
        )

        z_0 = self.model.seed(rgb_in=masked_x, sz=input_size)

        # iterate nca
        z_T = self.model(
            z_0,
            step_n=cfg.experiment.iter.inf.ca.value,
            update_rate=cfg.experiment.iter.inf.ca.update_rate,
            attn_size=list(cfg.experiment.attn_size.inf),
        )

        return {
            "output_cells": z_T,
            "masked_input": self.model.get_rgb_in(z_0),
            "output_img": self.model.get_rgb_out(z_T),
            "ground_truth": {"x": x, "y": y},
        }

    """
	Forward functions for non-CA networks
	"""

    def reg_train_forward(self, cfg, step, x, y):
        p, p_s = masking_schedule(
            step,
            schedule_start=cfg.experiment.masking.train.schedule_start,
            schedule_end=cfg.experiment.masking.train.schedule_end,
            max_prob=cfg.experiment.masking.train.max_prob,
            prob_stages=cfg.experiment.masking.train.prob_stages,
            max_patch_shape=cfg.experiment.masking.train.max_patch_shape,
            patch_shape_stages=cfg.experiment.masking.train.patch_shape_stages,
        )
        masked_x = random_mask(
            x,
            p=p,
            mask_type=cfg.experiment.masking.train.type,
            patch_shape=p_s,
            device=self.device,
        )

        # forward pass through model
        out = self.model(masked_x)

        return {
            "masked_input": masked_x,
            "output_img": out,
            "ground_truth": {"x": x, "y": y},
        }

    def reg_val_forward(self, cfg, step, x, y):
        # use all available kinds of patch shapes and probs
        # TODO: come up with random seed that's not dependent on step since val batch size can affect it
        p, p_s = masking_schedule(
            -1,
            max_prob=cfg.experiment.masking.val.max_prob,
            prob_stages=cfg.experiment.masking.val.prob_stages,
            max_patch_shape=cfg.experiment.masking.val.max_patch_shape,
            patch_shape_stages=cfg.experiment.masking.val.patch_shape_stages,
            random_seed=step,
        )
        masked_x = random_mask(
            x,
            p=p,
            mask_type=cfg.experiment.masking.val.type,
            patch_shape=p_s,
            random_seed=cfg.experiment.random_seed,
            device=self.model.device,
        )

        # forward pass through model
        out = self.model(masked_x)

        return {
            "masked_input": masked_x,
            "output_img": out,
            "ground_truth": {"x": x, "y": y},
        }

    def reg_test_forward(self, cfg, step, x, y):
        if cfg.no_noise:
            masked_x = x
        elif cfg.use_pretrained_cfg:
            # Use all available kinds of patch shapes and probs
            p, p_s = masking_schedule(
                -1,
                max_prob=cfg.experiment.masking.val.max_prob,
                prob_stages=cfg.experiment.masking.val.prob_stages,
                max_patch_shape=cfg.experiment.masking.val.max_patch_shape,
                patch_shape_stages=cfg.experiment.masking.val.patch_shape_stages,
            )
            masked_x = random_mask(
                x,
                p=p,
                mask_type=cfg.experiment.masking.val.type,
                patch_shape=p_s,
                device=self.model.device,
            )
        else:
            # Use all available kinds of patch shapes and probs
            p, p_s = masking_schedule(
                -1,
                max_prob=cfg.experiment.masking.test.max_prob,
                prob_stages=cfg.experiment.masking.test.prob_stages,
                max_patch_shape=cfg.experiment.masking.test.max_patch_shape,
                patch_shape_stages=cfg.experiment.masking.test.patch_shape_stages,
            )
            masked_x = random_mask(
                x,
                p=p,
                mask_type=cfg.experiment.masking.test.type,
                patch_shape=p_s,
                device=self.model.device,
            )

        # forward pass through model
        out = self.model(masked_x)

        return {
            "masked_input": masked_x,
            "output_img": out,
            "ground_truth": {"x": x, "y": y},
        }

    def reg_inf_forward(self, cfg, step, x, y):
        p = cfg.experiment.masking.inf.prob
        p_s = cfg.experiment.masking.inf.patch_shape
        mask_type = cfg.experiment.masking.inf.type
        masked_x = random_mask(
            x, p=p, mask_type=mask_type, patch_shape=p_s, device=self.model.device
        )

        # forward pass through model
        out = self.model(masked_x)

        return {
            "masked_input": masked_x,
            "output_img": out,
            "ground_truth": {"x": x, "y": y},
        }

    def update_pools(self, cfg, x, y, z_T, was_sampled):
        """
        If states were newly created, add new states to nca pool, shuffle, and retain first {pool_size} states.
        If states were sampled from the pool, replace their old states, shuffle, and retain first {pool_size}
        states.
        """
        pool_size = cfg.experiment.pool_size
        if was_sampled and cfg.experiment.sample_with_replacement:
            train_batch_size = cfg.experiment.batch_size.train
            self.x_pool[:train_batch_size] = torch.split(x, 1)
            self.y_pool[:train_batch_size] = torch.split(y, 1)
            self.z_pool[:train_batch_size] = torch.split(z_T, 1)
        else:
            self.x_pool += list(torch.split(x, 1))
            self.y_pool += list(torch.split(y, 1))
            self.z_pool += list(torch.split(z_T, 1))
        pools = list(zip(self.x_pool, self.y_pool, self.z_pool))
        random.shuffle(pools)
        self.x_pool, self.y_pool, self.z_pool = zip(*pools)
        self.x_pool = list(self.x_pool[:pool_size])
        self.y_pool = list(self.y_pool[:pool_size])
        self.z_pool = list(self.z_pool[:pool_size])

    def update_tracked_scalars(self, scalars, step, phase="train"):
        if phase == "train":
            for scalar_name, scalar in scalars.items():
                self.train_losses[f"{scalar_name}"].append((step, scalar))
        elif phase == "validation":
            for scalar_name, scalar in scalars.items():
                self.avg_val_losses[f"{scalar_name}"].append((step, scalar))
