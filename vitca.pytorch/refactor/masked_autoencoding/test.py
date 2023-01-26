"""
Main program
"""
import os
from pathlib import Path
import random
import logging
import time

import dill

import numpy as np
from sklearn.decomposition import PCA

from einops import rearrange

from torchvision.utils import make_grid

import submitit

import cloudpickle
import src

# otherwise pickle.load from submitit launcher won't find src since cloudpickle registers by reference by default
cloudpickle.register_pickle_by_value(src)

from masked_autoencoding.src.utils import py2pil
from masked_autoencoding.src.datasets import get_inf_dataset
from masked_autoencoding.src.model_and_trainer import ModelAndTrainer

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="test_config")
def test(cfg: DictConfig) -> float:
    try:
        env = submitit.JobEnvironment()
        LOGGER.info(
            "Process ID %s executing test experiment %s, with %s.",
            os.getpid(),
            cfg.experiment.name,
            env,
        )
    except RuntimeError:
        LOGGER.info(
            "Process ID %s executing test experiment %s",
            os.getpid(),
            cfg.experiment.name,
        )

    save = Path(os.getcwd())

    # For saving images produced during test
    imgs_save_folder = save / "imgs"
    imgs_save_folder.mkdir(exist_ok=True)

    # Retrieving the path to the pretrained model checkpoint
    pretrained_model_path = cfg.experiment.pretrained_model_path.get(
        str(cfg.dataset.name).lower(), None
    )

    LOGGER.info("Current working/save directory: %s", save)

    # Merging base config with pretrained config, overwriting cfg params with
    # pretrained_cfg params when there are matches (overwrite priority goes right-to-left)
    if cfg.use_pretrained_cfg:
        LOGGER.info("Merging config used to train pretrained model...")

        # Loading pretrained config
        pretrained_cfg = Path(pretrained_model_path).parent / ".hydra/config.yaml"
        pretrained_cfg = OmegaConf.load(str(pretrained_cfg))

        # Pretrained configs to use
        masking_val_cfg = OmegaConf.select(pretrained_cfg, "experiment.masking.val")
        loss_cfg = OmegaConf.select(pretrained_cfg, "experiment.trainer.loss")
        model_cfg = OmegaConf.select(pretrained_cfg, "model")
        dataset_cfg = OmegaConf.select(pretrained_cfg, "dataset")
        input_size_cfg = OmegaConf.select(pretrained_cfg, "experiment.input_size.train")

        # Remove deprecated keys
        loss_cfg.pop("jac_factor", None)
        model_cfg.pop("head_dim", None)

        # Remove old targets
        loss_cfg.pop("_target_", None)
        model_cfg.pop("_target_", None)

        # Merge/replace with pretrained config
        with open_dict(cfg):
            OmegaConf.update(
                cfg, "experiment.masking.val", masking_val_cfg, merge=False
            )
            if cfg.use_pretrained_size:
                OmegaConf.update(
                    cfg, "experiment.input_size.test", input_size_cfg, merge=False
                )
        OmegaConf.update(cfg, "experiment.trainer.loss", loss_cfg)
        OmegaConf.update(cfg, "model", model_cfg)
        OmegaConf.update(cfg, "dataset", dataset_cfg, merge=False)

        # Overwrite old config with merged one
        cfg_path = save / ".hydra" / "config.yaml"
        LOGGER.info(f"Overwriting old config with merged config at {cfg_path}...")
        OmegaConf.save(cfg, cfg_path, resolve=True)

    # Lazy importing for pickling purposes with submitit and cloudpickle so that cuda.deterministic=True and
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
        device = torch.device(cfg.experiment.device)
        devices = list(range(torch.cuda.device_count()))
        devices_properties = [
            torch.cuda.get_device_properties(torch.device(f"cuda:{d}")) for d in devices
        ]
        LOGGER.info("CUDA available.")
        for i in range(len(devices)):
            LOGGER.info("Using GPU/device %s: %s", i, devices_properties[i])
        mode = "gpu"
    else:
        device = torch.device("cpu")
        LOGGER.info("CUDA not available. Using CPU mode.")
        mode = "cpu"

    # Setup model
    LOGGER.info("Setting up model...")
    model_and_trainer = ModelAndTrainer(cfg, device, phase="test")
    model = model_and_trainer.model
    loss = model_and_trainer.loss

    # Instantiate LPIPS manually
    loss.inst_lpips(device)

    # Load user-defined checkpoint for pre-trained model
    if not model_and_trainer.load_pretrained_model(cfg, ckpt_pth=pretrained_model_path):
        raise RuntimeError("Pretrained model checkpoint must be provided.")

    # Log model and number of params
    LOGGER.info("Using model: %s", model)
    num_train_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    num_nontrain_params = sum(
        param.numel() for param in model.parameters() if not param.requires_grad
    )
    LOGGER.info("Trainable params: %s", num_train_params)
    LOGGER.info("Untrainable params: %s", num_nontrain_params)
    LOGGER.info("Total number of params: %s", num_train_params + num_nontrain_params)

    # Setup dataset and dataloader
    input_size = list(cfg.experiment.input_size.test)
    batch_size = cfg.experiment.batch_size.test
    attn_size = getattr(
        model, "localized_attn_neighbourhood", list(cfg.experiment.attn_size.test)
    )
    LOGGER.info(
        "Testing on dataset: %s, batch size: %s, image size: %s, attn size: %s",
        cfg.dataset.name,
        batch_size,
        input_size,
        attn_size,
    )
    dataset = get_inf_dataset(cfg.dataset.name, cfg.dataset.dataset_root, input_size)
    gpuargs = (
        {
            "num_workers": cfg.experiment.num_workers,
            "pin_memory": True,
            "drop_last": True,
        }
        if mode == "gpu"
        else {}
    )  # NOTE: drop_last should be False to use all test data
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **gpuargs)

    # Initialize final results dictionary
    final_results = {
        "losses": {
            "l1": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "lpips": 0.0,
            "overflow": 0.0,
        },
        "times": {
            "f-time": 0.0,
        },
        "scale_factors": {
            "l1": loss.rec_factor,
            "overflow": loss.overflow_factor,
        },
    }

    latents = {"data": [], "data_avg": [], "labels": []}

    # PCA for visualizing cell hidden channels in XYZ
    pca = PCA(
        n_components=cfg.pca_components,
        random_state=int(cfg.experiment.random_seed),
        svd_solver="arpack",
    )

    # Test loop
    model_and_trainer.eval()
    num_batches = len(loader)
    i = 1
    for i, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            # Forward pass
            forward_start = time.perf_counter()
            results = model_and_trainer(cfg, i, x, y, phase="test")
            forward_time = time.perf_counter() - forward_start

            # Compute losses
            losses = loss(cfg, model, results, phase="test")

            # Accumulate them
            final_results["losses"]["l1"] += losses["rec_loss"]
            final_results["losses"]["psnr"] += losses["psnr"]
            final_results["losses"]["ssim"] += losses["ssim"]
            final_results["losses"]["lpips"] += losses["lpips"]
            final_results["losses"]["overflow"] += losses["overflow_loss"]
            final_results["times"]["f-time"] += forward_time

            # Save images
            if i % cfg.experiment.save_frequency == 0:
                masked_in = results.pop("masked_input", None)
                out = results.pop("output_img", None)
                gt = results.pop("ground_truth", None)["x"]

                if all(images is not None for images in [masked_in, out, gt]):
                    # log input images, output images and ground truth
                    imgs = torch.cat([masked_in, out, gt], 0)
                    imgs = make_grid(imgs, nrow=x.shape[0], pad_value=1)
                    try:
                        img_path = str(imgs_save_folder / f"{i:04}.png")
                        LOGGER.debug(f"Writing image to disk: {img_path}")
                        py2pil(imgs).save(img_path)
                    except OSError:
                        LOGGER.exception("Can not save images.")

            # Accumulate cell hidden representation
            if cfg.save_pca_data and "CA" in cfg.model._target_:
                if cfg.dataset.name in ["MNIST", "FashionMNIST", "CIFAR10"]:
                    hidden = model.get_hidden(results["output_cells"])
                    hidden_avg = torch.mean(hidden, dim=(-2, -1))
                    hidden = rearrange(hidden, "b hidden h w -> b (h w hidden)")
                    latents["data"].append(hidden.cpu().numpy())
                    latents["data_avg"].append(hidden_avg.cpu().numpy())
                    latents["labels"].append(y.cpu().numpy())

        if i % cfg.experiment.log_frequency == 0:
            # Logging val results to logger
            LOGGER.info(
                f"Test results at val iter {i:04}/{num_batches:04}: "
                f'L1 loss: {losses["rec_loss"]:.10f}, scale factor: {loss.rec_factor}, '
                f'PSNR: {losses["psnr"]:.10f}, '
                f'SSIM: {losses["ssim"]:.10f}, '
                f'LPIPS: {losses["lpips"]:.10f}, '
                f'overflow loss: {losses["overflow_loss"]:.10f}, scale factor: {loss.overflow_factor}, '
                f"f-time: {forward_time}s"
            )

    # Average metrics over test set
    for result_type in final_results:
        if result_type == "scale_factors" or result_type == "other":
            continue
        for metric in final_results[result_type]:
            final_results[result_type][metric] /= float(i)
            if torch.is_tensor(final_results[result_type][metric]):
                final_results[result_type][metric] = final_results[result_type][
                    metric
                ].item()

    # Gather average results
    l1_loss = final_results["losses"]["l1"]
    l1_scale_factor = final_results["scale_factors"]["l1"]
    psnr = final_results["losses"]["psnr"]
    ssim = final_results["losses"]["ssim"]
    lpips = final_results["losses"]["lpips"]
    overflow_loss = final_results["losses"]["overflow"]
    overflow_scale_factor = final_results["scale_factors"]["overflow"]
    f_time = final_results["times"]["f-time"]

    # Log averaged results
    LOGGER.info(
        f"Avg L1 loss: {l1_loss}, scale factor: {l1_scale_factor}, "
        f"avg PSNR: {psnr}, "
        f"avg SSIM: {ssim}, "
        f"avg LPIPS: {lpips}, "
        f"avg overflow loss: {overflow_loss}, scale factor: {overflow_scale_factor}, "
        f"avg f-time: {f_time}s"
    )

    # Serialize averaged results along with config
    results_to_serialize = OmegaConf.masked_copy(
        OmegaConf.create(final_results), ["losses", "times"]
    )
    results_to_serialize = OmegaConf.merge(results_to_serialize, cfg)
    with open_dict(results_to_serialize):
        results_to_serialize.output_imgs_path = str(imgs_save_folder)
    res_path = save / "results.yaml"
    LOGGER.info(f"Saving results to {res_path}.")
    OmegaConf.save(results_to_serialize, res_path, resolve=True)

    # Save pca results
    if (
        cfg.dataset.name in ["MNIST", "FashionMNIST", "CIFAR10"]
        and len(latents["data"]) > 0
        and cfg.save_pca_data
    ):
        # Retrieve data and labels
        data = np.concatenate(latents["data"])  # N (h w hidden)
        data_avg = np.concatenate(latents["data_avg"])  # N (h w 1)
        labels = np.concatenate(latents["labels"])  # N

        # Zero-centre data and PCA dim reduce
        mean = np.mean(data, 0)
        data = data - mean
        data = pca.fit_transform(data)  # N n-components
        mean_avg = np.mean(data_avg, 0)
        data_avg = data_avg - mean_avg
        data_avg = pca.fit_transform(data_avg)  # N n-components

        # Saving pca data
        torch.save(
            {"data": data, "data_avg": data_avg, "labels": labels},
            str(save / "pca.pth.tar"),
            pickle_module=dill,
        )
    elif (
        cfg.dataset.name not in ["MNIST", "FashionMNIST", "CIFAR10"]
        and cfg.save_pca_data
    ):
        LOGGER.warn(
            "Can only plot pca results when MNIST, FashionMNIST, or CIFAR10 dataset is chosen. "
            "Skipping plotting."
        )

    return final_results["losses"]["psnr"]


if __name__ == "__main__":
    test()
