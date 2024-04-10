"""
Main program
"""
import os
from pathlib import Path
import random
import math
import logging
from multiprocessing import Pool

import numpy as np
from sklearn.decomposition import PCA
from einops import rearrange

from torchvision.utils import make_grid
import torch.fft as fft

from tqdm import tqdm, trange

import cloudpickle
import src

# otherwise pickle.load from submitit launcher won't find src since cloudpickle registers by reference by default
cloudpickle.register_pickle_by_value(src)

from masked_autoencoding.src.utils import (
    to_nhwc,
    draw_msg,
    VideoWriter,
    viz_attn_maps,
    imwrite,
)
from masked_autoencoding.src.datasets import get_inf_dataset, random_mask
from masked_autoencoding.src.model_and_trainer import ModelAndTrainer

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

from dawnet.data.image import show_images


LOGGER = logging.getLogger(__name__)

def record_cells(image_id, k, cells):
    show_images(
        cells[image_id],
        show=False,
        output=f"/home/john/repaper_qad/vitca.pytorch/reproduce/code/logs/celeba_small/cells/{image_id:02d}-{k:03d}.png"
    )


@hydra.main(config_path="conf", config_name="inference_config")
def inference(cfg: DictConfig) -> None:
    LOGGER.info(
        "Process ID %s executing inference experiment %s",
        os.getpid(),
        cfg.experiment.name,
    )

    save = Path(os.getcwd())

    # For saving videos and images generated during inference
    vid_save_folder = save / "vids"
    vid_save_folder.mkdir(exist_ok=True)
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
        masking_val_type_cfg = OmegaConf.select(
            pretrained_cfg, "experiment.masking.val.type"
        )
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
                cfg, "experiment.masking.inf.type", masking_val_type_cfg, merge=False
            )
            if cfg.use_pretrained_size:
                OmegaConf.update(
                    cfg, "experiment.input_size.inf", input_size_cfg, merge=False
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
    model_and_trainer = ModelAndTrainer(cfg, device, phase="inference")
    model = model_and_trainer.model

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
    input_size = list(cfg.experiment.input_size.inf)
    batch_size = cfg.experiment.batch_size.inf
    attn_size = getattr(
        model, "localized_attn_neighbourhood", list(cfg.experiment.attn_size.inf)
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
        {"num_workers": cfg.experiment.num_workers, "pin_memory": True}
        if mode == "gpu"
        else {}
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, **gpuargs
    )

    # PCA for visualizing cell hidden channels in RGB
    pca = PCA(
        n_components=3,
        random_state=int(cfg.experiment.random_seed),
        svd_solver="arpack",
    )

    # Inference loop
    LOGGER.info("Inferencing for %s iterations.", cfg.experiment.iter.inf.total)
    model.eval()
    if "CA" not in cfg.model._target_:
        for i, (x, _) in enumerate(loader, start=1):
            # Retrieve data
            x = x.to(device)

            # Create masked input
            masked_x = random_mask(
                x,
                p=cfg.experiment.masking.inf.prob,
                mask_type=cfg.experiment.masking.inf.type,
                patch_shape=cfg.experiment.masking.inf.patch_shape,
                random_seed=cfg.experiment.random_seed,
                device=device,
            )

            if cfg.completely_mask_input:
                masked_x *= 0.0

            # Feed forward
            output_img = model(masked_x)

            # Retrieving masked input imgs, output imgs, and ground truths. Also converting to RGB if Grayscale.
            if masked_x.shape[1] == 1:
                masked_x = masked_x.tile(1, 3, 1, 1)  # b 3 h w
            if output_img.shape[1] == 1:
                output_img = output_img.tile(1, 3, 1, 1)  # b 3 h w
            ground_truth = x  # b (1 or 3) h w
            if ground_truth.shape[1] == 1:
                ground_truth = ground_truth.tile(1, 3, 1, 1)  # b 3 h w

            # Making image grid of above images
            img = torch.cat(
                [
                    make_grid(masked_x, nrow=batch_size, pad_value=1),
                    make_grid(output_img, nrow=batch_size, pad_value=1),
                    make_grid(ground_truth, nrow=batch_size, pad_value=1),
                ],
                1,
            )

            # Save image to disk
            img_fname = str(imgs_save_folder / f"output_{i:03}.png")
            imwrite(img_fname, img)

            if i == cfg.experiment.iter.inf.total:
                break

    elif not cfg.switch_up.enabled:
        pe_dim = getattr(
            model, "cell_pe_patch_dim", getattr(model, "cell_pe_dim", None)
        )
        in_dim = getattr(
            model, "cell_in_patch_dim", getattr(model, "cell_in_dim", None)
        )
        out_dim = getattr(
            model, "cell_out_patch_dim", getattr(model, "cell_out_dim", None)
        )
        for i, (x, _) in enumerate(loader, start=1):
            # Retrieve data
            x = x.to(device)

            # For saving images to disk
            all_output_imgs = []
            all_attn_maps = []
            all_other_imgs = []

            saved_so_far = 0

            vid_fname = str(vid_save_folder / f"output_{i:03}.mp4")
            with VideoWriter(filename=vid_fname, fps=cfg.fps) as vid, torch.no_grad():

                # Create masked input. Re-use same random seed for each frame so noise pattern doesn't change between
                # frames
                masked_x = random_mask(
                    x,
                    p=cfg.experiment.masking.inf.prob,
                    mask_type=cfg.experiment.masking.inf.type,
                    patch_shape=cfg.experiment.masking.inf.patch_shape,
                    random_seed=cfg.experiment.random_seed,
                    device=device,
                )

                if cfg.completely_mask_input:
                    masked_x *= 0.0

                # Create seed cell grid with injected masked input
                cells = model.seed(rgb_in=masked_x, sz=input_size) # JODO: monitor `cells` variable

                iters_so_far = 0

                for k in trange(cfg.frames, desc="Inference..."):

                    # Damage a (H/2)x(W/2) patch of the cell grid (cell output and hidden channels)
                    if cfg.damage.enabled:  # JODO: what if we enable this?

                        # Only do damage if below conditions are met. Can do damage at random frames or user-specified
                        # frames
                        if cfg.damage.random.enabled:
                            do_damage = (
                                math.floor(random.random() + cfg.damage.random.rate)
                                == 1
                                and k >= cfg.damage.random.start_at
                                and k < cfg.damage.random.stop_at
                            )
                        elif k in list(cfg.damage.at_frames):
                            do_damage = True
                        else:
                            do_damage = False

                        if do_damage:
                            # Damage consists of values uniformly dist. between [-1,1]
                            dmg = (
                                torch.rand(
                                    batch_size,
                                    model.cell_update_dim,
                                    cells.shape[-2] // 2,
                                    cells.shape[-1] // 2,
                                    dtype=cells.dtype,
                                    device=cells.device,
                                )
                                * 2
                                - 1
                            )

                            # Setting center 2x2 patch of noise to white for ease of seeing where it is
                            # dmg[:, :out_dim,
                            # 	(dmg.shape[-2]-1)//2-2:(dmg.shape[-2]-1)//2+2,
                            # 	(dmg.shape[-1]-1)//2-2:(dmg.shape[-1]-1)//2+2] = 1

                            # Making sure the patch is fully within the cell grid
                            start_i = random.randint(0, cells.shape[-2] // 2 - 1)
                            start_j = random.randint(0, cells.shape[-1] // 2 - 1)
                            end_i, end_j = (
                                start_i + dmg.shape[-2],
                                start_j + dmg.shape[-1],
                            )

                            # Set cell states in damage region to damage values
                            cells[
                                :, pe_dim + in_dim :, start_i:end_i, start_j:end_j
                            ] = dmg

                    # Iterate nca (only after the first frame since we want to show seed)
                    if k > 0:
                        if cfg.damage.enabled or cfg.custom_growth_rate:
                            if (
                                k == 1
                            ):  # Always force 1 CA iteration for the very first frame
                                T = 1
                            else:
                                T = cfg.experiment.iter.inf.ca.value
                                if T > 1 and k == 2:
                                    T -= 1
                                elif (
                                    k >= cfg.speed_up.start_at and cfg.speed_up.enabled
                                ):
                                    T = (
                                        cfg.experiment.iter.inf.ca.value
                                        * cfg.speed_up.factor
                                    )
                        else:
                            # Rudimentary iteration schedule that accelerates CA iterations from 1 to 16 steps per frame
                            # during the first 121 frames (assuming 30 fps).
                            # # [1 (0<k<30), ..., 2 (k=30), ..., 4 (k=60), ..., 8 (k=90), ..., 16 (k=120), ...]
                            T = min(2 ** (k // cfg.fps), 16)

                        # Iterate
                        if not cfg.head_masking.enabled:
                            cells = model(
                                cells,
                                step_n=T,
                                update_rate=cfg.experiment.iter.inf.ca.update_rate,
                                attn_size=attn_size,
                            )
                        else:
                            cells = model(
                                cells,
                                step_n=T,
                                update_rate=cfg.experiment.iter.inf.ca.update_rate,
                                attn_size=attn_size,
                                mask=cfg.head_masking.heads_to_mask,
                            )
                        iters_so_far += T

                    # _cells = cells.cpu().clone().numpy()
                    # with Pool(2) as p:
                    #     p.starmap(record_cells, [(image_id, k, _cells) for image_id in range(32)])
                    # JODO-ANSWERED: what is the difference between get_rgb_in, get_rgb_out and the cells?
                    # The cells contain both get_rgb_in and get_rgb_out
                    # Retrieving masked input imgs and making RGB if Grayscale
                    masked_x = model.get_rgb_in(cells).clone()  # b (1 or 3) h w
                    if masked_x.shape[1] == 1:
                        masked_x = masked_x.tile(1, 3, 1, 1)  # b 3 h w

                    # Retrieving output imgs and making RGB if Grayscale
                    output_img = model.get_rgb_out(cells).clone()  # b (1 or 3) h w
                    if output_img.shape[1] == 1:
                        output_img = output_img.tile(1, 3, 1, 1)  # b 3 h w

                    # For visualizing the hidden channels using PCA (but retaining spatial dims).
                    #
                    # Since hidden chn values are between [-1,1], they are visualized in RGB (via PCA) through a
                    # "positive" and "negative" image. Positive PCA img contains positive hidden chn values [0,1] and
                    # negative PCA img contains negative hidden chn values flipped to [0,1].
                    #
                    # Despite manually setting PCA random seed and forcing arpack to be used, sometimes the axes flip
                    # which casuses the pos/neg PCA images to sort of colour invert.
                    hidden = model.get_hidden(cells).clone()  # b num_hidden h w
                    b, num_hidden, h, w = hidden.shape
                    hidden = rearrange(hidden, "b hidden h w -> (b h w) hidden")
                    try:
                        hidden_pca = pca.fit_transform(
                            hidden.cpu().numpy()
                        )  # (b h w) 3
                    except Exception:
                        LOGGER.error(
                            "Error with ARPACK. Will use zeros for pca instead."
                        )
                        hidden_pca = np.zeros((b * h * w, 3), dtype=np.float32)
                    hidden_img = rearrange(
                        hidden_pca, "(b h w) c -> b c h w", b=b, h=h, w=w
                    )
                    hidden_img_pos = torch.tensor(
                        hidden_img.copy(), dtype=torch.float32, device=output_img.device
                    )
                    hidden_img_neg = hidden_img_pos * -1.0

                    # Retrieving ground truth imgs and making RGB if Grayscale
                    ground_truth = x  # b (1 or 3) h w
                    if ground_truth.shape[1] == 1:
                        ground_truth = ground_truth.tile(1, 3, 1, 1)  # b 3 h w

                    # Retrieving FFT phase and magnitude
                    fft_img = fft.fftshift(fft.fft2(output_img, norm="ortho"))
                    fft_img_phase = torch.angle(fft_img)
                    fft_img_phase = (
                        fft_img_phase / torch.pi + 1
                    ) / 2  # clamp to [0, 1]
                    fft_img_mag = torch.abs(fft_img)

                    # Converting for display
                    fft_img_phase = fft_img_phase.mean(1, keepdim=True).repeat(
                        1, output_img.shape[1], 1, 1
                    )
                    fft_img_mag = fft_img_mag.mean(1, keepdim=True).repeat(
                        1, output_img.shape[1], 1, 1
                    )

                    # Creating visualized localized attention maps as colour-blended splats about each cell's position
                    # Each cell's local attention map is colour-coded based on the cell's position (e.g., top-left
                    # cell's 3x3 attention map is blue). Overlapping maps are colour-averaged.
                    if "ViT" in cfg.model._target_:
                        if iters_so_far > 0:
                            attn_maps = [
                                viz_attn_maps(
                                    layer[0].fn.attn_maps,
                                    attn_size,
                                    output_img,
                                    blend=cfg.viz_attn_maps.blend_with_output,
                                    brighten=cfg.viz_attn_maps.brighten_factor,
                                )
                                for layer in model.transformer.layers
                            ]
                        else:
                            attn_maps = [
                                torch.zeros(b * cfg.model.heads, 3, h, w, device=device)
                                for _ in model.transformer.layers
                            ]
                        attn_maps_img = draw_msg(
                            make_grid(attn_maps[0], nrow=batch_size, pad_value=1),
                            "layer 1 heads",
                            "prepend",
                        )
                        for j in range(1, len(attn_maps)):
                            _attn_maps_img = draw_msg(
                                make_grid(attn_maps[j], nrow=batch_size, pad_value=1),
                                f"layer {j+1} heads",
                                "prepend",
                            )
                            attn_maps_img = torch.cat(
                                [attn_maps_img, _attn_maps_img], 1
                            )

                    # Making image grid of above images
                    main_img = make_grid(
                        torch.cat([masked_x, output_img, ground_truth], 0),
                        nrow=batch_size,
                        pad_value=1,
                    )
                    main_img = draw_msg(main_img, "input, out, gt", "prepend")
                    fourier_img = make_grid(
                        torch.cat([fft_img_mag, fft_img_phase], 0),
                        nrow=batch_size,
                        pad_value=1,
                    )
                    fourier_img = draw_msg(fourier_img, "fourier analysis", "prepend")
                    hidden_img_pos = make_grid(
                        hidden_img_pos, nrow=batch_size, pad_value=1
                    )
                    hidden_img_pos = draw_msg(
                        hidden_img_pos, f"hidden pos (pca {num_hidden}->3)", "prepend"
                    )
                    hidden_img_neg = make_grid(
                        hidden_img_neg, nrow=batch_size, pad_value=1
                    )
                    hidden_img_neg = draw_msg(
                        hidden_img_neg, f"hidden neg (pca {num_hidden}->3)", "prepend"
                    )
                    if "ViT" in cfg.model._target_:
                        final_img = torch.cat(
                            [
                                main_img,
                                fourier_img,
                                attn_maps_img,
                                hidden_img_pos,
                                hidden_img_neg,
                            ],
                            1,
                        )
                    else:
                        final_img = torch.cat(
                            [main_img, fourier_img, hidden_img_pos, hidden_img_neg], 1
                        )
                    final_img = draw_msg(
                        final_img, f"iteration {iters_so_far}", "append"
                    )
                    final_img = to_nhwc(final_img).squeeze(0).cpu()

                    # Add grid image for current frame to VideoWriter. It will automatically convert to uint8 (0, 255)
                    vid.add(final_img)

                    # Storing inputs and outputs to be used for visualization later
                    if cfg.viz_for_paper.enabled:
                        if (
                            k == 0
                            or iters_so_far in list(cfg.viz_for_paper.ca_iters_to_save)
                            or "all" in list(cfg.viz_for_paper.ca_iters_to_save)
                        ):
                            all_output_imgs.append(output_img)
                            if "ViT" in cfg.model._target_:
                                attn_maps = torch.cat(attn_maps, 0)  # (l heads b) c h w
                                attn_maps = rearrange(
                                    attn_maps,
                                    "(l heads b) c h w -> b c (l heads h) w",
                                    b=batch_size,
                                    l=len(model.transformer.layers),
                                )
                                all_attn_maps.append(attn_maps)
                            if k == 0:
                                all_other_imgs.append(masked_x)
                            elif (
                                iters_so_far
                                == list(cfg.viz_for_paper.ca_iters_to_save)[-1]
                            ):
                                all_other_imgs.append(ground_truth)
                            elif k < cfg.frames - 1:
                                all_other_imgs.append(torch.ones_like(masked_x))
                            elif k == cfg.frames - 1:
                                all_other_imgs.append(ground_truth)
                            # Save to disk. It will automatically convert to uint8 (0, 255).
                            img_fname = str(
                                vid_save_folder / f"output_{i:03}_{k:03}.png"
                            )
                            imwrite(img_fname, final_img)
                            saved_so_far += 1

            # Creating and saving images designed for visualizing for paper
            if cfg.viz_for_paper.enabled:
                for j in trange(batch_size, desc="Saving images in minibatch..."):
                    _all_other_imgs = torch.stack(
                        [all_other_imgs[k][j] for k in range(saved_so_far)], 0
                    )
                    other_img = make_grid(
                        _all_other_imgs, nrow=saved_so_far, pad_value=1
                    )
                    _all_output_imgs = torch.stack(
                        [all_output_imgs[k][j] for k in range(saved_so_far)], 0
                    )
                    output_img = make_grid(
                        _all_output_imgs, nrow=saved_so_far, pad_value=1
                    )
                    if "ViT" in cfg.model._target_:
                        _all_attn_maps = torch.stack(
                            [all_attn_maps[k][j] for k in range(saved_so_far)], 0
                        )
                        attn_img = make_grid(
                            _all_attn_maps, nrow=saved_so_far, pad_value=1
                        )
                        img = torch.cat([other_img, output_img, attn_img], 1)
                    else:
                        img = torch.cat([other_img, output_img], 1)

                    # Save to disk. It will automatically convert to uint8 (0, 255).
                    img_fname = str(imgs_save_folder / f"output_{i:03}_{j:03}.png")
                    imwrite(img_fname, img)
                LOGGER.info(
                    "Inference iter %s/%s: saved images at: %s",
                    i,
                    cfg.experiment.iter.inf.total,
                    imgs_save_folder,
                )

            LOGGER.info(
                "Inference iter %s/%s: output path: %s",
                i,
                cfg.experiment.iter.inf.total,
                vid_fname,
            )
            if i == cfg.experiment.iter.inf.total:
                break

    # TODO: clean up below to avoid code duplication and promote code re-use
    else:  # Keeping the same cell-grid and re-seeding it as we iterate through dataset (aka "switch-up")
        pe_dim = getattr(
            model, "cell_pe_patch_dim", getattr(model, "cell_pe_dim", None)
        )
        in_dim = getattr(
            model, "cell_in_patch_dim", getattr(model, "cell_in_dim", None)
        )
        out_dim = getattr(
            model, "cell_out_patch_dim", getattr(model, "cell_out_dim", None)
        )
        vid_fname = str(vid_save_folder / f"output.mp4")
        with VideoWriter(filename=vid_fname, fps=cfg.fps) as vid, torch.no_grad():

            # Retrieve data
            loader = iter(loader)
            x, _ = next(loader)
            x = x.to(device)

            # For saving images to disk
            all_gt_imgs = []
            all_output_imgs = []
            all_attn_maps = []
            all_input_imgs = []

            # Create masked input. Re-use same random seed for each frame so noise pattern doesn't change between frames
            masked_x = random_mask(
                x,
                p=cfg.experiment.masking.inf.prob,
                mask_type=cfg.experiment.masking.inf.type,
                patch_shape=cfg.experiment.masking.inf.patch_shape,
                random_seed=cfg.experiment.random_seed,
                device=device,
            )

            if cfg.completely_mask_input:
                masked_x *= 0.0

            # Create seed cell grid with injected masked input
            cells = model.seed(rgb_in=masked_x, sz=input_size)

            iters_so_far = 0
            saved_so_far = 0

            for k in trange(cfg.frames, desc="Inference..."):

                # Damage a (H/2)x(W/2) patch of the cell grid (cell output and hidden channels)
                if cfg.damage.enabled:

                    # Only do damage if below conditions are met. Can do damage at random frames or user-specified
                    # frames
                    if cfg.damage.random.enabled:
                        do_damage = (
                            math.floor(random.random() + cfg.damage.random.rate) == 1
                            and k >= cfg.damage.random.start_at
                            and k < cfg.damage.random.stop_at
                        )
                    elif k in list(cfg.damage.at_frames):
                        do_damage = True
                    else:
                        do_damage = False

                    if do_damage:
                        # Damage consists of values uniformly dist. between [-1,1]
                        dmg = (
                            torch.rand(
                                batch_size,
                                model.cell_update_dim,
                                cells.shape[-2] // 2,
                                cells.shape[-1] // 2,
                                dtype=cells.dtype,
                                device=cells.device,
                            )
                            * 2
                            - 1
                        )

                        # Setting center 2x2 patch of noise to white for ease of seeing where it is
                        # dmg[:, :out_dim,
                        # 	(dmg.shape[-2]-1)//2-2:(dmg.shape[-2]-1)//2+2,
                        # 	(dmg.shape[-1]-1)//2-2:(dmg.shape[-1]-1)//2+2] = 1

                        # Making sure the patch is fully within the cell grid
                        start_i = random.randint(0, cells.shape[-2] // 2 - 1)
                        start_j = random.randint(0, cells.shape[-1] // 2 - 1)
                        end_i, end_j = start_i + dmg.shape[-2], start_j + dmg.shape[-1]

                        # Set cell states in damage region to damage values
                        cells[:, pe_dim + in_dim :, start_i:end_i, start_j:end_j] = dmg

                # Iterate nca (only after the first frame since we want to show seed)
                if k > 0:
                    if cfg.damage.enabled or cfg.custom_growth_rate:
                        if (
                            k == 1
                        ):  # Always force 1 CA iteration for the very first frame
                            T = 1
                        else:
                            T = cfg.experiment.iter.inf.ca.value
                            if T > 1 and k == 2:
                                T -= 1
                            elif k >= cfg.speed_up.start_at and cfg.speed_up.enabled:
                                T = 32
                    else:
                        # Rudimentary iteration schedule that accelerates CA iterations from 1 to 16 steps per frame
                        # during the first 121 frames (assuming 30 fps).
                        # # [1 (0<k<30), ..., 2 (k=30), ..., 4 (k=60), ..., 8 (k=90), ..., 16 (k=120), ...]
                        T = min(2 ** (k // cfg.fps), 16)

                    # Iterate
                    if not cfg.head_masking.enabled:
                        cells = model(
                            cells,
                            step_n=T,
                            update_rate=cfg.experiment.iter.inf.ca.update_rate,
                            attn_size=attn_size,
                        )
                    else:
                        cells = model(
                            cells,
                            step_n=T,
                            update_rate=cfg.experiment.iter.inf.ca.update_rate,
                            attn_size=attn_size,
                            mask=cfg.head_masking.heads_to_mask,
                        )
                    iters_so_far += T

                # Retrieving masked input imgs and making RGB if Grayscale
                masked_x = model.get_rgb_in(cells).clone()  # b (1 or 3) h w
                if masked_x.shape[1] == 1:
                    masked_x = masked_x.tile(1, 3, 1, 1)  # b 3 h w

                # Retrieving output imgs and making RGB if Grayscale
                output_img = model.get_rgb_out(cells).clone()  # b (1 or 3) h w
                if output_img.shape[1] == 1:
                    output_img = output_img.tile(1, 3, 1, 1)  # b 3 h w

                # For visualizing the hidden channels using PCA (but retaining spatial dims).
                #
                # Since hidden chn values are between [-1,1], they are visualized in RGB (via PCA) through a "positive"
                # and "negative" image. Positive PCA img contains positive hidden chn values [0,1] and negative PCA img
                # contains negative hidden chn values flipped to [0,1].
                #
                # Despite manually setting PCA random seed and forcing arpack to be used, sometimes the axes flip
                # which casuses the pos/neg PCA images to sort of colour invert.
                hidden = model.get_hidden(cells).clone()  # b num_hidden h w
                b, num_hidden, h, w = hidden.shape
                hidden = rearrange(hidden, "b hidden h w -> (b h w) hidden")
                try:
                    hidden_pca = pca.fit_transform(hidden.cpu().numpy())  # (b h w) 3
                except Exception:
                    LOGGER.error("Error with ARPACK. Will use zeros for pca instead.")
                    hidden_pca = np.zeros((b * h * w, 3), dtype=np.float32)
                hidden_img = rearrange(
                    hidden_pca, "(b h w) c -> b c h w", b=b, h=h, w=w
                )
                hidden_img_pos = torch.tensor(
                    hidden_img.copy(), dtype=torch.float32, device=output_img.device
                )
                hidden_img_neg = hidden_img_pos * -1.0

                # Retrieving ground truth imgs and making RGB if Grayscale
                ground_truth = x  # b (1 or 3) h w
                if ground_truth.shape[1] == 1:
                    ground_truth = ground_truth.tile(1, 3, 1, 1)  # b 3 h w

                # Retrieving FFT phase and magnitude
                fft_img = fft.fftshift(fft.fft2(output_img, norm="ortho"))
                fft_img_phase = torch.angle(fft_img)
                fft_img_phase = (fft_img_phase / torch.pi + 1) / 2  # clamp to [0, 1]
                fft_img_mag = torch.abs(fft_img)

                # Converting for display
                fft_img_phase = fft_img_phase.mean(1, keepdim=True).repeat(
                    1, output_img.shape[1], 1, 1
                )
                fft_img_mag = fft_img_mag.mean(1, keepdim=True).repeat(
                    1, output_img.shape[1], 1, 1
                )

                # Creating visualized localized attention maps as colour-blended splats about each cell's position
                # Each cell's local attention map is colour-coded based on the cell's position (e.g., top-left
                # cell's 3x3 attention map is blue). Overlapping maps are colour-averaged.
                if "ViT" in cfg.model._target_:
                    if iters_so_far > 0:
                        attn_maps = [
                            viz_attn_maps(
                                layer[0].fn.attn_maps,
                                attn_size,
                                output_img,
                                blend=cfg.viz_attn_maps.blend_with_output,
                                brighten=cfg.viz_attn_maps.brighten_factor,
                            )
                            for layer in model.transformer.layers
                        ]
                    else:
                        attn_maps = [
                            torch.zeros(b * cfg.model.heads, 3, h, w, device=device)
                            for _ in model.transformer.layers
                        ]
                    attn_maps_img = draw_msg(
                        make_grid(attn_maps[0], nrow=batch_size, pad_value=1),
                        "layer 1 heads",
                        "prepend",
                    )
                    for j in range(1, len(attn_maps)):
                        _attn_maps_img = draw_msg(
                            make_grid(attn_maps[j], nrow=batch_size, pad_value=1),
                            f"layer {j+1} heads",
                            "prepend",
                        )
                        attn_maps_img = torch.cat([attn_maps_img, _attn_maps_img], 1)

                # Making image grid of above images
                main_img = make_grid(
                    torch.cat([masked_x, output_img, ground_truth], 0),
                    nrow=batch_size,
                    pad_value=1,
                )
                main_img = draw_msg(main_img, "input, out, gt", "prepend")
                fourier_img = make_grid(
                    torch.cat([fft_img_mag, fft_img_phase], 0),
                    nrow=batch_size,
                    pad_value=1,
                )
                fourier_img = draw_msg(fourier_img, "fourier analysis", "prepend")
                hidden_img_pos = make_grid(hidden_img_pos, nrow=batch_size, pad_value=1)
                hidden_img_pos = draw_msg(
                    hidden_img_pos, f"hidden pos (pca {num_hidden}->3)", "prepend"
                )
                hidden_img_neg = make_grid(hidden_img_neg, nrow=batch_size, pad_value=1)
                hidden_img_neg = draw_msg(
                    hidden_img_neg, f"hidden neg (pca {num_hidden}->3)", "prepend"
                )
                if "ViT" in cfg.model._target_:
                    final_img = torch.cat(
                        [
                            main_img,
                            fourier_img,
                            attn_maps_img,
                            hidden_img_pos,
                            hidden_img_neg,
                        ],
                        1,
                    )
                else:
                    final_img = torch.cat(
                        [main_img, fourier_img, hidden_img_pos, hidden_img_neg], 1
                    )
                final_img = draw_msg(final_img, f"iteration {iters_so_far}", "append")
                final_img = to_nhwc(final_img).squeeze(0).cpu()

                # Add grid image for current frame to VideoWriter. It will automatically convert to uint8 (0, 255)
                vid.add(final_img)

                # Storing inputs and outputs to be used for visualization later
                if cfg.viz_for_paper.enabled:
                    if (
                        k == 0
                        or iters_so_far in list(cfg.viz_for_paper.ca_iters_to_save)
                        or "all" in list(cfg.viz_for_paper.ca_iters_to_save)
                    ):
                        all_input_imgs.append(masked_x)
                        all_output_imgs.append(output_img)
                        all_gt_imgs.append(ground_truth)
                        if "ViT" in cfg.model._target_:
                            attn_maps = torch.cat(attn_maps, 0)  # (l heads b) c h w
                            attn_maps = rearrange(
                                attn_maps,
                                "(l heads b) c h w -> b c (l heads h) w",
                                b=batch_size,
                                l=len(model.transformer.layers),
                            )
                            all_attn_maps.append(attn_maps)
                        saved_so_far += 1

                # Inject new input every N frames
                if (k + 1) % cfg.switch_up.frame_time == 0:
                    x, _ = next(loader)
                    x = x.to(device)
                    masked_x = random_mask(
                        x,
                        p=cfg.experiment.masking.inf.prob,
                        mask_type=cfg.experiment.masking.inf.type,
                        patch_shape=cfg.experiment.masking.inf.patch_shape,
                        random_seed=cfg.experiment.random_seed + k + 1,
                        device=device,
                    )
                    if cfg.completely_mask_input:
                        masked_x *= 0.0
                    if hasattr(model, "patchify"):
                        cells[:, pe_dim : pe_dim + in_dim] = model.patchify(
                            masked_x.clone()
                        )
                    else:
                        cells[:, pe_dim : pe_dim + in_dim] = masked_x.clone()

        # Creating and saving images designed for visualizing for paper
        if cfg.viz_for_paper.enabled:
            for j in trange(batch_size, desc="Saving images in minibatch..."):
                _all_input_imgs = torch.stack(
                    [all_input_imgs[k][j] for k in range(saved_so_far)], 0
                )
                input_img = make_grid(_all_input_imgs, nrow=saved_so_far, pad_value=1)
                _all_output_imgs = torch.stack(
                    [all_output_imgs[k][j] for k in range(saved_so_far)], 0
                )
                output_img = make_grid(_all_output_imgs, nrow=saved_so_far, pad_value=1)
                _all_gt_imgs = torch.stack(
                    [all_gt_imgs[k][j] for k in range(saved_so_far)], 0
                )
                gt_img = make_grid(_all_gt_imgs, nrow=saved_so_far, pad_value=1)
                if "ViT" in cfg.model._target_:
                    _all_attn_maps = torch.stack(
                        [all_attn_maps[k][j] for k in range(saved_so_far)], 0
                    )
                    attn_img = make_grid(_all_attn_maps, nrow=saved_so_far, pad_value=1)
                    img = torch.cat([input_img, output_img, gt_img, attn_img], 1)
                else:
                    img = torch.cat([input_img, output_img, gt_img], 1)

                # Save to disk. It will automatically convert to uint8 (0, 255).
                img_fname = str(imgs_save_folder / f"output_{j:03}.png")
                imwrite(img_fname, img)
            LOGGER.info("Inference output saved images path: %s", imgs_save_folder)

        LOGGER.info("Inference video output path: %s", vid_fname)
    return 1


if __name__ == "__main__":
    inference()
