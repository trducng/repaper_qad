import torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import __init__
from masked_autoencoding.src.utils import mono_to_rgb


class CALoss(torch.nn.Module):
    def __init__(self, rec_factor=1e2, overflow_factor=1e2):
        super().__init__()
        self.rec_factor = rec_factor
        self.overflow_factor = overflow_factor
        self.lpips = None

    def inst_lpips(self, device):
        self.lpips = LearnedPerceptualImagePatchSimilarity("vgg").to(device)
        self.lpips.requires_grad_(False)
        self.lpips.eval()

    def forward(self, cfg, model, results, phase="train"):
        cells = results["output_cells"]
        hidden = model.get_hidden(cells)
        output = results["output_img"]
        target = results["ground_truth"]["x"]

        losses = {}

        # L1 loss for image reconstruction task
        losses["rec_loss"] = self.rec_factor * torch.nn.functional.l1_loss(
            output, target
        )

        # Overflow loss to prevent cell state overflow
        hidden_overflow_loss = (hidden - torch.clip(hidden, -1.0, 1.0)).abs().mean()
        rgb_overflow_loss = (output - torch.clip(output, 0, 1)).abs().mean()
        losses["overflow_loss"] = self.overflow_factor * (
            hidden_overflow_loss + rgb_overflow_loss
        )

        if phase == "test" or phase == "validation":
            output = torch.clip(output, 0, 1)
            target = torch.clip(target, 0, 1)
            losses["psnr"] = psnr(output, target, data_range=1.0)

            if phase == "test":
                losses["ssim"] = ssim(output, target, data_range=1.0)

                if self.lpips:
                    # Expects input to be in range [-1,1] with shape (N, 3, H, W)
                    losses["lpips"] = self.lpips(
                        mono_to_rgb(output * 2 - 1), mono_to_rgb(target * 2 - 1)
                    )

        return losses


class Loss(torch.nn.Module):
    def __init__(self, rec_factor=1e2, overflow_factor=1e2):
        super().__init__()
        self.rec_factor = rec_factor
        self.overflow_factor = overflow_factor
        self.lpips = None

    def inst_lpips(self, device):
        self.lpips = LearnedPerceptualImagePatchSimilarity("vgg").to(device)
        self.lpips.requires_grad_(False)
        self.lpips.eval()

    def forward(self, cfg, model, results, phase="train"):
        output = results["output_img"]
        target = results["ground_truth"]["x"]

        losses = {}

        # L1 loss for image reconstruction task
        losses["rec_loss"] = self.rec_factor * torch.nn.functional.l1_loss(
            output, target
        )

        # overflow loss to prevent output overflow
        rgb_overflow_loss = (output - torch.clip(output, 0, 1)).abs().mean()
        losses["overflow_loss"] = self.overflow_factor * rgb_overflow_loss

        if phase == "test" or phase == "validation":
            output = torch.clip(output, 0, 1)
            target = torch.clip(target, 0, 1)
            losses["psnr"] = psnr(output, target, data_range=1.0)

            if phase == "test":
                losses["ssim"] = ssim(output, target, data_range=1.0)

                if self.lpips:
                    # Expects input to be in range [-1,1] with shape (N, 3, H, W)
                    losses["lpips"] = self.lpips(
                        mono_to_rgb(output * 2 - 1), mono_to_rgb(target * 2 - 1)
                    )

        return losses
