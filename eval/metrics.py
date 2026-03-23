"""
eval/metrics.py
---------------
All evaluation metrics for LiteEdit-Qwen.

Metrics tracked per experiment:
  - LPIPS   : perceptual similarity (lower = more similar to reference)
  - PSNR    : peak signal-to-noise ratio (higher = better)
  - FID     : Fréchet Inception Distance (lower = better distribution match)
  - latency : seconds per image (lower = faster)
  - peak_vram: peak GPU memory in GB (lower = more efficient)

Usage:
    from eval.metrics import MetricsTracker

    tracker = MetricsTracker(device="cuda")
    results = tracker.compute(pred_image, ref_image)
    print(results)
    # {'lpips': 0.12, 'psnr': 28.4, 'latency': 3.2, 'peak_vram_gb': 5.1}
"""

import time
import torch
import numpy as np
from PIL import Image
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional import peak_signal_noise_ratio


class MetricsTracker:

    def __init__(self, device: str = "cuda"):
        self.device = device

        # LPIPS: AlexNet backbone is fastest, VGG is more accurate
        self.lpips_fn = lpips.LPIPS(net="alex").to(device)
        self.lpips_fn.eval()

        # FID: needs enough images to be meaningful (>= 50 recommended)
        # We accumulate then compute at the end of a batch
        self.fid = FrechetInceptionDistance(feature=2048).to(device)
        self.fid_updated = False

        self._start_time = None
        self._peak_vram_start = 0

    # ------------------------------------------------------------------
    # Timing and VRAM
    # ------------------------------------------------------------------

    def start_timer(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._peak_vram_start = torch.cuda.memory_allocated()
        self._start_time = time.perf_counter()

    def stop_timer(self):
        elapsed = time.perf_counter() - self._start_time
        peak_vram = 0.0
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1e9  # GB
        return elapsed, peak_vram

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pil_to_tensor(image: Image.Image, size: int = 256) -> torch.Tensor:
        """Convert PIL image to normalized [-1, 1] tensor for LPIPS."""
        img = image.convert("RGB").resize((size, size))
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def pil_to_uint8_tensor(image: Image.Image, size: int = 256) -> torch.Tensor:
        """Convert PIL image to uint8 tensor for FID."""
        img = image.convert("RGB").resize((size, size))
        arr = np.array(img)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    # ------------------------------------------------------------------
    # Per-image metrics
    # ------------------------------------------------------------------

    def compute_lpips(self, pred: Image.Image, ref: Image.Image) -> float:
        """Compute LPIPS perceptual distance. Lower = more similar."""
        pred_t = self.pil_to_tensor(pred).to(self.device)
        ref_t  = self.pil_to_tensor(ref).to(self.device)
        with torch.no_grad():
            score = self.lpips_fn(pred_t, ref_t)
        return float(score.item())

    def compute_psnr(self, pred: Image.Image, ref: Image.Image) -> float:
        """Compute PSNR in dB. Higher = better."""
        pred_t = self.pil_to_uint8_tensor(pred).float() / 255.0
        ref_t  = self.pil_to_uint8_tensor(ref).float() / 255.0
        return float(peak_signal_noise_ratio(pred_t, ref_t, data_range=1.0).item())

    def compute_outside_psnr(
        self,
        pred: Image.Image,
        original: Image.Image,
        mask: Image.Image,
    ) -> float:
        """
        PSNR of unmasked region only.
        Key metric for local editing — measures preservation of unedited areas.
        Higher = better preservation.
        """
        pred_np = np.array(pred.convert("RGB").resize((256, 256))).astype(np.float32)
        orig_np = np.array(original.convert("RGB").resize((256, 256))).astype(np.float32)
        mask_np = np.array(mask.convert("L").resize((256, 256))) / 255.0

        outside = (1 - mask_np)[:, :, None]
        mse = ((pred_np - orig_np) ** 2 * outside).sum() / (outside.sum() * 3 + 1e-8)
        if mse < 1e-8:
            return 100.0  # perfect preservation
        return float(10 * np.log10(255**2 / mse))

    # ------------------------------------------------------------------
    # FID (batch-level)
    # ------------------------------------------------------------------

    def update_fid(self, pred: Image.Image, ref: Image.Image):
        """Add a pair to the FID accumulator. Call compute_fid() after all pairs."""
        pred_t = self.pil_to_uint8_tensor(pred)
        ref_t  = self.pil_to_uint8_tensor(ref)
        self.fid.update(pred_t.to(self.device), real=False)
        self.fid.update(ref_t.to(self.device),  real=True)
        self.fid_updated = True

    def compute_fid(self) -> float:
        """Compute FID over all accumulated images. Lower = better."""
        if not self.fid_updated:
            return -1.0
        score = float(self.fid.compute().item())
        self.fid.reset()
        self.fid_updated = False
        return score

    # ------------------------------------------------------------------
    # Convenience: compute all per-image metrics at once
    # ------------------------------------------------------------------

    def compute(
        self,
        pred: Image.Image,
        ref: Image.Image,
        original: Image.Image = None,
        mask: Image.Image = None,
    ) -> dict:
        """
        Compute all per-image metrics.

        Args:
            pred:     Model output image
            ref:      Ground-truth reference image
            original: Original input image (needed for outside_psnr)
            mask:     Edit mask (needed for outside_psnr)

        Returns:
            dict with lpips, psnr, and optionally outside_psnr
        """
        results = {
            "lpips": self.compute_lpips(pred, ref),
            "psnr":  self.compute_psnr(pred, ref),
        }
        if original is not None and mask is not None:
            results["outside_psnr"] = self.compute_outside_psnr(pred, original, mask)
        return results
