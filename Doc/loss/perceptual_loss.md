

## What is a "Good Range" for Perceptual Loss?

For a video VAE using LPIPS (Learned Perceptual Image Patch Similarity), here are the standard ranges:

* **0.01 – 0.05 (Excellent):** This is "perceptually indistinguishable." The generated video looks almost identical to the original. Your "Good" log was consistently in this range.
* **0.05 – 0.15 (Good):** The video looks sharp, but you might notice minor smoothing of textures (like grass or hair) or slight flickering.
* **0.15 – 0.30 (Acceptable but Blurry):** The video is recognizable, but looks "soft" or out of focus. Fine details are lost.
* **> 0.30 (Bad):** The video looks like a watercolor painting or has significant artifacts.

### Summary

Your current value of **~0.04** is technically in the "Excellent" range, but because your `nll_loss` is exploding (hitting **0.50**), this low perceptual loss is meaningless. It likely means the model has collapsed to outputting a specific pattern that minimizes feature loss but fails to actually reconstruct the specific video content.