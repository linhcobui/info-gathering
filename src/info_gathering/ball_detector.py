"""Detect the yellow ball in rendered or decoded frames by colour.

Yellowness = min(R, G) - B. The ball renders at about (1.0, 0.99, 0.18), i.e. ~0.8; the red
cups and the blue checker floor are <= 0. On the ground-truth 64px frames of episodes
950-999, every frame showing the ball has >= 30 pixels above 0.2 and every other frame has
none, which leaves a wide margin for blurry decoder output.
"""
import torch


def yellowness(frames):
    """frames (..., H, W, 3) float in [0, 1] -> (..., H, W)."""
    return torch.minimum(frames[..., 0], frames[..., 1]) - frames[..., 2]


def ball_pixels(frames, threshold=0.15):
    """Number of yellow pixels in each frame: (..., H, W, 3) -> (...)."""
    return (yellowness(frames) > threshold).flatten(-2).sum(-1)
