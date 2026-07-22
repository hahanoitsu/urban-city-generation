from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_preview(layers: np.ndarray, path: Path, scale: int = 2) -> None:
    """Create a debugging preview. The NPZ remains the authoritative dataset."""
    height, width = layers.shape[1:]
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    def paint(mask: np.ndarray, rgb: tuple[int, int, int], alpha: float = 1.0) -> None:
        active = mask > 0.05
        if not active.any():
            return
        source = np.asarray(rgb, dtype=np.float32)
        existing = canvas[active].astype(np.float32)
        canvas[active] = np.clip(existing * (1.0 - alpha) + source * alpha, 0, 255).astype(np.uint8)

    paint(layers[4], (236, 209, 150), 0.65)
    paint(layers[5], (214, 151, 177), 0.70)
    paint(layers[6], (174, 143, 121), 0.70)
    paint(layers[10], (161, 164, 214), 0.75)
    paint(layers[7], (124, 190, 116), 0.80)
    paint(layers[0], (91, 166, 222), 0.95)
    paint(layers[3], (186, 186, 186), 0.95)
    paint(layers[2], (132, 132, 132), 0.98)
    paint(layers[1], (65, 65, 65), 1.00)

    building = layers[8] > 0.05
    height_norm = np.clip(layers[9], 0.0, 1.0)
    if building.any():
        shade = (48 + (1.0 - height_norm) * 85).astype(np.uint8)
        canvas[building] = np.stack([shade[building], shade[building], shade[building]], axis=1)

    image = Image.fromarray(canvas, mode="RGB")
    if scale != 1:
        image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)
