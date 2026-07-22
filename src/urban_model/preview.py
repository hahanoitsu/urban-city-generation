from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from urban_dataset.preview import render_preview

from .reconstruct import reconstruct_layers


def save_reconstruction_preview(
    inputs,
    outputs,
    tile_ids: list[str],
    destination: str | Path,
    *,
    limit: int = 4,
) -> Path:
    reconstructed = reconstruct_layers(outputs).detach().cpu().numpy()
    originals = inputs.detach().cpu().numpy()
    count = min(limit, len(originals))
    if count == 0:
        raise ValueError("Cannot create a preview from an empty batch")

    tile_size = originals.shape[-1]
    label_height = 24
    canvas = Image.new("RGB", (tile_size * 2, count * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(count):
        original = render_preview(originals[index])
        reconstruction = render_preview(reconstructed[index])
        y = index * (tile_size + label_height)
        canvas.paste(original, (0, y))
        canvas.paste(reconstruction, (tile_size, y))
        draw.text((4, y + tile_size + 5), str(tile_ids[index]), fill="black")
        draw.text((tile_size + 4, y + tile_size + 5), "reconstruction", fill="black")

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)
    return path
