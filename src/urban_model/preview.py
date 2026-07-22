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
    known_layers=None,
    known_mask=None,
    full_target=None,
    limit: int = 4,
) -> Path:
    reconstructed = reconstruct_layers(outputs).detach().cpu()
    count = min(limit, len(reconstructed))
    if count == 0:
        raise ValueError("Cannot create a preview from an empty batch")

    label_height = 24
    if known_layers is not None and known_mask is not None and full_target is not None:
        known = known_layers.detach().cpu()
        mask = known_mask.detach().cpu()
        targets = full_target.detach().cpu()
        composed = known + reconstructed * (1.0 - mask)
        height, width = targets.shape[-2:]
        canvas = Image.new("RGB", (width * 3, count * (height + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        for index in range(count):
            y = index * (height + label_height)
            condition = render_preview(known[index].numpy())
            target = render_preview(targets[index].numpy())
            prediction = render_preview(composed[index].numpy())
            canvas.paste(condition, (0, y))
            canvas.paste(target, (width, y))
            canvas.paste(prediction, (width * 2, y))
            boundary = width // 2
            draw.line((boundary, y, boundary, y + height), fill="white", width=2)
            draw.line((width * 2 + boundary, y, width * 2 + boundary, y + height), fill="white", width=2)
            draw.text((4, y + height + 5), str(tile_ids[index]), fill="black")
            draw.text((width + 4, y + height + 5), "real continuation", fill="black")
            draw.text((width * 2 + 4, y + height + 5), "predicted continuation", fill="black")
    else:
        originals = inputs.detach().cpu().numpy()
        reconstructed_np = reconstructed.numpy()
        height, width = originals.shape[-2:]
        canvas = Image.new("RGB", (width * 2, count * (height + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        for index in range(count):
            original = render_preview(originals[index])
            reconstruction = render_preview(reconstructed_np[index])
            y = index * (height + label_height)
            canvas.paste(original, (0, y))
            canvas.paste(reconstruction, (width, y))
            draw.text((4, y + height + 5), str(tile_ids[index]), fill="black")
            draw.text((width + 4, y + height + 5), "reconstruction", fill="black")

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)
    return path
