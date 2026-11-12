from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from urban_layers.buildings import BuildingDataset
from urban_layers.city import load_layer
from urban_layers.data import load_cities, pool_plan
from urban_layers.decoder import sequence_loss
from urban_layers.streets import StreetDataset, collate


def held_out(name, x, y, size=2048.0, every=10):
    key = f"{name}:{int(x // size)}:{int(y // size)}"
    return int(hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest(), 16) % every == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1500, help="samples per city")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(0)
    model, checkpoint = load_layer(args.checkpoint, args.device)
    layer = checkpoint["layer"]
    known = checkpoint["cities"]
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    unit = float(checkpoint.get("unit", 1.0))
    file = {"streets": "streets.pt", "trunk": "trunk.pt", "buildings": "blocks.pt"}[layer]
    report = {}
    for city in load_cities(args.data):
        name = city["name"]
        records = torch.load(city["folder"] / file, weights_only=False)
        if name in known:
            records = [record for record in records if held_out(name, *(np.asarray(record["centre"] if layer == "buildings" else record["origin"]) * unit))]
        random.shuffle(records)
        items = [(0, record) for record in records[: args.limit]]
        if layer == "buildings":
            data = BuildingDataset(items, 1, scale, augment=False, max_len=model.config["max_len"])
        else:
            plan = pool_plan(city["plan"], unit) if layer == "trunk" else city["plan"]
            data = StreetDataset(items, [plan], 1, scale, augment=False, max_len=model.config["max_len"], stations=model.config.get("stations", False), plan_cells=int(checkpoint.get("plan_cells", 8)), spread=checkpoint.get("spread", ()), exits=model.config.get("exits", False), pinned=checkpoint.get("pinned", False), hidden=checkpoint.get("hidden", ()))
        codes = {"no city": np.zeros(len(known), dtype=np.float32)}
        if name in known:
            codes["own city"] = np.eye(len(known), dtype=np.float32)[known.index(name)]
        found = {"samples": len(items), "seen in training": name in known}
        for label, code in codes.items():
            total = 0.0
            count = 0
            for start in range(0, len(data), args.batch_size):
                batch = []
                for index in range(start, min(start + args.batch_size, len(data))):
                    item = data[index]
                    item["city"] = code
                    batch.append(item)
                arrays = {key: value.to(args.device) for key, value in collate(batch).items()}
                with torch.no_grad(), torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    total += float(sequence_loss(model(arrays), arrays["tokens"])) * len(batch)
                count += len(batch)
            found[label] = total / max(count, 1)
        report[name] = found
        print(name, json.dumps(found), flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
