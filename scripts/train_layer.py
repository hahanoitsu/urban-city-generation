from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from urban_layers.buildings import BuildingDataset, BuildingModel
from urban_layers.data import CHANNELS, PATTERN, TRUNK, load_cities, pool_plan
from urban_layers.decoder import sequence_loss
from urban_layers.plan import PatternCrops, PlanCrops, PlanNet, pattern_controls, plan_loss, plan_statistics, urban_cells
from urban_layers.streets import StreetDataset, StreetModel, collate


def held_out(name, x, y, size=2048.0, every=10):
    key = f"{name}:{int(x // size)}:{int(y // size)}"
    return int(hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest(), 16) % every == 0


def update_ema(ema, model, decay):
    with torch.no_grad():
        for average, parameter in zip(ema.parameters(), model.parameters()):
            average.lerp_(parameter, 1.0 - decay)


def move(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=["plan", "pattern", "trunk", "streets", "buildings"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500, help="plan steps per epoch")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--width", type=int, default=64, help="plan unet width")
    parser.add_argument("--crop", type=int, default=64, help="plan crop in cells")
    parser.add_argument("--max-len", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overfit", type=int, default=0, help="train and validate on this many samples")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cities", nargs="*", help="train on these cities only, any other city in the data folder is held out")
    parser.add_argument("--plan-cells", type=int, default=16, help="trunk: plan cells each tile sees")
    parser.add_argument("--spread", nargs="*", default=[], help="trunk: plan channels the model is told the amount of but not where, for example major rail")
    parser.add_argument("--start", type=Path, help="checkpoint to start from instead of random weights")
    parser.add_argument("--pattern", action="store_true", help="streets: also show the model the street pattern map")
    parser.add_argument("--reach", type=float, default=1.0, help="trunk: tile widths over which the position code stays unique, 2 for 16 plan cells")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    cities = load_cities(args.data)
    if args.cities:
        missing = set(args.cities) - {city["name"] for city in cities}
        if missing:
            raise SystemExit(f"no layer data for {sorted(missing)} in {args.data}")
        cities = [city for city in cities if city["name"] in args.cities]
    names = [city["name"] for city in cities]
    plans = [city["plan"] for city in cities]
    mean, std, control_scale = plan_statistics(plans)
    scale = (1.0 / std).astype(np.float32)
    print(json.dumps({"layer": args.layer, "cities": names, "device": str(device)}), flush=True)

    if args.layer == "plan":
        keep = [urban_cells(plan) for plan in plans]
        print(json.dumps({"urban_share": {name: round(float(mask.mean()), 3) for name, mask in zip(names, keep)}}), flush=True)
        model = PlanNet(len(cities), width=args.width)
        train_set = PlanCrops(plans, mean, std, control_scale, size=args.crop, length=args.steps * args.batch_size, keep=keep)
        validation_set = PlanCrops(plans, mean, std, control_scale, size=args.crop, length=16 * args.batch_size, keep=keep)
        extra = {
            "mean": mean,
            "std": std,
            "control_scale": control_scale,
            "city_controls": [plan[:, mask].mean(axis=1) for plan, mask in zip(plans, keep)],
        }
    elif args.layer == "pattern":
        patterns = [city["pattern"] for city in cities]
        if any(pattern is None for pattern in patterns):
            raise SystemExit("a city has no pattern.npy, build its layer data again")
        keep = [urban_cells(plan) for plan in plans]
        told_mean, told_std, _scale = plan_statistics(patterns)
        model = PlanNet(len(cities), channels=len(PATTERN), width=args.width, given=len(CHANNELS), controls=len(PATTERN) + 1)
        train_set = PatternCrops(patterns, plans, told_mean, told_std, mean, std, size=args.crop, length=args.steps * args.batch_size, keep=keep)
        validation_set = PatternCrops(patterns, plans, told_mean, told_std, mean, std, size=args.crop, length=16 * args.batch_size, keep=keep)
        extra = {
            "mean": told_mean,
            "std": told_std,
            "given_mean": mean,
            "given_std": std,
            "city_controls": [pattern_controls(pattern, mask) for pattern, mask in zip(patterns, keep)],
        }
    else:
        items = []
        files = {"streets": "streets.pt", "trunk": "trunk.pt", "buildings": "blocks.pt"}
        unit = TRUNK if args.layer == "trunk" else 1.0
        for index, city in enumerate(cities):
            if not (city["folder"] / files[args.layer]).exists():
                raise SystemExit(f"{city['name']} has no {files[args.layer]}, build its layer data again")
            records = torch.load(city["folder"] / files[args.layer], weights_only=False)
            for record in records:
                x, y = record["centre"] if args.layer == "buildings" else record["origin"]
                items.append((index, record, held_out(city["name"], x * unit, y * unit)))
        if args.overfit:
            items = sorted(items, key=lambda item: hashlib.sha1(repr(item[1].get("origin", item[1].get("centre"))).encode(), usedforsecurity=False).hexdigest())[: args.overfit]
            train_items = [(index, record) for index, record, _flag in items]
            validation_items = train_items
        else:
            train_items = [(index, record) for index, record, flag in items if not flag]
            validation_items = [(index, record) for index, record, flag in items if flag] or train_items[:64]
        extra = {"scale": scale}
        if args.layer == "streets":
            model = StreetModel(len(cities), args.d, args.layers, args.heads, args.max_len, dropout=0.0 if args.overfit else args.dropout, exits=True, guided=True, pattern=args.pattern)
            hidden = ("major", "rail")
            patterns = None
            if args.pattern:
                patterns = [city["pattern"] for city in cities]
                if any(pattern is None for pattern in patterns):
                    raise SystemExit("a city has no pattern.npy, build its layer data again")
            train_set = StreetDataset(train_items, plans, len(cities), scale, augment=not args.overfit, max_len=args.max_len, exits=True, pinned=True, hidden=hidden, patterns=patterns)
            validation_set = StreetDataset(validation_items, plans, len(cities), scale, augment=False, max_len=args.max_len, exits=True, pinned=True, hidden=hidden, patterns=patterns)
            extra = {"scale": scale, "pinned": True, "hidden": hidden}
        elif args.layer == "trunk":
            if (args.plan_cells - 1) / 4 >= 2 * args.reach:
                raise SystemExit("the position code repeats inside this many plan cells, raise --reach")
            coarse = [pool_plan(plan, TRUNK) for plan in plans]
            scale = (1.0 / plan_statistics(coarse)[1]).astype(np.float32)
            model = StreetModel(len(cities), args.d, args.layers, args.heads, args.max_len, dropout=0.0 if args.overfit else args.dropout, stations=True, exits=True, reach=args.reach)
            train_set = StreetDataset(train_items, coarse, len(cities), scale, augment=not args.overfit, max_len=args.max_len, stations=True, plan_cells=args.plan_cells, spread=args.spread, exits=True)
            validation_set = StreetDataset(validation_items, coarse, len(cities), scale, augment=False, max_len=args.max_len, stations=True, plan_cells=args.plan_cells, spread=args.spread, exits=True)
            extra = {"scale": scale, "unit": TRUNK, "plan_cells": args.plan_cells, "spread": args.spread}
        else:
            model = BuildingModel(len(cities), args.d, args.layers, args.heads, args.max_len, dropout=0.0 if args.overfit else args.dropout, roads=True)
            train_set = BuildingDataset(train_items, len(cities), scale, augment=not args.overfit, max_len=args.max_len)
            validation_set = BuildingDataset(validation_items, len(cities), scale, augment=False, max_len=args.max_len)
        print(json.dumps({"train": len(train_set), "validation": len(validation_set)}), flush=True)

    if args.start is not None:
        saved = torch.load(args.start, map_location="cpu", weights_only=False)
        if saved["cities"] != names:
            raise SystemExit(f"{args.start} was trained on {saved['cities']}, not {names}")
        state = saved["ema"]
        for key, value in model.state_dict().items():
            if key in state and state[key].shape != value.shape:
                wide = torch.zeros_like(value)
                wide[:, : state[key].shape[1]] = state[key]
                state[key] = wide
        model.load_state_dict(state)
    model = model.to(device)
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    options = {"batch_size": args.batch_size, "num_workers": args.workers, "persistent_workers": args.workers > 0}
    drawn = args.layer in ("plan", "pattern")
    if not drawn:
        options["collate_fn"] = collate
    train_loader = DataLoader(train_set, shuffle=not drawn, drop_last=len(train_set) > args.batch_size, **options)
    validation_loader = DataLoader(validation_set, shuffle=False, **options)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "experiment.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "cities": names, "parameters": sum(p.numel() for p in model.parameters())}, indent=2) + "\n")

    def loss_of(net, batch):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if drawn:
                return plan_loss(net, batch)
            return sequence_loss(net(batch), batch["tokens"])

    step = 0
    best = math.inf
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = move(batch, device)
            rate = args.learning_rate * min(1.0, (step + 1) / max(args.warmup, 1))
            for group in optimizer.param_groups:
                group["lr"] = rate
            loss = loss_of(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            update_ema(ema, model, min(args.ema, (1.0 + step) / (10.0 + step)))
            total += float(loss.detach())
            count += 1
        torch.manual_seed(1234)
        validation = 0.0
        batches = 0
        with torch.no_grad():
            for batch in validation_loader:
                validation += float(loss_of(ema, move(batch, device)))
                batches += 1
        torch.manual_seed(args.seed + epoch)
        validation /= max(batches, 1)
        checkpoint = {
            "layer": args.layer,
            "config": model.config,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "cities": names,
            "epoch": epoch,
            "validation": validation,
            **extra,
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if validation < best:
            best = validation
            torch.save(checkpoint, args.output / "best.pt")
        record = {"epoch": epoch, "train": total / max(count, 1), "validation": validation, "steps": step, "minutes": (time.time() - started) / 60.0}
        with (args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"epoch={epoch}/{args.epochs} train={record['train']:.4f} validation={validation:.4f} best={best:.4f} steps={step} minutes={record['minutes']:.1f}", flush=True)


if __name__ == "__main__":
    main()
