from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from urban_ai.codec import FIELDS
from urban_ai.loss import graph_program_loss
from urban_model.context_graph import ContextGraphModelConfig, ContextGraphProgramModel
from urban_model.context_graph_data import ContextGraphProgramDataset


def shift(batch):
    inputs = {field: batch[field][:, :-1] for field in FIELDS}
    targets = {field: batch[field][:, 1:] for field in FIELDS}
    return inputs, targets


def accuracy(logits, targets):
    result = {}
    op = targets["op"]
    masks = {
        "op": op.ne(0),
        "x": op.eq(2) | op.eq(3),
        "y": op.eq(2) | op.eq(3),
        "id1": op.eq(3) | op.eq(4),
        "id2": op.eq(4),
        "mode": op.eq(2) | op.eq(3) | op.eq(4),
        "class": op.eq(3) | op.eq(4),
        "width": op.eq(3) | op.eq(4),
        "vertical": op.eq(2) | op.eq(3) | op.eq(4),
        "layer": op.eq(2) | op.eq(3) | op.eq(4),
    }
    for name, mask in masks.items():
        if not bool(mask.any()):
            result[name] = 0.0
            continue
        expected = targets[name] if name == "op" else targets[name] - 1
        predicted = logits[name].argmax(dim=-1)
        result[name] = float((predicted[mask] == expected[mask]).float().mean().item())
    return result


def corrupt_commands(commands, probability):
    if probability <= 0:
        return commands
    result = {name: value.clone() for name, value in commands.items()}
    mask = torch.rand_like(result["op"], dtype=torch.float32).lt(probability)
    mask &= result["op"].ne(0)
    mask[:, 0] = False
    for name in result:
        result[name][mask] = 0
    return result


def run_epoch(model, loader, dataset, device, optimizer=None, command_dropout=0.0):
    training = optimizer is not None
    model.train(training)
    losses = []
    acc = {}
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            commands, targets = shift({field: batch[field].to(device) for field in FIELDS})
            model_commands = corrupt_commands(commands, command_dropout if training else 0.0)
            values = batch["context"].to(device)
            ports = batch["ports"].to(device)
            padding = batch["port_padding"].to(device)
            relations = dataset.relations.to(device).unsqueeze(0).expand(values.shape[0], -1, -1, -1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(model_commands, values, relations, ports, padding)
                loss, _parts = graph_program_loss(logits, targets)
                root_x = F.cross_entropy(logits["x"][:, 0], targets["x"][:, 0] - 1)
                root_y = F.cross_entropy(logits["y"][:, 0], targets["y"][:, 0] - 1)
                loss = loss + 0.5 * (root_x + root_y)
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach()))
            current = accuracy(logits, targets)
            for name, value in current.items():
                acc.setdefault(name, []).append(value)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "accuracy": {name: sum(values) / len(values) for name, values in acc.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--command-dropout", type=float, default=0.35)
    args = parser.parse_args()

    torch.manual_seed(5132)
    torch.cuda.manual_seed_all(5132)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")
    dataset = ContextGraphProgramDataset(
        args.data,
        maximum_samples=args.samples,
        maximum_commands=768,
        maximum_ports=96,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    config = ContextGraphModelConfig(
        codec=dataset.codec,
        context_dimensions=dataset.context_dimensions,
        port_dimensions=dataset.port_dimensions,
        relation_count=len(dataset.relation_names),
        maximum_sequence_length=767,
    )
    model = ContextGraphProgramModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "samples": len(dataset),
        "sample_ids": [row[0]["id"] for row in dataset.samples],
        "feature_names": dataset.feature_names,
        "context_nodes": len(dataset.node_ids),
        "context_dimensions": dataset.context_dimensions,
        "port_dimensions": dataset.port_dimensions,
        "relation_names": list(dataset.relation_names),
        "context_mean": dataset.context_mean.tolist(),
        "context_std": dataset.context_std.tolist(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "command_dropout": args.command_dropout,
        "config": config.to_dict(),
    }
    (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")

    started = time.time()
    best = math.inf
    records = []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model,
            loader,
            dataset,
            device,
            optimizer,
            command_dropout=args.command_dropout,
        )
        evaluation = run_epoch(model, loader, dataset, device)
        record = {
            "epoch": epoch,
            "train": train,
            "evaluation": evaluation,
            "seconds": time.time() - started,
        }
        records.append(record)
        with (args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "config": config.to_dict(),
            "best_loss": min(best, evaluation["loss"]),
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if evaluation["loss"] < best:
            best = evaluation["loss"]
            torch.save(checkpoint, args.output / "best.pt")
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} train={train['loss']:.4f} eval={evaluation['loss']:.4f} "
                f"op={evaluation['accuracy']['op']:.3f} x={evaluation['accuracy']['x']:.3f} "
                f"y={evaluation['accuracy']['y']:.3f}",
                flush=True,
            )

    summary = {
        "epochs": args.epochs,
        "samples": len(dataset),
        "best_loss": best,
        "seconds": time.time() - started,
        "last": records[-1],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
