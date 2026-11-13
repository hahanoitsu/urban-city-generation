from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from urban_layers.audit import audit_city
from urban_layers.check import check_city, failures
from urban_layers.city import SLIDERS, generate_city, load_layer, save_city


def pairs(values):
    result = {}
    for value in values or []:
        name, number = value.split("=")
        result[name] = float(number)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--streets", type=Path, required=True)
    parser.add_argument("--buildings", type=Path, required=True)
    parser.add_argument("--trunk", type=Path, help="model for rail, stations and main roads, without it they are drawn by rule")
    parser.add_argument("--pattern", type=Path, help="model for the street pattern map, which way streets run and how regular they are")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=float, nargs=2, default=(4096.0, 4096.0), help="width and height in metres")
    parser.add_argument("--style", nargs="*", help="city=weight, for example singapore=0.7 tokyo=0.3")
    parser.add_argument("--slider", nargs="*", help=f"name=factor, names: {', '.join(SLIDERS)}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan-steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--pattern-guidance", type=float, default=1.0, help="how hard the streets follow the pattern map, 1 is as trained")
    parser.add_argument("--corridor-guidance", type=float, default=1.0, help="how hard the corridors follow the plan's main road and rail cells, 1 is as trained")
    parser.add_argument("--map-guidance", type=float, default=3.0, help="guidance when the pattern map itself is drawn")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--street-temperature", type=float, help="temperature for the street model alone, lower gives more regular streets")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--building-samples", type=int, default=4, help="samples per block, the one closest to the planned coverage is kept")
    parser.add_argument("--building-batch", type=int, default=64, help="blocks per call to the building model, more is faster on a big gpu")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-transport-rules", action="store_true", help="keep rail and roads exactly as the models drew them, no rail layer and no joining of roads")
    parser.add_argument("--no-repairs", action="store_true", help="skip the repairs after the models: joining near misses, removing stubs, linking and dropping cut off pieces")
    parser.add_argument("--list", action="store_true", help="print the cities the models know and stop")
    args = parser.parse_args()

    models = {name: load_layer(path, args.device) for name, path in (("plan", args.plan), ("pattern", args.pattern), ("trunk", args.trunk), ("streets", args.streets), ("buildings", args.buildings)) if path is not None}
    if args.list:
        for name, (_model, checkpoint) in models.items():
            print(name, checkpoint["cities"])
        return
    started = time.time()
    city, plan = generate_city(
        models,
        size_m=tuple(args.size),
        style=pairs(args.style),
        sliders=pairs(args.slider),
        seed=args.seed,
        plan_steps=args.plan_steps,
        guidance=args.guidance,
        pattern_guidance=args.pattern_guidance,
        corridor_guidance=args.corridor_guidance,
        map_guidance=args.map_guidance,
        street_temperature=args.street_temperature,
        temperature=args.temperature,
        top_p=args.top_p,
        building_samples=args.building_samples,
        building_batch=args.building_batch,
        transport_rules=not args.no_transport_rules,
        repairs=not args.no_repairs,
        device=args.device,
        log=lambda text: print(text, flush=True),
    )
    save_city(city, plan, args.output)
    report = audit_city(city, plan)
    report["seconds"] = time.time() - started
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    checked = check_city(city, plan)
    checked["failures"] = failures(checked)
    (args.output / "check.json").write_text(json.dumps(checked, indent=2) + "\n")
    print("spatial checks:", "all pass" if not checked["failures"] else checked["failures"])
    print(json.dumps({"network": report["network"], "buildings": {k: v for k, v in report["buildings"].items() if k != "landuse"}, "rail": report["rail"], "cover": report["cover"]}, indent=2))


if __name__ == "__main__":
    main()
