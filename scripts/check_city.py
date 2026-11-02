from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from urban_layers.check import check_city, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=Path, nargs="+", help="output folders of generate_city.py")
    args = parser.parse_args()
    for folder in args.folders:
        city = json.loads((folder / "city.json").read_text())
        plan = np.load(folder / "plan.npy") if (folder / "plan.npy").exists() else None
        report = check_city(city, plan)
        found = failures(report)
        report["failures"] = found
        (folder / "check.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{folder.name}: {'all spatial checks pass' if not found else str(len(found)) + ' problems'}")
        for line in found:
            print("   ", line)
        print("    seams per km2", report["tile_seams_per_km2"], "near miss ends per km2", report["near_miss_ends_per_km2"], "road on the ground over water m", report["roads_on_the_ground_over_water_m"], "elevated ends with no road near", report["elevated_ends_with_no_road_near"])
        if plan is not None:
            print("    plan orientation", {key: round(value, 2) for key, value in report["plan_orientation"].items()})


if __name__ == "__main__":
    main()
