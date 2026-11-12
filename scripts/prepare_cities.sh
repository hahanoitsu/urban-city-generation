#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
LAYERS="${LAYERS:-data/layers-cities}"
REAL="${REAL:-runs/real}"

cities=("$@")
if [ ${#cities[@]} -eq 0 ]; then
    cities=(singapore tokyo seoul london paris barcelona berlin newyork melbourne)
fi

for city in "${cities[@]}"; do
    config="configs/cities/$city.yaml"
    pbf="$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['input']['pbf_path'])" "$config")"
    gpkg="$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['output']['gpkg_path'])" "$config")"
    name="$(basename "$pbf" .osm.pbf)"
    echo "== $city $(date +%H:%M:%S)"
    if [ ! -f "$pbf" ]; then
        mkdir -p "$(dirname "$pbf")"
        wget -q -c -O "$pbf.part" "https://download.bbbike.org/osm/bbbike/$name/$name.osm.pbf"
        mv "$pbf.part" "$pbf"
    fi
    if [ ! -f "$gpkg" ]; then
        python -m urban_dataset prepare-city --config "$config"
    fi
    if [ ! -f "$LAYERS/$city/meta.json" ]; then
        python scripts/build_layer_data.py --city "$gpkg" --name "$city" --output "$LAYERS"
    fi
    if [ ! -f "$REAL/$city-4km/summary.json" ]; then
        python scripts/audit_real_city.py --city "$gpkg" --plan "$LAYERS/$city/plan.npz" --output "$REAL/$city-4km" --size 4096 --count 6 > "$REAL-$city-4km.log" 2>&1 || echo "real audit failed for $city, see $REAL-$city-4km.log"
    fi
    echo "== $city done $(date +%H:%M:%S)"
done
