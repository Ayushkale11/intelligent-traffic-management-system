#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_DIR"

echo "=============================================="
echo "CREATING V3 MULTI-JUNCTION RANDOM TRAFFIC"
echo "=============================================="

python "$SUMO_HOME/tools/randomTrips.py" \
    -n v3.net.xml \
    -r v3.rou.xml \
    -e 600 \
    -b 0 \
    --period 2.0 \
    --seed 42 \
    --min-distance 100 \
    --fringe-factor 10 \
    --validate

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Route generation failed."
    exit 1
fi

echo
echo "=============================================="
echo "V3 ROUTES CREATED"
echo "=============================================="

echo "Route file:"
echo "$PROJECT_DIR/v3.rou.xml"
