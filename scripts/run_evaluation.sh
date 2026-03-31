#!/bin/bash
set -e

echo "Running evaluation suite..."

MODEL=${1:-runs/sac_test/best_model.zip}
N_EPISODES=${2:-50}
OUTPUT=results/evaluation

echo "  Model:    $MODEL"
echo "  Episodes: $N_EPISODES"
echo "  Output:   $OUTPUT"
echo ""

python -m quantflow.evaluation.compare \
    --model-path "$MODEL" \
    --n-episodes "$N_EPISODES" \
    --output-dir "$OUTPUT"

python -m quantflow.evaluation.report \
    --input "$OUTPUT/results.parquet" \
    --output-dir "$OUTPUT"

python -m quantflow.evaluation.plots \
    --input "$OUTPUT/results.parquet" \
    --output-dir "$OUTPUT/plots"

echo ""
echo "Done. Results in $OUTPUT/"
echo "  $OUTPUT/results.parquet"
echo "  $OUTPUT/trajectories.parquet"
echo "  $OUTPUT/report.txt"
echo "  $OUTPUT/plots/*.png"
