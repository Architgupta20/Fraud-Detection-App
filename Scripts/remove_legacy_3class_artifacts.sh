#!/usr/bin/env bash
# Remove datasets and models built with Low/Medium/High (pre v2.1 binary).
# Safe to re-run; only deletes known artifact paths.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ARTIFACTS=(
  "Data/Scored_Datasets/fraud_risk_scored_prescribers.csv"
  "Data/Model_Data/fraud_detection_gbt_sklearn_predictions.csv"
  "Data/Model_Data/fraud_detection_xgb_predictions.csv"
  "Models/gbt_sklearn.pkl"
  "Models/xgb_calibrated.pkl"
)

echo "Removing legacy 3-class artifacts under $ROOT ..."
for f in "${ARTIFACTS[@]}"; do
  if [[ -e "$f" ]]; then
    rm -f "$f"
    echo "  deleted: $f"
  fi
done

echo ""
echo "Regenerate Low/High pipeline outputs:"
echo "  export BASE_DIR=\"\$(pwd)\""
echo "  python run_pipeline.py score"
echo "  python Models/train_xgb.py --sample-frac 1.0 --strict-rules-version"
echo "  python Models/train_sklearn.py --sample-frac 1.0 --strict-rules-version"
