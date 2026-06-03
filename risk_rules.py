"""
Single source of truth for prescriber risk rules and ML training alignment.

- Scoring: run_pipeline.py score, Scripts/fraud_risk_scoring.py → apply_risk_scoring_spark()
- Training: Models/ml_common.py, train_xgb.py, train_sklearn.py → ML_FEATURE_COLS, LABEL_*
- UI: Streamlit → evaluate_rules_for_row()

After changing rules here: python run_pipeline.py score, then retrain models.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

RULES_VERSION = "2.0.0"

# --- Labels produced by scoring (training targets) ---
LABEL_COL = "fraud_risk_category"
LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

# Columns that define rules / labels — never use as ML features
RULE_INPUT_COLUMNS: List[str] = [
    "payment_to_drug_cost_ratio",
    "opioid_claims",
    "high_payment_flag",
    "high_opioid_flag",
    "peer_deviation_score",
    "elderly_focus_flag",
    "antibiotic_claim_ratio",
    "total_payment_amount",
    "opioid_volume_pct_flag",
    "peer_outlier_pct_flag",
    "payment_spiky_pct_flag",
    "total_payments_pct_flag",
    "rules_fired",
    "rules_version",
    "risk_points",
    "fraud_risk_score",
]

# Allowed model features (must not overlap RULE_INPUT_COLUMNS except noted in docs)
ML_FEATURE_COLS: List[str] = [
    "total_claims",
    "total_drug_cost",
    "opioid_cost",
    "antibiotic_claims",
    "avg_risk_score",
    "payment_variability",  # also used in rules; keep for signal diversity — see docs/LABEL_LEAKAGE.md
    "adjusted_risk_payment",
]

SCORED_OUTPUT_COLUMNS: List[str] = [
    "risk_points",
    "rules_fired",
    "rules_version",
    "fraud_risk_category",
    "fraud_risk_score",
    *[
        "opioid_volume_pct_flag",
        "peer_outlier_pct_flag",
        "payment_spiky_pct_flag",
        "total_payments_pct_flag",
    ],
]

# --- Documented thresholds (fixed fallbacks; Spark also uses within-specialty percentiles) ---
PAYMENT_RATIO_HIGH = 1.0
OPIOID_CLAIMS_ABSOLUTE = 100
PEER_DEVIATION_ABSOLUTE = 5.0
PAYMENT_VARIABILITY_ABSOLUTE = 3.0
ANTIBIOTIC_RATIO_HIGH = 0.25
ANTIBIOTIC_CLAIMS_MIN = 50
TOTAL_PAYMENT_ABSOLUTE = 50_000.0
PERCENTILE_CUTOFF = 0.95

HIGH_CATEGORY_MIN_POINTS = 4
MEDIUM_CATEGORY_MIN_POINTS = 2

# rule_id -> points when fired
RULE_POINTS: Dict[str, int] = {
    "payment_ratio_high": 2,
    "opioid_volume_high": 2,
    "high_payment_flag": 1,
    "high_opioid_flag": 1,
    "peer_outlier": 1,
    "payment_spiky": 1,
    "antibiotic_heavy": 1,
    "elderly_focus": 1,
    "total_payments_high": 1,
}

RULE_LABELS: Dict[str, str] = {
    "payment_ratio_high": "High payment vs drug cost",
    "opioid_volume_high": "Heavy opioid prescribing",
    "high_payment_flag": "High average payment size",
    "high_opioid_flag": "Opioid-majority prescribing",
    "peer_outlier": "Payment outlier vs peers",
    "payment_spiky": "Spiky payment pattern",
    "antibiotic_heavy": "Heavy antibiotic use",
    "elderly_focus": "Elderly-focused practice",
    "total_payments_high": "High total industry payments",
}


def _f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    val = row.get(key, default)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def evaluate_rules_for_row(
    row: Dict[str, Any],
    *,
    use_percentile_flags: bool = False,
) -> Tuple[int, List[str], List[Tuple[str, str]]]:
    """
    Return (risk_points, rule_ids, display_tuples) for one prescriber row.

    When use_percentile_flags=True, row may include precomputed *_pct_flag columns
    from the scored CSV (Spark output). Otherwise only absolute thresholds apply.
    """
    fired: List[str] = []

    if _f(row, "payment_to_drug_cost_ratio") > PAYMENT_RATIO_HIGH:
        fired.append("payment_ratio_high")

    opioid_pct = row.get("opioid_volume_pct_flag")
    if use_percentile_flags and opioid_pct in (1, "1", True):
        fired.append("opioid_volume_high")
    elif _f(row, "opioid_claims") > OPIOID_CLAIMS_ABSOLUTE:
        fired.append("opioid_volume_high")

    if _f(row, "high_payment_flag") == 1:
        fired.append("high_payment_flag")
    if _f(row, "high_opioid_flag") == 1:
        fired.append("high_opioid_flag")

    peer_pct = row.get("peer_outlier_pct_flag")
    if use_percentile_flags and peer_pct in (1, "1", True):
        fired.append("peer_outlier")
    elif _f(row, "peer_deviation_score") > PEER_DEVIATION_ABSOLUTE:
        fired.append("peer_outlier")

    var_pct = row.get("payment_spiky_pct_flag")
    if use_percentile_flags and var_pct in (1, "1", True):
        fired.append("payment_spiky")
    elif _f(row, "payment_variability") > PAYMENT_VARIABILITY_ABSOLUTE:
        fired.append("payment_spiky")

    if (
        _f(row, "antibiotic_claim_ratio") > ANTIBIOTIC_RATIO_HIGH
        and _f(row, "antibiotic_claims") > ANTIBIOTIC_CLAIMS_MIN
    ):
        fired.append("antibiotic_heavy")

    if _f(row, "elderly_focus_flag") == 1:
        fired.append("elderly_focus")

    pay_pct = row.get("total_payments_pct_flag")
    if use_percentile_flags and pay_pct in (1, "1", True):
        fired.append("total_payments_high")
    elif _f(row, "total_payment_amount") > TOTAL_PAYMENT_ABSOLUTE:
        fired.append("total_payments_high")

    points = sum(RULE_POINTS[r] for r in fired)
    display = [(RULE_LABELS[r], _rule_detail(r, row)) for r in fired]
    return points, fired, display


def _rule_detail(rule_id: str, row: Dict[str, Any]) -> str:
    details = {
        "payment_ratio_high": (
            f"Payment-to-drug-cost ratio is {_f(row, 'payment_to_drug_cost_ratio'):.2f} (>{PAYMENT_RATIO_HIGH})."
        ),
        "opioid_volume_high": f"Opioid claims are {_f(row, 'opioid_claims'):.0f} (high vs peers or >{OPIOID_CLAIMS_ABSOLUTE}).",
        "high_payment_flag": "Average payment size exceeds $1,000.",
        "high_opioid_flag": "More than half of claims are opioid-related.",
        "peer_outlier": f"Peer deviation score is {_f(row, 'peer_deviation_score'):.2f}.",
        "payment_spiky": f"Payment variability is {_f(row, 'payment_variability'):.2f}.",
        "antibiotic_heavy": "Antibiotic ratio and volume are both elevated.",
        "elderly_focus": "Average patient age is above 70.",
        "total_payments_high": f"Total industry payments are ${_f(row, 'total_payment_amount'):,.0f}.",
    }
    return details.get(rule_id, rule_id)


def category_from_points(points: int) -> str:
    if points >= HIGH_CATEGORY_MIN_POINTS:
        return "High"
    if points >= MEDIUM_CATEGORY_MIN_POINTS:
        return "Medium"
    return "Low"


def validate_ml_feature_cols() -> None:
    """Ensure training features do not include rule-input columns."""
    overlap = set(ML_FEATURE_COLS) & set(RULE_INPUT_COLUMNS)
    if overlap:
        raise ValueError(f"ML_FEATURE_COLS overlap RULE_INPUT_COLUMNS: {sorted(overlap)}")


def check_scored_rules_version(scored_rules_version: str | None, *, strict: bool = False) -> None:
    """
    Warn or fail when the scored CSV was built with a different rules version.
    Call from training scripts after loading data.
    """
    if scored_rules_version is None or scored_rules_version == "":
        print(
            f"WARNING: scored CSV has no rules_version column; "
            f"re-run `python run_pipeline.py score` (expected {RULES_VERSION})."
        )
        return
    if str(scored_rules_version) != RULES_VERSION:
        msg = (
            f"Scored CSV rules_version={scored_rules_version!r} "
            f"does not match risk_rules.RULES_VERSION={RULES_VERSION!r}. "
            "Re-run scoring before training."
        )
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}")


validate_ml_feature_cols()


def apply_risk_scoring_spark(df: "Any") -> "Any":
    """Add risk columns to a Spark DataFrame (enriched prescriber features)."""
    from pyspark.sql.functions import col, concat_ws, lit, when
    from pyspark.sql.window import Window
    from pyspark.sql.functions import percent_rank

    def pct_rank(column: str):
        window = Window.partitionBy("provider_type").orderBy(col(column).cast("double"))
        return percent_rank().over(window)

    df = df.withColumn("_pct_opioid", pct_rank("opioid_claims"))
    df = df.withColumn("_pct_payments", pct_rank("total_payment_amount"))
    df = df.withColumn("_pct_variability", pct_rank("payment_variability"))
    df = df.withColumn("_pct_peer", pct_rank("peer_deviation_score"))

    cond = {
        "payment_ratio_high": col("payment_to_drug_cost_ratio") > lit(PAYMENT_RATIO_HIGH),
        "opioid_volume_high": (col("opioid_claims") > lit(OPIOID_CLAIMS_ABSOLUTE))
        | (col("_pct_opioid") >= lit(PERCENTILE_CUTOFF)),
        "high_payment_flag": col("high_payment_flag") == lit(1),
        "high_opioid_flag": col("high_opioid_flag") == lit(1),
        "peer_outlier": (col("peer_deviation_score") > lit(PEER_DEVIATION_ABSOLUTE))
        | (col("_pct_peer") >= lit(PERCENTILE_CUTOFF)),
        "payment_spiky": (col("payment_variability") > lit(PAYMENT_VARIABILITY_ABSOLUTE))
        | (col("_pct_variability") >= lit(PERCENTILE_CUTOFF)),
        "antibiotic_heavy": (col("antibiotic_claim_ratio") > lit(ANTIBIOTIC_RATIO_HIGH))
        & (col("antibiotic_claims") > lit(ANTIBIOTIC_CLAIMS_MIN)),
        "elderly_focus": col("elderly_focus_flag") == lit(1),
        "total_payments_high": (col("total_payment_amount") > lit(TOTAL_PAYMENT_ABSOLUTE))
        | (col("_pct_payments") >= lit(PERCENTILE_CUTOFF)),
    }

    df = df.withColumn(
        "opioid_volume_pct_flag",
        when(col("_pct_opioid") >= lit(PERCENTILE_CUTOFF), lit(1)).otherwise(lit(0)),
    )
    df = df.withColumn(
        "peer_outlier_pct_flag",
        when(col("_pct_peer") >= lit(PERCENTILE_CUTOFF), lit(1)).otherwise(lit(0)),
    )
    df = df.withColumn(
        "payment_spiky_pct_flag",
        when(col("_pct_variability") >= lit(PERCENTILE_CUTOFF), lit(1)).otherwise(lit(0)),
    )
    df = df.withColumn(
        "total_payments_pct_flag",
        when(col("_pct_payments") >= lit(PERCENTILE_CUTOFF), lit(1)).otherwise(lit(0)),
    )

    points_expr = lit(0)
    for rule_id, pts in RULE_POINTS.items():
        points_expr = points_expr + when(cond[rule_id], lit(pts)).otherwise(lit(0))

    df = df.withColumn("risk_points", points_expr)
    df = df.withColumn(
        "rules_fired",
        concat_ws(
            "|",
            *[
                when(cond[rule_id], lit(rule_id)).otherwise(lit(None))
                for rule_id in RULE_POINTS
            ],
        ),
    )
    df = df.withColumn("rules_version", lit(RULES_VERSION))

    df = df.withColumn(
        "fraud_risk_category",
        when(col("risk_points") >= lit(HIGH_CATEGORY_MIN_POINTS), lit("High"))
        .when(col("risk_points") >= lit(MEDIUM_CATEGORY_MIN_POINTS), lit("Medium"))
        .otherwise(lit("Low")),
    )
    # Backward-compatible alias used by older scripts / ML labels
    df = df.withColumn("fraud_risk_score", col("risk_points"))

    drop_cols = ["_pct_opioid", "_pct_payments", "_pct_variability", "_pct_peer"]
    return df.drop(*drop_cols)


def streamlit_rule_checks() -> List[Tuple[str, Callable[[Dict], bool], str]]:
    """Checks for manual single-record UI (absolute thresholds + optional row flags)."""
    return [
        (
            RULE_LABELS["payment_ratio_high"],
            lambda r: _f(r, "payment_to_drug_cost_ratio") > PAYMENT_RATIO_HIGH,
            _rule_detail("payment_ratio_high", {}),
        ),
        (
            RULE_LABELS["opioid_volume_high"],
            lambda r: _f(r, "opioid_claims") > OPIOID_CLAIMS_ABSOLUTE
            or r.get("opioid_volume_pct_flag") in (1, "1", True),
            "Opioid volume is high (absolute or peer percentile flag).",
        ),
        (
            RULE_LABELS["high_payment_flag"],
            lambda r: _f(r, "high_payment_flag") == 1,
            _rule_detail("high_payment_flag", {}),
        ),
        (
            RULE_LABELS["high_opioid_flag"],
            lambda r: _f(r, "high_opioid_flag") == 1,
            _rule_detail("high_opioid_flag", {}),
        ),
        (
            RULE_LABELS["peer_outlier"],
            lambda r: _f(r, "peer_deviation_score") > PEER_DEVIATION_ABSOLUTE
            or r.get("peer_outlier_pct_flag") in (1, "1", True),
            "Payments are high vs same provider type.",
        ),
        (
            RULE_LABELS["payment_spiky"],
            lambda r: _f(r, "payment_variability") > PAYMENT_VARIABILITY_ABSOLUTE
            or r.get("payment_spiky_pct_flag") in (1, "1", True),
            "Payment sizes are unusually spiky.",
        ),
        (
            RULE_LABELS["antibiotic_heavy"],
            lambda r: _f(r, "antibiotic_claim_ratio") > ANTIBIOTIC_RATIO_HIGH
            and _f(r, "antibiotic_claims") > ANTIBIOTIC_CLAIMS_MIN,
            _rule_detail("antibiotic_heavy", {}),
        ),
        (
            RULE_LABELS["elderly_focus"],
            lambda r: _f(r, "elderly_focus_flag") == 1,
            _rule_detail("elderly_focus", {}),
        ),
        (
            RULE_LABELS["total_payments_high"],
            lambda r: _f(r, "total_payment_amount") > TOTAL_PAYMENT_ABSOLUTE
            or r.get("total_payments_pct_flag") in (1, "1", True),
            "Total industry payments are high.",
        ),
    ]
