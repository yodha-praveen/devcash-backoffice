"""Late-payment risk scoring for unpaid invoices.

Approach
--------
A scikit-learn pipeline (StandardScaler + LogisticRegression) is trained on
historical invoices with status "Paid" and a non-null paid_date. Each invoice
is labelled late (1) only if paid more than GRACE_DAYS (5) days after due_date,
otherwise on time (0). Features
are FEATURE_COLUMNS: the client's avg_days_late, amount_inr, and
payment_term_days (due_date - issue_date).

Each unpaid invoice (status != "Paid") is scored with the model's late
probability, converted to a 0-100 percentage, plus a risk tier and a short
plain-English reason.

Fallback
--------
If there are fewer than MIN_TRAINING_ROWS training rows, or the training rows
contain only one label class, no pipeline is fitted. Instead a constant
probability is used: the observed late rate of the training rows, or 0.5 if
there is no training data. The returned LateModel has is_fallback=True.

Risk tier rule
--------------
late_probability < 40 -> "Low"; < 70 -> "Medium"; otherwise "High".
If days_overdue > 30 the tier is raised to at least "High".

Limitations
-----------
The training set is tiny (the sample has only 13 Paid invoices), so the model
is unstable and should be read as a rough indicator. avg_days_late is both a
feature and the historical signal that the label is derived from, so the model
is partly circular: it largely re-learns the client's past lateness.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_manager import format_inr  # noqa: F401  (available for UI callers)

FEATURE_COLUMNS = ["avg_days_late", "amount_inr", "payment_term_days"]
OUTPUT_COLUMNS = [
    "invoice_id",
    "client_id",
    "client_name",
    "amount_inr",
    "due_date",
    "days_overdue",
    "late_probability",
    "risk_tier",
    "reason",
]
MIN_TRAINING_ROWS = 2
LATE_HIGH_THRESHOLD_PCT = 70
LATE_MEDIUM_THRESHOLD_PCT = 40
OVERDUE_FORCE_HIGH_DAYS = 30
LATE_AVG_DAYS_REASON = 7
HIGH_VALUE_MULTIPLIER = 2
# Freelance grace period: only payments more than this many days late count
# as materially late (label 1) when training the model.
GRACE_DAYS = 5


@dataclass
class LateModel:
    """Result of training the late-payment model.

    pipeline:    fitted sklearn Pipeline, or None when falling back.
    base_rate:   observed late rate of the training rows (0.5 if no data).
    is_fallback: True when a constant probability is used instead of a model.
    """

    pipeline: Pipeline | None
    base_rate: float
    is_fallback: bool


def _build_features(invoices_df: pd.DataFrame, clients_df: pd.DataFrame) -> pd.DataFrame:
    """Join invoices to client data and compute the model features."""
    inv = invoices_df.loc[
        :,
        ["invoice_id", "client_id", "amount_inr", "issue_date", "due_date", "status", "paid_date"],
    ]
    cli = clients_df.loc[:, ["client_id", "client_name", "avg_days_late"]]
    merged = inv.merge(cli, on="client_id", how="left")

    term_days = (merged["due_date"] - merged["issue_date"]).dt.days.fillna(0).astype(int)
    merged = merged.assign(
        avg_days_late=merged["avg_days_late"].fillna(0).astype(int),
        amount_inr=merged["amount_inr"].fillna(0).astype(int),
        payment_term_days=term_days,
    )
    return merged


def train_late_payment_model(invoices_df: pd.DataFrame, clients_df: pd.DataFrame) -> LateModel:
    """Train the late-payment classifier on Paid invoices with a paid_date.

    Falls back to a constant probability (see module docstring) when there are
    fewer than MIN_TRAINING_ROWS rows or only one label class.
    """
    feats = _build_features(invoices_df, clients_df)
    paid = feats.loc[(feats["status"] == "Paid") & feats["paid_date"].notna()].copy()
    days_late = (paid["paid_date"] - paid["due_date"]).dt.days
    paid.loc[:, "label"] = (days_late > GRACE_DAYS).astype(int)

    n_rows = len(paid)
    base_rate = float(paid["label"].mean()) if n_rows > 0 else 0.5

    if n_rows < MIN_TRAINING_ROWS or paid["label"].nunique() < 2:
        return LateModel(pipeline=None, base_rate=base_rate, is_fallback=True)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(paid.loc[:, FEATURE_COLUMNS], paid["label"])
    return LateModel(pipeline=pipe, base_rate=base_rate, is_fallback=False)


def score_unpaid_invoices(
    invoices_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    as_of: date | None = None,
    model: LateModel | None = None,
) -> pd.DataFrame:
    """Score every unpaid invoice (status != "Paid").

    as_of:  reference date for days_overdue; defaults to today.
    model:  LateModel to use; trained from invoices_df/clients_df if None.

    Returns a DataFrame with columns OUTPUT_COLUMNS, sorted by late_probability
    then days_overdue, both descending. Empty (with columns) if nothing is unpaid.

    late_probability is the model probability * 100, rounded to 1 decimal.
    risk_tier: "Low" if < 40, "Medium" if < 70, else "High"; raised to at
    least "High" when days_overdue > 30.
    reason joins applicable factors with " + ": "Client averages N days late"
    (avg_days_late >= 7), "high invoice value" (amount >= 2x the median of all
    invoice amounts), "N days overdue" (days_overdue > 0). If none apply,
    "Typical client and invoice profile".
    """
    if model is None:
        model = train_late_payment_model(invoices_df, clients_df)

    feats = _build_features(invoices_df, clients_df)
    unpaid = feats.loc[feats["status"] != "Paid"].copy()
    if unpaid.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    as_of_ts = pd.Timestamp(as_of if as_of is not None else date.today()).normalize()
    days_overdue = (
        (as_of_ts - unpaid["due_date"]).dt.days.fillna(0).clip(lower=0).astype(int)
    )

    if model.pipeline is not None:
        probs = model.pipeline.predict_proba(unpaid.loc[:, FEATURE_COLUMNS])[:, 1]
    else:
        probs = np.full(len(unpaid), model.base_rate, dtype=float)
    late_prob = np.round(np.asarray(probs, dtype=float) * 100, 1)

    tier = np.where(
        late_prob >= LATE_HIGH_THRESHOLD_PCT,
        "High",
        np.where(late_prob >= LATE_MEDIUM_THRESHOLD_PCT, "Medium", "Low"),
    )
    overdue_arr = days_overdue.to_numpy()
    tier = np.where(overdue_arr > OVERDUE_FORCE_HIGH_DAYS, "High", tier)

    if len(invoices_df) > 0:
        high_value_threshold = HIGH_VALUE_MULTIPLIER * float(invoices_df["amount_inr"].median())
    else:
        high_value_threshold = np.inf

    reasons = []
    for avg_late, amount, overdue in zip(
        unpaid["avg_days_late"].to_numpy(),
        unpaid["amount_inr"].to_numpy(),
        overdue_arr,
    ):
        parts = []
        if avg_late >= LATE_AVG_DAYS_REASON:
            parts.append(f"Client averages {int(avg_late)} days late")
        if amount >= high_value_threshold:
            parts.append("high invoice value")
        if overdue > 0:
            parts.append(f"{int(overdue)} days overdue")
        reasons.append(" + ".join(parts) if parts else "Typical client and invoice profile")

    result = pd.DataFrame(
        {
            "invoice_id": unpaid["invoice_id"].to_numpy(),
            "client_id": unpaid["client_id"].to_numpy(),
            "client_name": unpaid["client_name"].to_numpy(),
            "amount_inr": unpaid["amount_inr"].to_numpy(),
            "due_date": unpaid["due_date"].to_numpy(),
            "days_overdue": overdue_arr,
            "late_probability": late_prob,
            "risk_tier": tier,
            "reason": reasons,
        },
        columns=OUTPUT_COLUMNS,
    )
    result = result.sort_values(
        ["late_probability", "days_overdue"], ascending=[False, False], kind="stable"
    )
    return result.reset_index(drop=True)
