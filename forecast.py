"""Cash-flow forecasting and actionable alerts.

build_cash_forecast projects a day-by-day cash balance for a 30, 60 or 90 day
horizon from unpaid invoices (inflows) and scheduled bills (outflows), with
optional what-if receipt delays and a one-off extra expense.

generate_actionable_alerts turns that forecast and the receivables summary
into a prioritised list of alerts, each with a concrete next action.
"""

from datetime import date

import pandas as pd

from data_manager import format_inr

VALID_HORIZONS = (30, 60, 90)
SEVERITY_RANK = {"Critical": 0, "High": 1, "Medium": 2}
FORECAST_COLUMNS = [
    "date",
    "opening_balance",
    "inflows",
    "outflows",
    "closing_balance",
    "inflow_invoices",
    "outflow_items",
]
ALERT_COLUMNS = [
    "priority",
    "severity",
    "date",
    "alert",
    "next_action",
    "invoice_id",
    "client_name",
    "amount_inr",
    "days_overdue",
]


def _join_ids(items):
    """Comma-join the non-empty values in items."""
    return ", ".join(str(item) for item in items if str(item) != "")


def _delay_targets(unpaid, overdue_mask, invoices_df, delay_client_id,
                   high_risk_invoice_ids):
    """Return a boolean Series marking the unpaid invoices a what-if delay applies to."""
    if delay_client_id is None:
        return pd.Series(False, index=unpaid.index)
    if delay_client_id == "ALL_OVERDUE":
        return overdue_mask
    if delay_client_id == "HIGH_RISK":
        if high_risk_invoice_ids is None:
            raise ValueError(
                "high_risk_invoice_ids is required when delay_client_id is 'HIGH_RISK'"
            )
        return unpaid["invoice_id"].isin(list(high_risk_invoice_ids))
    if delay_client_id in set(invoices_df["client_id"]):
        return unpaid["client_id"] == delay_client_id
    raise ValueError(f"Unknown delay_client_id: {delay_client_id!r}")


def build_cash_forecast(invoices_df, bills_df, starting_cash, horizon_days=60,
                        delay_client_id=None, delay_days=0, extra_expense=0,
                        as_of=None, high_risk_invoice_ids=None):
    """Project a daily cash balance over the horizon.

    Unpaid invoices (status != "Paid") are received on their due date, or on
    as_of if already overdue. Bills due inside the horizon are outflows on
    their due date. delay_client_id/delay_days shift the receipt dates of the
    targeted invoices later; extra_expense is an outflow on as_of.

    Returns a DataFrame with columns: date, opening_balance, inflows, outflows,
    closing_balance, inflow_invoices, outflow_items (one row per day).
    """
    if horizon_days not in VALID_HORIZONS:
        raise ValueError(
            f"horizon_days must be one of {VALID_HORIZONS}, got {horizon_days}"
        )
    if delay_days < 0:
        raise ValueError("delay_days must be >= 0")
    if extra_expense < 0:
        raise ValueError("extra_expense must be >= 0")

    as_of = pd.Timestamp(date.today() if as_of is None else as_of).normalize()
    end = as_of + pd.Timedelta(days=horizon_days - 1)
    dates = pd.date_range(as_of, end, freq="D")

    # Inflows: unpaid invoices, overdue ones received today, optional delay.
    unpaid = invoices_df[invoices_df["status"] != "Paid"].assign(
        due_date=lambda df: pd.to_datetime(df["due_date"]).dt.normalize()
    )
    overdue_mask = unpaid["due_date"] < as_of
    receipt = unpaid["due_date"].where(~overdue_mask, as_of)
    targeted = _delay_targets(
        unpaid, overdue_mask, invoices_df, delay_client_id, high_risk_invoice_ids
    )
    receipt = receipt.where(~targeted, receipt + pd.Timedelta(days=delay_days))
    receipts = unpaid.assign(receipt_date=receipt)
    receipts = receipts[receipts["receipt_date"] <= end]
    inflow_groups = receipts.groupby("receipt_date")
    inflows = inflow_groups["amount_inr"].sum().reindex(dates, fill_value=0)
    inflow_ids = inflow_groups["invoice_id"].agg(_join_ids).reindex(dates, fill_value="")

    # Outflows: bills due inside the horizon, plus any one-off extra expense.
    bills = bills_df.assign(
        due_date=lambda df: pd.to_datetime(df["due_date"]).dt.normalize()
    )
    bills = bills[(bills["due_date"] >= as_of) & (bills["due_date"] <= end)]
    bill_groups = bills.groupby("due_date")
    outflows = bill_groups["amount_inr"].sum().reindex(dates, fill_value=0)
    outflow_ids = bill_groups["bill_id"].agg(_join_ids).reindex(dates, fill_value="")
    if extra_expense:
        outflows.loc[as_of] += extra_expense
        outflow_ids.loc[as_of] = _join_ids([outflow_ids.loc[as_of], "Extra expense"])

    # Running balance: closing = starting cash + cumulative net flow.
    net = inflows - outflows
    closing = starting_cash + net.cumsum()
    opening = closing - net

    return pd.DataFrame({
        "date": dates.date,
        "opening_balance": opening.to_numpy(),
        "inflows": inflows.to_numpy(),
        "outflows": outflows.to_numpy(),
        "closing_balance": closing.to_numpy(),
        "inflow_invoices": inflow_ids.to_numpy(),
        "outflow_items": outflow_ids.to_numpy(),
    })[FORECAST_COLUMNS].reset_index(drop=True)


def _alert_row(severity, alert_date, alert, next_action, invoice_id="",
               client_name="", amount=0, days_overdue=0):
    """Build one alert record (priority is filled in later)."""
    return {
        "severity": severity,
        "date": alert_date,
        "alert": alert,
        "next_action": next_action,
        "invoice_id": invoice_id,
        "client_name": client_name,
        "amount_inr": amount,
        "days_overdue": days_overdue,
    }


def generate_actionable_alerts(forecast_df, receivables_df, safety_buffer=25000,
                               as_of=None):
    """Build a prioritised alert table from the forecast and receivables.

    Raises a Critical/High cash-breach alert on the first forecast day whose
    closing balance is below safety_buffer, plus a High/Medium alert for each
    overdue receivable. Sorted by severity, then cash-breach first, then
    days_overdue and amount descending. Returns an empty DataFrame with
    ALERT_COLUMNS when there is nothing to report.
    """
    as_of_ts = pd.Timestamp(date.today() if as_of is None else as_of)
    entries = []  # (sort_key, alert_record)

    # Cash breach: first forecast day (from as_of) below the safety buffer.
    forecast_dates = pd.to_datetime(forecast_df["date"])
    future = forecast_df[forecast_dates >= as_of_ts]
    breaches = future[future["closing_balance"] < safety_buffer]
    overdue = receivables_df[receivables_df["days_overdue"] > 0].sort_values(
        ["days_overdue", "amount_inr"], ascending=[False, False]
    )
    if not breaches.empty:
        breach = breaches.iloc[0]
        breach_date = pd.Timestamp(breach["date"]).date()
        closing = int(round(breach["closing_balance"]))
        severity = "Critical" if closing < 0 else "High"

        if not overdue.empty:
            top = overdue.iloc[0]
            next_action = (
                f"Chase {top['client_name']} Invoice #{top['invoice_id']} "
                f"({format_inr(int(top['amount_inr']))}, "
                f"{int(top['days_overdue'])} days overdue) today"
            )
        else:
            due_ts = pd.to_datetime(receivables_df["due_date"])
            candidates = receivables_df[
                (due_ts <= pd.Timestamp(breach_date))
                & (receivables_df["status"] != "Paid")
            ].sort_values("amount_inr", ascending=False)
            if not candidates.empty:
                top = candidates.iloc[0]
                next_action = (
                    f"Follow up {top['client_name']} Invoice #{top['invoice_id']} "
                    f"({format_inr(int(top['amount_inr']))}, "
                    f"due {pd.Timestamp(top['due_date']):%d %b %Y}) "
                    f"before {breach_date:%d %b %Y}"
                )
            else:
                next_action = (
                    f"Defer or reduce upcoming outflows before {breach_date:%d %b %Y}"
                )

        alert = (
            f"Cash drops below {format_inr(int(safety_buffer))} on "
            f"{breach_date:%d %b %Y} (projected balance {format_inr(closing)})"
        )
        sort_key = (SEVERITY_RANK[severity], 0, 0, 0)
        entries.append((sort_key, _alert_row(
            severity, breach_date, alert, next_action
        )))

    # Overdue receivables: one alert each.
    for _, row in overdue.iterrows():
        days = int(row["days_overdue"])
        amount = int(row["amount_inr"])
        severity = "High" if days > 30 else "Medium"
        client = row["client_name"]
        invoice_id = row["invoice_id"]
        alert = (
            f"{client} owes {format_inr(amount)} on Invoice #{invoice_id}, "
            f"{days} days overdue"
        )
        next_action = (
            f"Send a payment reminder to {client} for Invoice #{invoice_id} today"
        )
        sort_key = (SEVERITY_RANK[severity], 1, -days, -amount)
        entries.append((sort_key, _alert_row(
            severity,
            pd.Timestamp(row["due_date"]).date(),
            alert,
            next_action,
            invoice_id=invoice_id,
            client_name=client,
            amount=amount,
            days_overdue=days,
        )))

    entries.sort(key=lambda entry: entry[0])
    rows = [
        {"priority": priority, **record}
        for priority, (_, record) in enumerate(entries, start=1)
    ]
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)
