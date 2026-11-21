"""DevCash: single-file Streamlit dashboard for the One-Person Back Office.

Run with:  streamlit run app.py
All money is INR (₹). Backend logic lives in data_manager, ml_risk, forecast,
reminders and seed_data; this file only wires them to the UI.
"""

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import seed_data
from data_manager import (
    add_invoice,
    format_inr,
    get_receivables_summary,
    load_bills,
    load_clients,
    load_invoices,
    mark_invoice_paid,
)
from forecast import build_cash_forecast, generate_actionable_alerts
from ml_risk import score_unpaid_invoices, train_late_payment_model
from reminders import draft_bad_day_batch, draft_reminder

st.set_page_config(page_title="DevCash", layout="wide")

TODAY = date.today()
FOOTER = "Demo uses 100% synthetic data."
SPECIAL_TARGETS = {
    "ALL_OVERDUE": "All overdue invoices",
    "HIGH_RISK": "High-risk invoices",
}
FLASH_RENDERERS = {"success": st.success, "error": st.error}
ALERT_RENDERERS = {"Critical": st.error, "High": st.warning, "Medium": st.info}


def fmt_date(value):
    """Format a date-like value as 'DD Mon YYYY', or a dash when missing."""
    ts = pd.Timestamp(value) if value is not None else pd.NaT
    return "—" if pd.isna(ts) else ts.strftime("%d %b %Y")


def first_dip(forecast_df, safety_buffer):
    """Return the first forecast row whose closing balance is below the buffer, or None."""
    hits = forecast_df.loc[forecast_df["closing_balance"] < safety_buffer]
    return None if hits.empty else hits.iloc[0]


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    starting_cash = st.number_input(
        "Starting Cash Balance (₹)", min_value=0, value=120000, step=1000
    )
    safety_buffer = st.number_input(
        "Safety Buffer (₹)", min_value=0, value=25000, step=1000
    )
    horizon = st.selectbox("Forecast Horizon (days)", [30, 60, 90], index=2)
    st.divider()
    if st.button("Reset to Sample Data"):
        try:
            seed_data.main()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state["flash"] = ("success", "Sample data reset.")
            st.rerun()
        except Exception as exc:  # show the message, never a traceback
            st.error(f"Reset failed: {exc}")
    st.caption(FOOTER)

# ---------------------------------------------------------------- header
st.title("DevCash")
st.markdown("*The One-Person Back Office for Freelance Developers*")

flash = st.session_state.pop("flash", None)
if flash:
    FLASH_RENDERERS[flash[0]](flash[1])

# ---------------------------------------------------------------- data
try:
    clients = load_clients()
    invoices = load_invoices()
    bills = load_bills()
    model = train_late_payment_model(invoices, clients)
    scored = score_unpaid_invoices(invoices, clients, as_of=TODAY, model=model)
    receivables = get_receivables_summary(invoices, clients, as_of=TODAY)
    baseline = build_cash_forecast(
        invoices, bills, starting_cash, horizon_days=horizon, as_of=TODAY
    )
    alerts = generate_actionable_alerts(
        baseline, receivables, safety_buffer=safety_buffer, as_of=TODAY
    )
except Exception as exc:
    st.error(f"Could not load or process the data files: {exc}")
    st.stop()

client_names = dict(zip(clients["client_id"], clients["client_name"]))
client_contacts = dict(zip(clients["client_id"], clients["contact_name"]))
high_risk_ids = scored.loc[scored["risk_tier"] == "High", "invoice_id"].tolist()

tab_cmd, tab_forecast, tab_invoices, tab_followup = st.tabs(
    [
        "Command Center",
        "Cash Forecast & What-If",
        "Invoices & Quick Actions",
        "Smart Follow-Ups & Bad Day Mode",
    ]
)

# ---------------------------------------------------------------- tab 1
with tab_cmd:
    st.subheader("Who Owes Me")
    total_owed = float(receivables["amount_inr"].sum())
    overdue_amount = float(receivables.loc[receivables["days_overdue"] > 0, "amount_inr"].sum())
    lowest_cash = float(baseline["closing_balance"].min())
    high_risk_count = int((scored["risk_tier"] == "High").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Owed (₹)", format_inr(total_owed))
    k2.metric("Overdue Amount (₹)", format_inr(overdue_amount))
    k3.metric("Projected Lowest Cash (₹)", format_inr(lowest_cash))
    k4.metric("High-Risk Invoices", high_risk_count)

    st.markdown("#### Prioritized Actionable Alerts")
    if alerts.empty:
        st.success("No alerts: no cash breach and no overdue receivables.")
    for _, alert in alerts.iterrows():
        renderer = ALERT_RENDERERS.get(alert["severity"], st.info)
        renderer(f"**{alert['severity']}**: {alert['alert']}  \n**Next action:** {alert['next_action']}")

    st.markdown("#### Receivables")
    if model.is_fallback:
        st.caption("Risk model is using the fallback probability (too few Paid invoices to train).")
    table = pd.DataFrame(
        {
            "Client": scored["client_name"],
            "Invoice ID": scored["invoice_id"],
            "Amount (₹)": scored["amount_inr"].map(format_inr),
            "Due Date": pd.to_datetime(scored["due_date"]).dt.date,
            "Days Overdue": scored["days_overdue"],
            "ML Risk Tier": scored["risk_tier"],
            "Late Probability (%)": scored["late_probability"],
            "Reason": scored["reason"],
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)

# ---------------------------------------------------------------- tab 2
with tab_forecast:
    st.subheader("30–90 Day Cash Forecast & What-If")
    client_options = list(SPECIAL_TARGETS) + list(client_names)

    def target_label(key):
        return SPECIAL_TARGETS.get(key) or client_names.get(key, key)

    c1, c2, c3 = st.columns(3)
    target = c1.selectbox(
        "Delay applies to",
        client_options,
        index=0,  # default: ALL_OVERDUE, the most common late-payment scenario
        format_func=target_label,
    )
    delay_days = c2.slider("Delay Days", min_value=0, max_value=45, value=14)
    extra_expense = c3.number_input(
        "Unexpected Extra Expense (₹)", min_value=0, value=0, step=1000
    )

    try:
        whatif = build_cash_forecast(
            invoices,
            bills,
            starting_cash,
            horizon_days=horizon,
            delay_client_id=target,
            delay_days=delay_days,
            extra_expense=extra_expense,
            as_of=TODAY,
            high_risk_invoice_ids=high_risk_ids,
        )
    except Exception as exc:
        st.error(f"Could not build the what-if forecast: {exc}")
        st.stop()

    base_x = pd.to_datetime(baseline["date"])
    wi_x = pd.to_datetime(whatif["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base_x, y=baseline["closing_balance"], mode="lines",
        name="Baseline", line=dict(color="#4C78A8", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=wi_x, y=whatif["closing_balance"], mode="lines",
        name="What-If", line=dict(color="#54A24B", width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.add_hline(
        y=safety_buffer, line_dash="dash", line_color="orange",
        annotation_text="Safety buffer", annotation_position="top left",
    )

    base_dip = first_dip(baseline, safety_buffer)
    wi_dip = first_dip(whatif, safety_buffer)
    if base_dip is not None:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(base_dip["date"])], y=[base_dip["closing_balance"]],
            mode="markers", name="First dip (Baseline)",
            marker=dict(color="grey", size=11, symbol="circle-open"),
        ))
    if wi_dip is not None:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(wi_dip["date"])], y=[wi_dip["closing_balance"]],
            mode="markers", name="First dip (What-If)",
            marker=dict(color="red", size=13, symbol="x"),
        ))
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Closing balance (₹)", legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, width="stretch")

    if wi_dip is None:
        st.success("No dip below the safety buffer or ₹0 over this horizon (What-If).")
    else:
        dip_amount = format_inr(wi_dip["closing_balance"])
        dip_when = fmt_date(wi_dip["date"])
        if wi_dip["closing_balance"] < 0:
            st.error(f"What-If: cash goes below ₹0 on {dip_when}, reaching {dip_amount}.")
        else:
            st.warning(
                f"What-If: cash first dips below the safety buffer on {dip_when}, "
                f"at {dip_amount}."
            )
    if base_dip is not None:
        st.caption(f"Baseline (no delay, no expense) first dips on {fmt_date(base_dip['date'])}.")

    m1, m2 = st.columns(2)
    base_min = float(baseline["closing_balance"].min())
    wi_min = float(whatif["closing_balance"].min())
    m1.metric("Baseline lowest balance", format_inr(base_min))
    m2.metric(
        "What-if lowest balance", format_inr(wi_min),
        delta=format_inr(wi_min - base_min),
    )

# ---------------------------------------------------------------- tab 3
with tab_invoices:
    st.subheader("Create New Invoice")
    with st.form("create_invoice"):
        f_client = st.selectbox(
            "Client", list(client_names), format_func=lambda cid: client_names[cid]
        )
        f_description = st.text_input("Description")
        f_amount = st.number_input("Amount (₹)", min_value=0, value=50000, step=1000)
        f_term = st.number_input("Payment Term Days", min_value=0, value=30, step=1)
        submitted = st.form_submit_button("Create Invoice")

    if submitted:
        try:
            new_id = add_invoice(
                f_client, f_description, f_amount, TODAY, term_days=int(f_term)
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not create invoice: {exc}")
        else:
            st.session_state["flash"] = ("success", f"Created invoice {new_id}.")
            st.rerun()

    st.subheader("Invoices")
    header = st.columns([1.2, 2.2, 3.2, 1.5, 1.5, 1.2, 1.6])
    for col, label in zip(header, ["Invoice", "Client", "Description", "Amount", "Due", "Status", ""]):
        col.markdown(f"**{label}**")

    for _, inv in invoices.iterrows():
        cols = st.columns([1.2, 2.2, 3.2, 1.5, 1.5, 1.2, 1.6])
        cols[0].write(inv["invoice_id"])
        cols[1].write(client_names.get(inv["client_id"], inv["client_id"]))
        cols[2].write(inv["description"])
        cols[3].write(format_inr(inv["amount_inr"]))
        cols[4].write(fmt_date(inv["due_date"]))
        cols[5].write(inv["status"])
        if inv["status"] != "Paid":
            if cols[6].button("Mark as Paid", key=f"paid_{inv['invoice_id']}"):
                try:
                    mark_invoice_paid(inv["invoice_id"], TODAY)
                except Exception as exc:
                    st.error(f"Could not mark {inv['invoice_id']} as paid: {exc}")
                else:
                    st.session_state["flash"] = ("success", f"{inv['invoice_id']} marked as paid.")
                    st.rerun()

# ---------------------------------------------------------------- tab 4
with tab_followup:
    st.subheader("Smart Follow-Ups")
    channel = st.radio("Channel", ["WhatsApp", "Email"], horizontal=True)
    if receivables.empty:
        st.info("No unpaid invoices to follow up on.")
    else:
        labels = [
            f"{r.invoice_id} — {r.client_name} — {format_inr(r.amount_inr)} — "
            f"{int(r.days_overdue)} days overdue"
            for r in receivables.itertuples()
        ]
        pick = st.selectbox(
            "Unpaid invoice", list(range(len(labels))), format_func=lambda i: labels[i]
        )
        row = receivables.iloc[pick]
        contact = client_contacts.get(row["client_id"], "")
        try:
            message = draft_reminder(
                row["client_name"],
                row["invoice_id"],
                row["amount_inr"],
                row["due_date"],
                int(row["days_overdue"]),
                channel=channel,
                contact_name=contact or None,
            )
            st.code(message, language=None)
        except Exception as exc:
            st.error(f"Could not draft the reminder: {exc}")

    st.divider()
    st.subheader("Bad Day Mode")
    if st.button("Bad Day Mode (Chase All Overdue)"):
        try:
            batch = draft_bad_day_batch(receivables, channel=channel)
        except Exception as exc:
            st.error(f"Could not draft the batch: {exc}")
        else:
            st.info(batch["summary"])
            b1, b2 = st.columns(2)
            b1.metric("Clients to chase", batch["clients"])
            b2.metric("Total overdue", format_inr(batch["total_overdue"]))
            for msg in batch["messages"].itertuples():
                st.markdown(
                    f"**{msg.client_name}** · Invoices {msg.invoice_ids} · "
                    f"{format_inr(msg.amount_inr)} · oldest {msg.oldest_days_overdue} days overdue"
                )
                st.code(msg.message, language=None)

st.divider()
st.caption(FOOTER)
