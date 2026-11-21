"""Draft payment reminder messages for unpaid invoices.

Produces WhatsApp, SMS and Email reminder text for a single invoice, plus
batch helpers that work on receivables_df as returned by
data_manager.get_receivables_summary (unpaid rows only).
"""

import pandas as pd

from data_manager import format_inr

CHANNELS = ("WhatsApp", "SMS", "Email")
TONES = ("polite", "firm", "urgent")

POLITE_MAX_DAYS = 7    # 1 to 7 days overdue -> polite
FIRM_MAX_DAYS = 30     # 8 to 30 days overdue -> firm; above -> urgent
SMS_MAX_CHARS = 320

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

SIGN_OFF = "Thanks,\n[Your name]"

BATCH_COLUMNS = [
    "invoice_id", "client_name", "amount_inr", "days_overdue",
    "tone", "channel", "message",
]
CLIENT_COLUMNS = [
    "client_name", "invoice_ids", "amount_inr", "oldest_days_overdue",
    "tone", "channel", "message",
]


def _normalise_channel(channel):
    key = str(channel).strip().lower()
    for name in CHANNELS:
        if name.lower() == key:
            return name
    raise ValueError(
        f"channel must be one of {', '.join(CHANNELS)}; got {channel!r}"
    )


def _normalise_tone(tone):
    if tone is None:
        return None
    key = str(tone).strip().lower()
    if key in TONES:
        return key
    raise ValueError(
        f"tone must be None or one of {', '.join(TONES)}; got {tone!r}"
    )


def _auto_tone(days_overdue):
    if days_overdue <= POLITE_MAX_DAYS:
        return "polite"
    if days_overdue <= FIRM_MAX_DAYS:
        return "firm"
    return "urgent"


def _effective_tone(days_overdue, tone):
    """Tone actually used. Pre-due reminders (days <= 0) are always polite."""
    if days_overdue <= 0:
        return "polite"
    return tone or _auto_tone(days_overdue)


def _plural(count, word):
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _format_due(due_date):
    ts = pd.Timestamp(due_date)
    if pd.isna(ts):
        return "an unknown date"
    return f"{ts.day:02d} {MONTHS[ts.month - 1]} {ts.year}"


def _greeting(client_name, contact_name=None):
    if contact_name:
        return f"Hi {contact_name},"
    return f"Hi {client_name} team,"


def _single_parts(tone, days_overdue, invoice_id, amount_text, due_text):
    """Return (main paragraph, optional closing line) for one invoice."""
    if days_overdue <= 0:
        main = (f"This is a friendly reminder that Invoice #{invoice_id} "
                f"for {amount_text} is due on {due_text}.")
        closing = "Could you let us know if anything is needed to process it?"
        return main, closing

    days_text = _plural(days_overdue, "day")
    if tone == "polite":
        main = (f"This is a friendly reminder that Invoice #{invoice_id} "
                f"for {amount_text} was due on {due_text} and is now "
                f"{days_text} overdue.")
        closing = "Could you let us know if anything is needed from our side to process it?"
    elif tone == "firm":
        main = (f"Invoice #{invoice_id} for {amount_text} was due on "
                f"{due_text} and is now {days_text} past due. Please arrange "
                f"payment this week.")
        closing = "Could you confirm the expected payment date?"
    else:
        main = (f"Invoice #{invoice_id} for {amount_text} was due on "
                f"{due_text} and is now {days_text} overdue. Please arrange "
                f"payment within 2 working days and reply to confirm.")
        closing = ("New work may need to pause until the balance is cleared. "
                   "If there is a problem, please let us know so we can "
                   "resolve it together.")
    return main, closing


def _group_parts(tone, days_overdue, invoice_ids, amount_text, due_text):
    """Return (main paragraph, optional closing line) for a client's invoices."""
    count = len(invoice_ids)
    refs = ", ".join(f"#{i}" for i in invoice_ids)
    phrase = f"{'Invoice' if count == 1 else 'Invoices'} {refs} totalling {amount_text}"
    verb = "is" if count == 1 else "are"
    pronoun = "it" if count == 1 else "them"

    if days_overdue <= 0:
        main = (f"This is a friendly reminder that {phrase} {verb} due "
                f"on {due_text}.")
        closing = f"Could you let us know if anything is needed to process {pronoun}?"
        return main, closing

    days_text = _plural(days_overdue, "day")
    if tone == "polite":
        main = (f"This is a friendly reminder that {phrase} {verb} overdue, "
                f"the oldest by {days_text}.")
        closing = f"Could you let us know if anything is needed from our side to process {pronoun}?"
    elif tone == "firm":
        main = (f"{phrase} {verb} now past due, the oldest by {days_text}. "
                f"Please arrange payment this week.")
        closing = "Could you confirm the expected payment date?"
    else:
        main = (f"{phrase} {verb} overdue, the oldest by {days_text}. Please "
                f"arrange payment within 2 working days and reply to confirm.")
        closing = ("New work may need to pause until the balance is cleared. "
                   "If there is a problem, please let us know so we can "
                   "resolve it together.")
    return main, closing


def _assemble(greeting, main, closing):
    body = f"{greeting}\n\n{main}"
    if closing:
        body += f"\n\n{closing}"
    return f"{body}\n\n{SIGN_OFF}"


def _render(channel, subject, greeting, main, closing):
    if channel == "Email":
        return f"Subject: {subject}\n\n{_assemble(greeting, main, closing)}"
    text = _assemble(greeting, main, closing)
    if channel == "SMS" and len(text) > SMS_MAX_CHARS and closing:
        text = _assemble(greeting, main, None)
    return text


def draft_reminder(client_name, invoice_id, amount, due_date, days_overdue,
                   channel='WhatsApp', tone=None, contact_name=None) -> str:
    """Return one reminder message for a single invoice.

    channel: "WhatsApp", "SMS" or "Email" (case-insensitive).
    tone: None (auto from days_overdue), "polite", "firm" or "urgent".
    Pre-due reminders (days_overdue <= 0) always use the polite pre-due text.
    """
    channel = _normalise_channel(channel)
    tone = _normalise_tone(tone)
    days = int(days_overdue)
    tone = _effective_tone(days, tone)

    amount_text = format_inr(amount)
    due_text = _format_due(due_date)
    main, closing = _single_parts(tone, days, invoice_id, amount_text, due_text)
    subject = f"Payment reminder - Invoice #{invoice_id}"
    return _render(channel, subject, _greeting(client_name, contact_name),
                   main, closing)


def draft_batch_reminders(receivables_df, channel='WhatsApp', tone=None,
                          min_days_overdue=1) -> pd.DataFrame:
    """Return one reminder per overdue invoice, worst first.

    Columns: invoice_id, client_name, amount_inr, days_overdue, tone,
    channel, message. Empty DataFrame with those columns if none qualify.
    """
    channel = _normalise_channel(channel)
    tone = _normalise_tone(tone)

    overdue = receivables_df.loc[receivables_df["days_overdue"] >= min_days_overdue]
    overdue = overdue.sort_values(["days_overdue", "amount_inr"],
                                  ascending=[False, False])

    rows = []
    for _, row in overdue.iterrows():
        days = int(row["days_overdue"])
        rows.append({
            "invoice_id": row["invoice_id"],
            "client_name": row["client_name"],
            "amount_inr": row["amount_inr"],
            "days_overdue": days,
            "tone": _effective_tone(days, tone),
            "channel": channel,
            "message": draft_reminder(
                row["client_name"], row["invoice_id"], row["amount_inr"],
                row["due_date"], days, channel=channel, tone=tone,
            ),
        })
    return pd.DataFrame(rows, columns=BATCH_COLUMNS)


def draft_bad_day_batch(receivables_df, channel='WhatsApp',
                        min_days_overdue=1) -> dict:
    """Return a summary plus one combined reminder per client.

    Keys: "summary" (str), "clients" (int), "total_overdue" (int),
    "messages" (DataFrame with CLIENT_COLUMNS, oldest overdue first).
    """
    channel = _normalise_channel(channel)

    overdue = receivables_df.loc[receivables_df["days_overdue"] >= min_days_overdue]
    if overdue.empty:
        return {
            "summary": "No overdue invoices.",
            "clients": 0,
            "total_overdue": 0,
            "messages": pd.DataFrame(columns=CLIENT_COLUMNS),
        }

    total = int(overdue["amount_inr"].sum())
    oldest = int(overdue["days_overdue"].max())
    client_count = int(overdue["client_name"].nunique())
    verb = "has" if client_count == 1 else "have"
    invoice_phrase = ("an overdue invoice" if len(overdue) == 1
                      else "overdue invoices")
    summary = (f"{_plural(client_count, 'client')} {verb} {invoice_phrase} "
               f"totalling {format_inr(total)} "
               f"(oldest: {_plural(oldest, 'day')} overdue)")

    rows = []
    for client_name, group in overdue.groupby("client_name", sort=False):
        invoice_ids = [str(i) for i in group["invoice_id"]]
        amount = int(group["amount_inr"].sum())
        days = int(group["days_overdue"].max())
        tone = _effective_tone(days, None)
        main, closing = _group_parts(
            tone, days, invoice_ids, format_inr(amount),
            _format_due(group["due_date"].min()),
        )
        subject = f"Payment reminder - Invoices {', '.join(invoice_ids)}"
        rows.append({
            "client_name": client_name,
            "invoice_ids": ", ".join(invoice_ids),
            "amount_inr": amount,
            "oldest_days_overdue": days,
            "tone": tone,
            "channel": channel,
            "message": _render(channel, subject, _greeting(client_name),
                               main, closing),
        })

    messages = (pd.DataFrame(rows, columns=CLIENT_COLUMNS)
                .sort_values("oldest_days_overdue", ascending=False,
                             kind="stable")
                .reset_index(drop=True))
    return {
        "summary": summary,
        "clients": client_count,
        "total_overdue": total,
        "messages": messages,
    }
