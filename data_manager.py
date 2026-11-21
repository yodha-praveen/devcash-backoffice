"""Data access layer for the invoicing / bills app.

Loads and saves clients.csv, invoices.csv and bills.csv from the ``data``
folder next to this file, handles date parsing, adds and marks invoices,
builds the receivables summary and formats amounts in Indian rupee style.
"""

import os
from datetime import date

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CLIENT_COLS = [
    "client_id",
    "client_name",
    "contact_name",
    "contact_email",
    "client_type",
    "avg_days_late",
]
INVOICE_COLS = [
    "invoice_id",
    "client_id",
    "description",
    "amount_inr",
    "issue_date",
    "due_date",
    "status",
    "paid_date",
]
BILL_COLS = [
    "bill_id",
    "category",
    "vendor",
    "amount_inr",
    "frequency",
    "due_date",
    "auto_pay",
]

RECEIVABLE_COLS = [
    "invoice_id",
    "client_id",
    "client_name",
    "amount_inr",
    "issue_date",
    "due_date",
    "status",
    "days_overdue",
]

CLIENT_TEXT_COLS = ["client_id", "client_name", "contact_name", "contact_email", "client_type"]
INVOICE_TEXT_COLS = ["invoice_id", "client_id", "description", "status"]
BILL_TEXT_COLS = ["bill_id", "category", "vendor", "frequency", "auto_pay"]

DATE_FMT = "%Y-%m-%d"


def _path(path, filename):
    return path if path is not None else os.path.join(DATA_DIR, filename)


def _read(path, text_cols, date_cols, numeric_cols):
    df = pd.read_csv(path)
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _write(df, path, columns, date_cols):
    out = df.reindex(columns=columns).copy()
    for col in date_cols:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime(DATE_FMT).fillna("")
    out = out.fillna("")
    out.to_csv(path, index=False)


def load_clients(path=None):
    """Load clients.csv into a DataFrame."""
    return _read(_path(path, "clients.csv"), CLIENT_TEXT_COLS, [], ["avg_days_late"])


def save_clients(df, path=None):
    """Write clients DataFrame to clients.csv (no index column)."""
    _write(df, _path(path, "clients.csv"), CLIENT_COLS, [])


def load_invoices(path=None):
    """Load invoices.csv with parsed dates and numeric amounts."""
    return _read(
        _path(path, "invoices.csv"),
        INVOICE_TEXT_COLS,
        ["issue_date", "due_date", "paid_date"],
        ["amount_inr"],
    )


def save_invoices(df, path=None):
    """Write invoices DataFrame to invoices.csv (dates as YYYY-MM-DD)."""
    _write(df, _path(path, "invoices.csv"), INVOICE_COLS, ["issue_date", "due_date", "paid_date"])


def load_bills(path=None):
    """Load bills.csv with parsed due dates and numeric amounts."""
    return _read(
        _path(path, "bills.csv"),
        BILL_TEXT_COLS,
        ["due_date"],
        ["amount_inr"],
    )


def save_bills(df, path=None):
    """Write bills DataFrame to bills.csv (dates as YYYY-MM-DD)."""
    _write(df, _path(path, "bills.csv"), BILL_COLS, ["due_date"])


def add_invoice(client_id, description, amount_inr, issue_date, term_days=30, path=None):
    """Create a new Pending invoice, save it to invoices.csv and return its id.

    ``path`` points at invoices.csv; clients.csv is read from the same folder.
    """
    invoices_path = _path(path, "invoices.csv")
    clients = load_clients(os.path.join(os.path.dirname(invoices_path), "clients.csv"))
    if str(client_id).strip() not in set(clients["client_id"]):
        raise ValueError(f"Unknown client_id: {client_id}")

    amount = float(amount_inr)
    if not amount > 0:
        raise ValueError("amount_inr must be greater than 0")

    term = int(term_days)
    if term <= 0:
        raise ValueError("term_days must be greater than 0")

    issue = pd.Timestamp(issue_date)
    if pd.isna(issue):
        raise ValueError(f"Invalid issue_date: {issue_date}")
    issue = issue.normalize()
    due = issue + pd.Timedelta(days=term)

    invoices = load_invoices(invoices_path)
    numbers = pd.to_numeric(
        invoices["invoice_id"].str.replace(r"\D", "", regex=True), errors="coerce"
    )
    next_no = int(numbers.max()) + 1 if numbers.notna().any() else 1
    new_id = f"INV{next_no:03d}"

    new_row = pd.DataFrame(
        [
            {
                "invoice_id": new_id,
                "client_id": str(client_id).strip(),
                "description": str(description).strip(),
                "amount_inr": int(round(amount)),
                "issue_date": issue,
                "due_date": due,
                "status": "Pending",
                "paid_date": pd.NaT,
            }
        ],
        columns=INVOICE_COLS,
    )
    invoices = pd.concat([invoices, new_row], ignore_index=True)
    save_invoices(invoices, invoices_path)
    return new_id


def mark_invoice_paid(invoice_id, paid_date, path=None):
    """Mark an invoice Paid on ``paid_date``, save, and return the updated row as a dict."""
    invoices_path = _path(path, "invoices.csv")
    invoices = load_invoices(invoices_path)

    mask = invoices["invoice_id"] == str(invoice_id).strip()
    if not mask.any():
        raise KeyError(f"Invoice not found: {invoice_id}")
    idx = invoices.index[mask][0]

    paid = pd.Timestamp(paid_date)
    if pd.isna(paid):
        raise ValueError(f"Invalid paid_date: {paid_date}")

    invoices.loc[idx, "status"] = "Paid"
    invoices.loc[idx, "paid_date"] = paid.normalize()
    save_invoices(invoices, invoices_path)
    return invoices.loc[idx].to_dict()


def get_receivables_summary(invoices_df, clients_df, as_of=None):
    """Return unpaid invoices with client names and days overdue, sorted by overdue then amount."""
    as_of_ts = pd.Timestamp(date.today()) if as_of is None else pd.Timestamp(as_of)
    as_of_ts = as_of_ts.normalize()

    unpaid = invoices_df[invoices_df["status"] != "Paid"].copy()
    if unpaid.empty:
        return pd.DataFrame(columns=RECEIVABLE_COLS)

    names = clients_df.set_index("client_id")["client_name"]
    unpaid.loc[:, "client_name"] = unpaid["client_id"].map(names)
    unpaid.loc[:, "issue_date"] = pd.to_datetime(unpaid["issue_date"], errors="coerce")
    unpaid.loc[:, "due_date"] = pd.to_datetime(unpaid["due_date"], errors="coerce")
    unpaid.loc[:, "amount_inr"] = pd.to_numeric(unpaid["amount_inr"], errors="coerce")

    days = (as_of_ts - unpaid["due_date"]).dt.days
    unpaid.loc[:, "days_overdue"] = days.clip(lower=0).fillna(0).astype(int)

    result = unpaid.loc[:, RECEIVABLE_COLS]
    result = result.sort_values(["days_overdue", "amount_inr"], ascending=[False, False])
    return result.reset_index(drop=True)


def format_inr(amount):
    """Format a number as Indian rupees with lakh/crore grouping, e.g. 1234567 -> ₹12,34,567."""
    value = int(round(float(amount)))
    sign = "-" if value < 0 else ""
    digits = str(abs(value))

    last3 = digits[-3:]
    rest = digits[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)

    grouped = ",".join(parts + [last3]) if parts else last3
    return f"{sign}\u20b9{grouped}"
