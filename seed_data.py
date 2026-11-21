"""Synthetic seed data generator for the One-Person Back Office prototype.

All data written by this script is SYNTHETIC and fictional. Client names,
contact names, emails (example.com / example.in), vendors and amounts are
made up. No real people, real companies, phone numbers, bank details or
PAN/GST numbers are used.

Usage:
    python seed_data.py

Writes clients.csv, invoices.csv and bills.csv into the data/ folder that
sits next to this script (created if missing), then prints the paths and
row counts. Amounts are plain numbers in INR (the rupee sign is not stored
in the CSV values).
"""

import csv
import datetime
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLIENT_FIELDS = [
    "client_id", "client_name", "contact_name", "contact_email",
    "client_type", "avg_days_late",
]
INVOICE_FIELDS = [
    "invoice_id", "client_id", "description", "amount_inr", "issue_date",
    "due_date", "status", "paid_date",
]
BILL_FIELDS = [
    "bill_id", "category", "vendor", "amount_inr", "frequency", "due_date",
    "auto_pay",
]

# Fictional clients. avg_days_late: a few reliable payers (0-7) and a few
# chronic late payers (18-35).
CLIENTS = [
    {"client_id": "CL001", "client_name": "Northwind Labs", "contact_name": "Anika Rao",
     "contact_email": "anika.rao@example.com", "client_type": "Startup", "avg_days_late": 2},
    {"client_id": "CL002", "client_name": "Bluepeak Agency", "contact_name": "Rohan Iyer",
     "contact_email": "rohan.iyer@example.com", "client_type": "Agency", "avg_days_late": 5},
    {"client_id": "CL003", "client_name": "Maple Street Studio", "contact_name": "Tara Menon",
     "contact_email": "tara.menon@example.in", "client_type": "Small Business", "avg_days_late": 12},
    {"client_id": "CL004", "client_name": "Quantum Fox Analytics", "contact_name": "Meera Shah",
     "contact_email": "meera.shah@example.in", "client_type": "Startup", "avg_days_late": 0},
    {"client_id": "CL005", "client_name": "Redwood Media House", "contact_name": "Kabir Nair",
     "contact_email": "kabir.nair@example.com", "client_type": "Agency", "avg_days_late": 28},
    {"client_id": "CL006", "client_name": "Lotus Retail Demo", "contact_name": "Sana Qureshi",
     "contact_email": "sana.qureshi@example.in", "client_type": "Small Business", "avg_days_late": 35},
    {"client_id": "CL007", "client_name": "Orbit Health Tech", "contact_name": "Vikram Joshi",
     "contact_email": "vikram.joshi@example.com", "client_type": "Startup", "avg_days_late": 18},
    {"client_id": "CL008", "client_name": "Greenfield Consultants", "contact_name": "Neha Kapoor",
     "contact_email": "neha.kapoor@example.in", "client_type": "Agency", "avg_days_late": 7},
]

DESCRIPTIONS = [
    "Website redesign - phase 1",
    "Monthly retainer - content",
    "API integration sprint",
    "Dashboard build",
    "Mobile app bug-fix pack",
    "Data pipeline setup",
    "Support retainer",
    "Landing page design",
    "Analytics consulting",
]

# (category, vendor, amount_inr, frequency, months_between, day_of_month, auto_pay)
# day_of_month None means the anchor is 45 days from today.
RECURRING_BILLS = [
    ("Rent (home office)", "Example Property Management", 18000, "Monthly", 1, 5, "No"),
    ("Internet", "Demo Fibre Net", 1199, "Monthly", 1, 10, "Yes"),
    ("Cloud hosting", "Example Cloud Co.", 3200, "Monthly", 1, 3, "Yes"),
    ("AI/LLM API subscriptions", "Demo AI Platform Inc.", 4500, "Monthly", 1, 7, "Yes"),
    ("Coworking desk", "Demo Coworking Hub", 8500, "Monthly", 1, 1, "No"),
    ("Tax reserve (transfer)", "Own tax reserve account - transfer", 40000, "Quarterly", 3, 15, "No"),
    ("Professional liability insurance", "Example Insurance Agency", 12000, "Yearly", 12, None, "No"),
]

BILL_WINDOW_DAYS = 90


def add_months(date, months):
    """Return date shifted by whole months, clamping the day if needed."""
    total = date.month - 1 + months
    year = date.year + total // 12
    month = total % 12 + 1
    day = date.day
    while True:
        try:
            return datetime.date(year, month, day)
        except ValueError:
            day -= 1


def first_due_on_or_after(today, day_of_month):
    """First date on day_of_month that is today or later."""
    candidate = datetime.date(today.year, today.month, day_of_month)
    if candidate < today:
        candidate = add_months(candidate, 1)
    return candidate


def build_invoices(today, client_ids, avg_late):
    """Build 25 synthetic invoices: 13 Paid, 6 Overdue, 6 Pending."""
    rows = []

    def make(client_id, issue, term, status, paid_date):
        return {
            "client_id": client_id,
            "description": random.choice(DESCRIPTIONS),
            "amount_inr": random.randrange(25000, 450001, 500),
            "issue_date": issue.isoformat(),
            "due_date": (issue + datetime.timedelta(days=term)).isoformat(),
            "status": status,
            "paid_date": paid_date.isoformat() if paid_date else "",
        }

    # 13 Paid: issued 60-120 days ago, paid a few days after due date,
    # with lateness centred on the client's avg_days_late.
    for _ in range(13):
        cid = random.choice(client_ids)
        issue = today - datetime.timedelta(days=random.randint(60, 120))
        term = random.choice([15, 30])
        due = issue + datetime.timedelta(days=term)
        lateness = max(0, avg_late[cid] + random.randint(-2, 2))
        paid = min(due + datetime.timedelta(days=lateness), today)
        rows.append(make(cid, issue, term, "Paid", paid))

    # 6 Overdue: unpaid, due date already passed.
    for _ in range(6):
        cid = random.choice(client_ids)
        issue = today - datetime.timedelta(days=random.randint(35, 100))
        term = random.choice([15, 30])
        rows.append(make(cid, issue, term, "Overdue", None))

    # 6 Pending: unpaid, due today or later (within the next 30 days,
    # since terms are 15 or 30 days and issue dates cannot be in the future).
    for _ in range(6):
        cid = random.choice(client_ids)
        term = random.choice([15, 30])
        offset = random.randint(1, term)
        due = today + datetime.timedelta(days=offset)
        issue = due - datetime.timedelta(days=term)
        rows.append(make(cid, issue, term, "Pending", None))

    rows.sort(key=lambda r: r["issue_date"])
    for n, row in enumerate(rows, start=1):
        row["invoice_id"] = f"INV{n:03d}"
    return rows


def build_bills(today):
    """Expand recurring fixed expenses into one row per occurrence (next 90 days)."""
    end = today + datetime.timedelta(days=BILL_WINDOW_DAYS)
    occurrences = []
    for category, vendor, amount, frequency, step, day, auto_pay in RECURRING_BILLS:
        if day is None:
            anchor = today + datetime.timedelta(days=45)
        else:
            anchor = first_due_on_or_after(today, day)
        k = 0
        while True:
            due = add_months(anchor, k * step)
            if due > end:
                break
            if due >= today:
                occurrences.append({
                    "category": category,
                    "vendor": vendor,
                    "amount_inr": amount,
                    "frequency": frequency,
                    "due_date": due.isoformat(),
                    "auto_pay": auto_pay,
                })
            k += 1

    occurrences.sort(key=lambda r: (r["due_date"], r["category"]))
    for n, row in enumerate(occurrences, start=1):
        row["bill_id"] = f"B{n:03d}"
    return occurrences


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(42)
    today = datetime.date.today()  # as-of date

    os.makedirs(DATA_DIR, exist_ok=True)

    client_ids = [c["client_id"] for c in CLIENTS]
    avg_late = {c["client_id"]: c["avg_days_late"] for c in CLIENTS}

    invoices = build_invoices(today, client_ids, avg_late)
    bills = build_bills(today)

    outputs = [
        ("clients.csv", CLIENT_FIELDS, CLIENTS),
        ("invoices.csv", INVOICE_FIELDS, invoices),
        ("bills.csv", BILL_FIELDS, bills),
    ]

    for filename, fieldnames, rows in outputs:
        path = os.path.join(DATA_DIR, filename)
        write_csv(path, fieldnames, rows)
        print(f"Wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
