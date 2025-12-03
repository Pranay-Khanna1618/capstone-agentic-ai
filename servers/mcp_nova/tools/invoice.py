import pandas as pd

# Load invoice records into memory
INVOICE_DATA = pd.read_csv("data/invoices.csv")


def get_invoice_status(account_id: str, invoice_id: str | None = None, period: str | None = None) -> dict:

    # Start with invoices matching the account
    filtered = INVOICE_DATA.query("account_id == @account_id")

    # Filter by invoice_id if specified
    if invoice_id:
        filtered = filtered.query("invoice_id == @invoice_id")

    # Filter by billing period if provided
    if period:
        filtered = filtered.query("period == @period")

    # No matching invoice
    if filtered.empty:
        return {"error": f"No invoice found for account '{account_id}'"}

    record = filtered.iloc[0].to_dict()

    return {
        "invoice_id": record.get("invoice_id"),
        "account_id": record.get("account_id"),
        "period": record.get("period"),
        "amount": record.get("amount"),
        "status": record.get("status"),
        "due_date": record.get("due_date"),
        "paid_date": record.get("paid_date"),
    }
