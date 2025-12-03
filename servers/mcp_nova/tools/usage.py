import pandas as pd

# Load usage data once at module import
USAGE_DATA = pd.read_csv("data/usage.csv")


def get_usage_report(account_id: str, month: str | None = None) -> dict:

    # Filter by account
    account_rows = USAGE_DATA.query("account_id == @account_id")

    # Filter by month if provided
    if month is not None:
        account_rows = account_rows.query("month == @month")

    # No matching usage record
    if account_rows.empty:
        return {"error": f"No usage records found for account '{account_id}'"}

    # Use the first matching row
    row = account_rows.iloc[0].to_dict()

    return {
        "account_id": row.get("account_id"),
        "month": row.get("month"),
        "api_calls": row.get("api_calls"),
        "plan_limit": row.get("plan_limit"),
        "overage": row.get("overage"),
    }
