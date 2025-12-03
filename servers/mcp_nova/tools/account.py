import pandas as pd

# Load account metadata into memory
ACCOUNT_DATA = pd.read_csv("data/accounts.csv")


def lookup_account(account_id: str, company: str | None = None) -> dict:

    # Filter rows by account ID first
    subset = ACCOUNT_DATA.query("account_id == @account_id")

    # Apply optional company filter (case-insensitive)
    if company:
        company_normalized = company.strip().lower()
        subset = subset[subset["company"].str.lower() == company_normalized]

    # No match found
    if subset.empty:
        return {"error": f"No account found for '{account_id}'"}

    record = subset.iloc[0].to_dict()

    return {
        "account_id": record.get("account_id"),
        "company": record.get("company"),
        "plan": record.get("plan_name"),
        "renewal_date": record.get("renewal_date"),
        "customer_success_manager": record.get("customer_success_manager"),
        "flags": record.get("flags"),
    }
