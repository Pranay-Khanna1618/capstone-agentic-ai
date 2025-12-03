import pandas as pd

# Load ticket data once at import time
TICKET_DATA = pd.read_csv("data/tickets.csv")


def summarize_tickets(account_id: str, lookback_days: int = 90, limit: int = 3) -> dict:

    # Select only open tickets for the given account
    open_items = TICKET_DATA.query("account_id == @account_id and status == 'Open'")

    if open_items.empty:
        return {"error": f"No open tickets for account '{account_id}'"}

    # Work with a copy to avoid modifying the global DataFrame
    df = open_items.copy()
    df["opened_on"] = pd.to_datetime(df["opened_on"])

    # Filter based on the time window provided
    threshold = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
    recent_rows = df[df["opened_on"] >= threshold]

    # Format the top N items
    tickets = [
        {
            "ticket_id": row.ticket_id,
            "subject": row.subject,
            "priority": row.priority,
            "opened_on": row.opened_on.isoformat(),
        }
        for _, row in recent_rows.head(limit).iterrows()
    ]

    return {"tickets": tickets}
