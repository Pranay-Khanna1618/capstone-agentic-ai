Intent Classification Goal
Categorize each user query into one of three intents: FAQ, DataLookup, or Escalation.

Illustrative Cases

“Explain about the billing module?” → FAQ

“Whats the billing cycle for account A038 renew?” → DataLookup

“Payments missing for account A016s — urgent” → Escalation

Decision Principles

If the question depends on details about a specific account or record, assign DataLookup.

If the user signals urgency, system failure, or operational risk, select Escalation.