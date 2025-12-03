# NovaCRM Capstone Dataset

This archive contains all data files and reference documents required for the
**Agentic AI Capstone Project** covering Prompt Engineering, LangChain, LangGraph, and MCP.

## Included Files

### 1. CSV Datasets (for MCP tools)
- **accounts.csv** — Master account details with plan, tier, billing cycle, and renewal date.
- **invoices.csv** — Invoice history (Paid, Pending, Overdue) for past months.
- **tickets.csv** — Customer support tickets with priority and status.
- **usage.csv** — Monthly usage metrics (API calls, emails sent, storage used).

These support your MCP tools: `account_lookup`, `invoice_status`, `ticket_summary`, and `usage_report`.

### 2. Knowledge Base (for RAG)
Located under `kb/` folder:
- overview.md
- pricing_plans.md
- billing_module.md
- campaigns_module.md
- support_module.md
- api_guide.md
- security_faq.md

Use these to build your FAISS/Chroma vector store for FAQ and information retrieval tasks.

### 3. Suggested Folder Placement
```
capstone/
  data/
    accounts.csv
    invoices.csv
    tickets.csv
    usage.csv
    kb/
      overview.md
      pricing_plans.md
      billing_module.md
      campaigns_module.md
      support_module.md
      api_guide.md
      security_faq.md
```

### Author
Prepared for **Agentic AI Capstone Evaluation** (Prompt Engineering, LangChain, LangGraph, MCP).