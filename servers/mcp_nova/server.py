from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from dotenv import load_dotenv
from servers.mcp_nova.tools import account, invoice, ticket, usage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

app = FastAPI(title="MCP Nova Tool Server")


@app.get("/account_lookup")
def account_lookup(account_id: str, company: Optional[str] = None):

    return account.lookup_account(account_id=account_id, company=company)


@app.get("/invoice_status")
def invoice_status(account_id: str, invoice_id: Optional[str] = None, period: Optional[str] = None):

    return invoice.get_invoice_status(
        account_id=account_id,
        invoice_id=invoice_id,
        period=period,
    )


@app.get("/ticket_summary")
def ticket_summary(account_id: str, lookback_days: int = 90, limit: int = 3):
    
    return ticket.summarize_tickets(
        account_id=account_id,
        lookback_days=lookback_days,
        limit=limit,
    )


@app.get("/usage_report")
def usage_report(account_id: str, month: Optional[str] = None):

    return usage.get_usage_report(account_id=account_id, month=month)


@app.get("/kb_search")
def kb_search(query: str, k: int = 5):

    try:

        # Compute project root relative to this file
        project_root = Path(__file__).resolve().parents[3]
        index_path = project_root / "index"

        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        docs = vector_store.similarity_search(query, k=k)

        results = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk": doc.metadata.get("chunk", "?"),
            }
            for doc in docs
        ]

        return {
            "query": query,
            "results": results,
            "count": len(results),
        }

    except Exception as exc:
        return {
            "query": query,
            "error": str(exc),
        }
