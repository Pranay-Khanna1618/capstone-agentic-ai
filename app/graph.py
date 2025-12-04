from __future__ import annotations
import json
import re

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field


# ------------------------------------------------------------------------------
# Environment & paths
# ------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


# ------------------------------------------------------------------------------
# State model
# ------------------------------------------------------------------------------

class AgentState(BaseModel):

    history: List[str] = Field(default_factory=list)
    intent: Optional[Literal["FAQ", "DataLookup", "Escalation"]] = None
    query: str
    answer: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None

    retrieved_context: Optional[str] = None
    tool_results: Optional[Union[str, Dict[str, Any]]] = None

    temperature: float = 0.0
    account: Optional[str] = None

    # Allow extra fields if needed by LangGraph / other tooling
    model_config = ConfigDict(extra="allow")


# ------------------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------------------

def load_prompt_file(name: str) -> str:
    prompt_path = APP_DIR / "prompts" / name
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read()


_vector_store: Optional[FAISS] = None


def get_vector_store() -> FAISS:
    global _vector_store
    if _vector_store is None:
        index_dir = PROJECT_ROOT / "index"
        embeddings = OpenAIEmbeddings()
        _vector_store = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vector_store


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


# ------------------------------------------------------------------------------
# Intent routing
# ------------------------------------------------------------------------------

def route_intent_node(state: AgentState) -> AgentState:
    try:
        system_prompt = load_prompt_file("system.md")
        router_prompt = load_prompt_file("router.md")

        prompt = f"""{system_prompt}

{router_prompt}

User Query: {state.query}

Classify this query as exactly one of: FAQ, DataLookup, Escalation.
Respond with ONLY the intent name."""
        llm = get_llm(temperature=0.0)
        result = llm.invoke(prompt)
        raw = (getattr(result, "content", None) or str(result)).strip().upper()

        if "FAQ" in raw:
            state.intent = "FAQ"
        elif "DATALOOKUP" in raw or "DATA_LOOKUP" in raw:
            state.intent = "DataLookup"
        elif "ESCALATION" in raw or "ESCALATE" in raw:
            state.intent = "Escalation"
        else:
            # Fallback heuristic
            q = state.query.lower()
            if any(k in q for k in ["invoice", "plan", "account"]):
                state.intent = "DataLookup"
            elif any(k in q for k in ["urgent", "not working", "broken"]):
                state.intent = "Escalation"
            else:
                state.intent = "FAQ"

    except Exception as exc:
        state.errors.append(f"Router error: {exc}")
        q = state.query.lower()
        if any(k in q for k in ["invoice", "plan"]):
            state.intent = "DataLookup"
        elif any(k in q for k in ["urgent", "not working"]):
            state.intent = "Escalation"
        else:
            state.intent = "FAQ"

    return state


# ------------------------------------------------------------------------------
# Retrieval (RAG)
# ------------------------------------------------------------------------------

def retrieve_knowledge_node(state: AgentState) -> AgentState:

    try:
        store = get_vector_store()
        docs = store.similarity_search(state.query, k=3)

        chunks: List[str] = []
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            idx = doc.metadata.get("chunk", "?")
            state.evidence.append(f"{src}:chunk_{idx}")
            chunks.append(doc.page_content)

        state.retrieved_context = "\n\n".join(chunks) if chunks else ""

    except Exception as exc:
        state.errors.append(f"Retrieval error: {exc}")
        state.evidence.clear()
        state.retrieved_context = None

    return state


# ------------------------------------------------------------------------------
# Query parsing helpers
# ------------------------------------------------------------------------------

def extract_account_id(query: str, default: Optional[str] = None) -> Optional[str]:

    if default:
        return str(default).upper()

    patterns = [
        r"account\s+([A-Z]?\d+)",
        r"account_id\s+([A-Z]?\d+)",
        r"for\s+account\s+([A-Z]?\d+)",
        r"acc\s+([A-Z]?\d+)",
        r"\b([A-Z]\d{3})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def extract_invoice_id(query: str) -> Optional[str]:

    patterns = [
        r"invoice\s+(\d+)",
        r"invoice_id\s+(\d+)",
        r"inv\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def choose_tool_name(query: str) -> str:

    text = query.lower()

    if "invoice" in text:
        return "invoice"
    if "ticket" in text:
        return "ticket"
    if "usage" in text or "report" in text:
        return "usage"
    if "account" in text or "plan" in text:
        return "account"

    # Default data lookup target
    return "invoice"


# ------------------------------------------------------------------------------
# Tool calls (MCP server-backed tools)
# ------------------------------------------------------------------------------

def run_tools_node(state: AgentState) -> AgentState:

    try:
        from servers.mcp_nova.tools import account, invoice, ticket, usage

        system_prompt = load_prompt_file("system.md")
        tool_prompt = load_prompt_file("tool_check.md")

        account_id = extract_account_id(state.query, state.account)
        if not account_id:
            msg = (
                "Could not determine account_id from your query. "
                "Please either use the --account flag or mention it explicitly "
                "(e.g., 'invoice status for account A001')."
            )
            state.errors.append(msg)
            state.evidence.append("error:missing_account_id")
            return state

        tool_name = choose_tool_name(state.query)
        invoice_id = extract_invoice_id(state.query)

        justification_prompt = f"""{system_prompt}

{tool_prompt}

Query: {state.query}
Account ID: {account_id}
Tool to call: {tool_name}

Provide a one-line rationale for calling this tool."""
        try:
            llm = get_llm(temperature=0.0)
            resp = llm.invoke(justification_prompt)
            justification_text = (getattr(resp, "content", None) or str(resp)).strip()
            state.history.append(f"[Tool Justification] {justification_text}")
        except Exception:
            state.history.append(
                f"[Tool Justification] Using {tool_name} for account {account_id}"
            )

        tool_output: Any = None
        if tool_name == "invoice":
            tool_output = invoice.get_invoice_status(
                account_id=account_id,
                invoice_id=invoice_id,
                period=None,
            )
            if invoice_id:
                state.evidence.append(
                    f"invoice_status:account_{account_id}:invoice_{invoice_id}"
                )
            else:
                state.evidence.append(f"invoice_status:account_{account_id}")
        elif tool_name == "account":
            tool_output = account.lookup_account(account_id=account_id)
            state.evidence.append(f"account_lookup:account_{account_id}")
        elif tool_name == "ticket":
            tool_output = ticket.summarize_tickets(account_id=account_id)
            state.evidence.append(f"ticket_summary:account_{account_id}")
        elif tool_name == "usage":
            tool_output = usage.get_usage_report(account_id=account_id)
            state.evidence.append(f"usage_report:account_{account_id}")

        # Format tool output for later synthesis
        if tool_output is None:
            state.errors.append("Tool call returned no results.")
            state.tool_results = "No data found."
            return state

        if isinstance(tool_output, dict):
            if "error" in tool_output:
                err = tool_output.get("error", "Tool returned an error")
                state.errors.append(err)
                state.tool_results = f"Error: {err}"
            else:
                lines = [
                    f"{k}: {v}"
                    for k, v in tool_output.items()
                    if k != "source"
                ]
                state.tool_results = "\n".join(lines)
                if "source" in tool_output:
                    state.evidence.append(tool_output["source"])
        else:
            state.tool_results = str(tool_output)

    except Exception as exc:
        state.errors.append(f"Tool call error: {exc}")
        state.tool_results = f"Error: {exc}"

    return state


# ------------------------------------------------------------------------------
# Answer synthesis
# ------------------------------------------------------------------------------

def synthesize_answer_node(state: AgentState) -> AgentState:

    try:
        if not state.evidence:
            state.answer = (
                "I don’t have enough reliable information to answer this confidently, "
                "so I’m escalating it to the support team."
            )
            state.confidence = 0.3
            return state

        rag_context = state.retrieved_context or ""
        tool_context = state.tool_results or ""

        if rag_context and tool_context:
            merged_context = (
                f"Knowledge Base Information:\n{rag_context}\n\n"
                f"Tool/Data Results:\n{tool_context}"
            )
        elif tool_context:
            merged_context = f"Tool/Data Results:\n{tool_context}"
        else:
            merged_context = f"Knowledge Base Information:\n{rag_context}"

        system_prompt = load_prompt_file("system.md")
        synth_prompt = load_prompt_file("rag_synth.md")

        evidence_lines = "\n".join(f"- {e}" for e in state.evidence)

        full_prompt = f"""{system_prompt}

{synth_prompt}

User Question: {state.query}

Retrieved Context:
{merged_context}

Evidence Sources (to reference in your response):
{evidence_lines}

Provide a concise answer based on the retrieved context above, and include an Evidence section."""
        llm = get_llm(temperature=state.temperature)
        resp = llm.invoke(full_prompt)
        answer_text = getattr(resp, "content", None) or str(resp)

        state.answer = answer_text
        state.confidence = 0.9 if merged_context else 0.3
        state.history.append(f"Q: {state.query}\nA: {answer_text[:200]}...")

    except Exception as exc:
        state.errors.append(f"Synthesis error: {exc}")
        state.answer = (
            "I ran into an issue while generating your answer. "
            "Please try again, or contact support if it persists."
        )
        state.confidence = 0.2

    return state

# ------------------------------------------------------------------------------
# Escalation
# ------------------------------------------------------------------------------

def escalate_fallback_node(state: AgentState) -> AgentState:

    state.answer = (
        "This request has been escalated to the support team due to "
        "insufficient data or urgency."
    )
    return state


# ------------------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------------------

def create_agent_graph() -> RunnableLambda:
    builder = StateGraph(AgentState)

    builder.add_node("router", route_intent_node)
    builder.add_node("retrieve", retrieve_knowledge_node)
    builder.add_node("tool_call", run_tools_node)
    builder.add_node("synthesize", synthesize_answer_node)
    builder.add_node("escalate", escalate_fallback_node)

    builder.set_entry_point("router")

    # Route based on intent
    builder.add_conditional_edges(
        "router",
        lambda s: s.intent,
        {"FAQ": "retrieve", "DataLookup": "tool_call", "Escalation": "escalate"},
    )

    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("tool_call", "synthesize")
    builder.add_edge("escalate", END)

    db_path = PROJECT_ROOT / "state.db"
    import sqlite3
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)


graph = create_agent_graph()


# ------------------------------------------------------------------------------
# Public entry point (used by CLI)
# ------------------------------------------------------------------------------

def run_agent(query: str, account: Optional[str] = None, temperature: float = 0.0) -> AgentState:

    initial = AgentState(query=query, account=account, temperature=temperature)
    result: AgentState = graph.invoke(
        initial,
        config={"configurable": {"thread_id": "cli"}},
    )
    return result
