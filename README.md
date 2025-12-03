# NovaCRM Agentic AI Capstone - 

A robust agentic AI solution designed to respond to customer queries, recap account activity, and automate basic operational workflows, powered by LangChain, LangGraph, and an MCP tool server.

## 1. Setup Environment

python -m venv .venv
venv\Scripts\activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

## 2. Configure API Key

Create a `.env` file in the project root:

OPENAI_API_KEY=sk-your-api-key-here

## 3. Build the Knowledge Base Index

python scripts/build_index.py

## 4. MCP Server

uvicorn servers.mcp_nova.server:app --reload --port 8001

## 5. CLI

python app/cli.py

