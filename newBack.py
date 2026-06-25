
import asyncio
import os
from typing import Annotated, TypedDict, List, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

# ---------------- FIRESTORE ----------------
from services.firestore import (
    get_today_visits,
    get_pending_followups,
    get_weekly_performance_data
)

# ---------------- LLM ----------------
llm = ChatGroq(
    temperature=0.4,
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ---------------- HASH IDS ----------------
SUMMARY_ID = "8f3c1a9b2e7d4c6f"
FOLLOWUPS_ID = "b72e4d1f9a3c8e65"
PERFORMANCE_ID = "c91d5e7a2b4f8c30"

# ---------------- STATE ----------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "chat history"]
    user_id: str
    next_node: str

# ---------------- UTILITY (IMPORTANT) ----------------
def safe_to_string(content: Any) -> str:
    """Convert any LLM output into safe string for JSON response"""
    if content is None:
        return "No response generated"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return " ".join([str(c) for c in content])

    return str(content)

# ---------------- SUPERVISOR ----------------
async def supervisor_node(state: AgentState):
    q = state["messages"][-1].content.strip()

    if q == SUMMARY_ID:
        node = "summary_agent"
    elif q == FOLLOWUPS_ID:
        node = "followup_agent"
    elif q == PERFORMANCE_ID:
        node = "performance_agent"
    else:
        node = "sales_expert_agent"

    return {"next_node": node}

# ---------------- SUMMARY ----------------
async def summary_node(state: AgentState):
    data = get_today_visits(state["user_id"])

    if not data:
        return {"messages": [SystemMessage(content="📊 No visits today")]}

    context = "\n".join([
        f"{d.get('customerName')}: {d.get('summary')}"
        for d in data
    ])

    prompt = f"""
You are a sales manager.

Analyze:
{context}

Give:
1. Performance summary
2. Insights
3. Next steps
"""

    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [res]}

# ---------------- FOLLOWUPS ----------------
async def followup_node(state: AgentState):
    data = get_pending_followups(state["user_id"])

    if not data:
        return {"messages": [SystemMessage(content="📞 No follow-ups")]}

    context = "\n".join([
        f"{d.get('customerName')}: {d.get('nextStep')}"
        for d in data if d.get("nextStep")
    ])

    prompt = f"""
You are a sales execution expert.

Analyze:
{context}

Give priorities and actions.
"""

    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [res]}

# ---------------- PERFORMANCE ----------------
async def performance_node(state: AgentState):
    data = get_weekly_performance_data(state["user_id"])

    if not data:
        return {"messages": [SystemMessage(content="📉 No data")]}

    context = "\n".join([
        f"{d.get('customerName')} | {d.get('painPoints')} | {d.get('probability')}"
        for d in data
    ])

    prompt = f"""
You are a strategist.

Analyze:
{context}

Give insights and recommendations.
"""

    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [res]}

# ---------------- SALES EXPERT ----------------
async def sales_expert_node(state: AgentState):
    user_input = state["messages"][-1].content
    user_id = state["user_id"]

    today = get_today_visits(user_id)
    followups = get_pending_followups(user_id)
    performance = get_weekly_performance_data(user_id)

    context = f"""
Visits: {today[:3]}
Followups: {followups[:3]}
Issues: {performance[:3]}
"""

    prompt = f"""
You are a senior sales expert.

Data:
{context}

Question:
{user_input}

Give actionable advice.
"""

    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [res]}

# ---------------- GRAPH ----------------
builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("summary_agent", summary_node)
builder.add_node("followup_agent", followup_node)
builder.add_node("performance_agent", performance_node)
builder.add_node("sales_expert_agent", sales_expert_node)

builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda x: x["next_node"],
    {
        "summary_agent": "summary_agent",
        "followup_agent": "followup_agent",
        "performance_agent": "performance_agent",
        "sales_expert_agent": "sales_expert_agent"
    }
)

builder.add_edge("summary_agent", END)
builder.add_edge("followup_agent", END)
builder.add_edge("performance_agent", END)
builder.add_edge("sales_expert_agent", END)

sales_brain = builder.compile()

# ---------------- FASTAPI ----------------
app = FastAPI(title="Agentic Sales AI")

# ✅ CORS (production-safe tweak later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SCHEMAS ----------------
class ChatRequest(BaseModel):
    message: str
    user_id: str

class ChatResponse(BaseModel):
    reply: str

# ---------------- ROUTES ----------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await sales_brain.ainvoke({
            "messages": [HumanMessage(content=req.message)],
            "user_id": req.user_id
        })

        raw = result["messages"][-1].content
        reply = safe_to_string(raw)

        print("✅ RESPONSE:", reply)

        return {"reply": reply}

    except Exception as e:
        print("❌ ERROR:", e)
        return {"reply": f"Error: {str(e)}"}

@app.get("/")
def root():
    return {"status": "🚀 Sales AI Running"}

# ---------------- CLI ----------------
async def run():
    print("\n🚀 AI Ready")

    user_id = "iPr3xaUi9DXQruUtefalaDVkAJH2"

    while True:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]:
            break

        res = await sales_brain.ainvoke({
            "messages": [HumanMessage(content=q)],
            "user_id": user_id
        })

        print("\nAI:", safe_to_string(res["messages"][-1].content))

if __name__ == "__main__":
    asyncio.run(run())
