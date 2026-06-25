import asyncio
import os
from typing import Annotated, TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

from services.firestore import (
    get_today_visits,
    get_pending_followups,
    get_weekly_performance_data
)

# ---------------- LLM ----------------
llm = ChatGroq(
    temperature=0.5,
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")  # unchanged
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

    print(f"DEBUG → {node}")
    return {"next_node": node}

# ---------------- SUMMARY ----------------
async def summary_node(state: AgentState):
    data = get_today_visits(state["user_id"])

    if not data:
        return {"messages": [SystemMessage(content="📊 No visits today")]}

    context = "Today's visit summaries:\n\n"

    for d in data:
        name = d.get("customerName") or "Unknown"
        summary = (d.get("summary") or "No summary").strip()

        context += f"{name}: {summary}\n"

    prompt = f"""
You are a senior sales manager.

Analyze these visit summaries and provide:
1. Overall performance
2. Key observations
3. Suggested next steps

Keep it concise and structured.

{context}
"""

    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [res]}

# ---------------- FOLLOWUPS ----------------
async def followup_node(state: AgentState):
    data = get_pending_followups(state["user_id"])

    if not data:
        return {"messages": [SystemMessage(content="📞 No pending follow-ups")]}

    context = "Pending follow-ups:\n\n"

    for d in data:
        name = d.get("customerName") or "Unknown"
        step = (d.get("nextStep") or "").strip()

        if step:
            context += f"{name}: {step}\n"

    prompt = f"""
You are a sales execution expert.

Based on these follow-ups:
1. Prioritize tasks
2. Suggest execution strategy
3. Highlight urgent actions

Be clear and actionable.

{context}
"""

    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [res]}

# ---------------- PERFORMANCE ----------------
async def performance_node(state: AgentState):
    data = get_weekly_performance_data(state["user_id"])

    if not data:
        return {"messages": [SystemMessage(content="📉 No performance data")]}

    context = "Weekly performance data:\n\n"

    for d in data:
        name = d.get("customerName") or "Unknown"
        pain = (d.get("painPoints") or "").strip()
        prob = d.get("probability") or 0

        context += f"{name} | Issue: {pain} | Probability: {prob}\n"

    prompt = f"""
You are a senior sales strategist.

Analyze this data and provide:
1. Key problems
2. Patterns
3. Actionable recommendations

Keep it practical and concise.

{context}
"""

    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [res]}

# ---------------- SALES EXPERT ----------------
async def sales_expert_node(state: AgentState):
    user_input = state["messages"][-1].content.lower()
    user_id = state["user_id"]

    # 🔥 STEP 1: Detect NON-sales queries
    non_sales_keywords = [
        "how are you",
        "who are you",
        "who is manager",
        "hello",
        "hi",
        "what is your name",
        "weather",
        "time",
        "date"
    ]

    if any(k in user_input for k in non_sales_keywords):
        return {
            "messages": state["messages"] + [
                SystemMessage(content="🤖 Please consult your manager for this query.")
            ]
        }

    # 🔥 STEP 2: Proceed with normal sales logic

    today = get_today_visits(user_id)
    followups = get_pending_followups(user_id)
    performance = get_weekly_performance_data(user_id)

    context = "Your sales data:\n\n"

    context += "Today's Visits:\n"
    for d in today[:5]:
        context += f"- {d.get('customerName')}: {d.get('summary')}\n"

    context += "\nPending Followups:\n"
    for d in followups[:5]:
        context += f"- {d.get('customerName')}: {d.get('nextStep')}\n"

    context += "\nRecent Issues:\n"
    for d in performance[:5]:
        context += f"- {d.get('customerName')}: {d.get('painPoints')}\n"

    prompt = f"""
You are a senior sales expert.

Answer ONLY based on sales context.

FORMAT:
- Keep it short
- Bullet points
- Actionable

DATA:
{context}

Question:
{user_input}
"""

    res = await llm.ainvoke([HumanMessage(content=prompt)])

    return {
        "messages": state["messages"] + [res]
    }

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

# ---------------- CLI ----------------
async def run_cli():
    print("\n🚀 AI Sales Intelligence CLI Ready")
    print("Type 'exit' to quit\n")

    USER_ID = "iPr3xaUi9DXQruUtefalaDVkAJH2"

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        try:
            result = await sales_brain.ainvoke({
                "messages": [HumanMessage(content=query)],
                "user_id": USER_ID
            })

            reply = result["messages"][-1].content
            print("\nAI:\n", reply, "\n")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

# ---------------- RUN ----------------
if __name__ == "__main__":
    asyncio.run(run_cli())