import asyncio
import os
from typing import Annotated, Any, List, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from services.firestore import (
    get_today_visits,
    get_pending_followups,
    get_weekly_performance_data,
)

load_dotenv()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

llm = ChatGroq(
    temperature=0.4,
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
)

# These IDs can be sent from your Android app for direct navigation.
SUMMARY_ID = "8f3c1a9b2e7d4c6f"
FOLLOWUPS_ID = "b72e4d1f9a3c8e65"
PERFORMANCE_ID = "c91d5e7a2b4f8c30"


# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "chat history"]
    user_id: str
    next_node: str


# -------------------------------------------------
# UTILITY
# -------------------------------------------------
def safe_to_string(content: Any) -> str:
    """Always return a JSON-safe string."""
    if content is None:
        return "No response generated."

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return " ".join(str(item) for item in content)

    return str(content)


def build_general_sales_context(user_id: str) -> str:
    """Fetch and format the sales data used by the general sales expert."""
    today = get_today_visits(user_id)
    followups = get_pending_followups(user_id)
    performance = get_weekly_performance_data(user_id)

    context = "YOUR VERIFIED SALES DATA\n\n"

    context += "TODAY'S VISITS:\n"
    if today:
        for visit in today[:5]:
            context += (
                f"- Customer: {visit.get('customerName') or 'Unknown'}\n"
                f"  Summary: {visit.get('summary') or 'No summary'}\n"
                f"  Outcome: {visit.get('outcomeStatus') or 'Not recorded'}\n"
            )
    else:
        context += "- No visits found.\n"

    context += "\nPENDING FOLLOW-UPS:\n"
    if followups:
        for visit in followups[:5]:
            context += (
                f"- Customer: {visit.get('customerName') or 'Unknown'}\n"
                f"  Next step: {visit.get('nextStep') or 'Not recorded'}\n"
            )
    else:
        context += "- No pending follow-ups found.\n"

    context += "\nWEEKLY PERFORMANCE / ISSUES:\n"
    if performance:
        for visit in performance[:5]:
            context += (
                f"- Customer: {visit.get('customerName') or 'Unknown'}\n"
                f"  Pain point: {visit.get('painPoints') or 'Not recorded'}\n"
                f"  Probability: {visit.get('probability') or 'Not recorded'}\n"
            )
    else:
        context += "- No weekly performance data found.\n"

    return context


# -------------------------------------------------
# SUPERVISOR
# -------------------------------------------------
async def supervisor_node(state: AgentState):
    query = safe_to_string(state["messages"][-1].content).strip()

    if query == SUMMARY_ID:
        node = "summary_agent"
    elif query == FOLLOWUPS_ID:
        node = "followup_agent"
    elif query == PERFORMANCE_ID:
        node = "performance_agent"
    else:
        node = "sales_expert_agent"

    print(f"ROUTER -> {node}")
    return {"next_node": node}


# -------------------------------------------------
# SUMMARY AGENT
# -------------------------------------------------
async def summary_node(state: AgentState):
    data = get_today_visits(state["user_id"])

    if not data:
        return {"messages": [SystemMessage(content="📊 No visits found for today.")]}

    context = "TODAY'S VISIT SUMMARIES:\n\n"

    for visit in data:
        name = visit.get("customerName") or "Unknown customer"
        summary = (visit.get("summary") or "No summary").strip()
        outcome = visit.get("outcomeStatus") or "Not recorded"

        context += f"- {name}\n  Summary: {summary}\n  Outcome: {outcome}\n\n"

    prompt = f"""
You are a senior sales manager.

Analyze the verified visit data below.

Return:
1. Overall performance summary
2. Key observations
3. Strong opportunities
4. Risks or blockers
5. Suggested next steps

Rules:
- Be concise and structured.
- Use only the provided data.
- Do not invent facts.

{context}
"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


# -------------------------------------------------
# FOLLOW-UP AGENT
# -------------------------------------------------
async def followup_node(state: AgentState):
    data = get_pending_followups(state["user_id"])

    if not data:
        return {"messages": [SystemMessage(content="📞 No pending follow-ups found.")]}

    context = "PENDING FOLLOW-UPS:\n\n"

    for visit in data:
        name = visit.get("customerName") or "Unknown customer"
        next_step = (visit.get("nextStep") or "No next step recorded").strip()
        probability = visit.get("probability") or "Not recorded"

        context += (
            f"- {name}\n"
            f"  Next step: {next_step}\n"
            f"  Deal probability: {probability}\n\n"
        )

    prompt = f"""
You are a sales execution manager.

Based only on these verified follow-ups, provide:

1. Urgent follow-ups
2. High-value opportunities
3. Priority order
4. Clear action for each important customer

Keep the answer practical and concise.

{context}
"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


# -------------------------------------------------
# PERFORMANCE AGENT
# -------------------------------------------------
async def performance_node(state: AgentState):
    data = get_weekly_performance_data(state["user_id"])

    if not data:
        return {"messages": [SystemMessage(content="📉 No weekly performance data found.")]}

    context = "WEEKLY PERFORMANCE DATA:\n\n"

    for visit in data:
        name = visit.get("customerName") or "Unknown customer"
        pain_point = (visit.get("painPoints") or "No pain point recorded").strip()
        probability = visit.get("probability") or "Not recorded"

        context += (
            f"- {name}\n"
            f"  Pain point: {pain_point}\n"
            f"  Deal probability: {probability}\n\n"
        )

    prompt = f"""
You are a senior sales strategist.

Analyze this verified weekly sales data and provide:

1. Key problems
2. Repeating patterns
3. Strongest opportunities
4. Actionable recommendations for next week

Do not invent facts. Keep it practical.

{context}
"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


# -------------------------------------------------
# GENERAL SALES EXPERT + GREETINGS + GUARDRAIL
# -------------------------------------------------
async def sales_expert_node(state: AgentState):
    user_input = safe_to_string(state["messages"][-1].content)
    normalized_input = user_input.lower().strip()

    # Allow normal greetings without checking Firestore or the guardrail.
    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
    }

    if normalized_input in greetings:
        return {
            "messages": [
                SystemMessage(
                    content=(
                        "Hello! I’m your Sales AI assistant. "
                        "How can I help you with your sales work today?"
                    )
                )
            ]
        }

    context = build_general_sales_context(state["user_id"])

    # This prevents unsupported answers about HR, policy, targets, etc.
    check_prompt = f"""
You are a strict classifier.

Can the user's question be answered accurately using ONLY the verified sales data?

Verified sales data:
{context}

User question:
{user_input}

Reply with exactly one word:
YES or NO

Reply NO when:
- the question is unrelated to sales data
- it asks about company policy, HR, finance, targets, or management decisions
- the data is insufficient
"""

    check = await llm.ainvoke([HumanMessage(content=check_prompt)])

    if safe_to_string(check.content).strip().upper() != "YES":
        return {
            "messages": [
                SystemMessage(
                    content=(
                        "⚠️ I do not have enough verified sales information "
                        "to answer this accurately. Please contact your Territory Manager."
                    )
                )
            ]
        }

    prompt = f"""
You are a senior sales expert with 20 years of experience.

Answer the user's question using ONLY the verified sales data below.

{context}

User question:
{user_input}

Rules:
- Give specific, actionable advice.
- Mention customer names only when relevant.
- Do not invent facts.
- Keep the answer structured with short bullet points.
"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


# -------------------------------------------------
# LANGGRAPH
# -------------------------------------------------
builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("summary_agent", summary_node)
builder.add_node("followup_agent", followup_node)
builder.add_node("performance_agent", performance_node)
builder.add_node("sales_expert_agent", sales_expert_node)

builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"],
    {
        "summary_agent": "summary_agent",
        "followup_agent": "followup_agent",
        "performance_agent": "performance_agent",
        "sales_expert_agent": "sales_expert_agent",
    },
)

builder.add_edge("summary_agent", END)
builder.add_edge("followup_agent", END)
builder.add_edge("performance_agent", END)
builder.add_edge("sales_expert_agent", END)

sales_brain = builder.compile()


# -------------------------------------------------
# FASTAPI
# -------------------------------------------------
app = FastAPI(title="Agentic Sales AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def root():
    return {"status": "🚀 Agentic Sales AI is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await sales_brain.ainvoke(
            {
                "messages": [HumanMessage(content=request.message)],
                "user_id": request.user_id,
            }
        )

        reply = safe_to_string(result["messages"][-1].content)
        print("RESPONSE:", reply)

        return {"reply": reply}

    except Exception as error:
        print("CHAT ERROR:", error)
        raise HTTPException(
            status_code=500,
            detail="Unable to process this sales request.",
        )


# -------------------------------------------------
# CLI TEST MODE
# -------------------------------------------------
async def run_cli():
    print("\n🚀 Agentic Sales AI CLI Ready")
    print("Type 'exit' to quit.\n")

    user_id = "iPr3xaUi9DXQruUtefalaDVkAJH2"

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        result = await sales_brain.ainvoke(
            {
                "messages": [HumanMessage(content=query)],
                "user_id": user_id,
            }
        )

        print("\nAI:\n", safe_to_string(result["messages"][-1].content), "\n")


if __name__ == "__main__":
    asyncio.run(run_cli())