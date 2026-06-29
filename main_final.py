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

    # -------------------------------
    # Friendly conversation
    # -------------------------------
    friendly_responses = {
        "hi": "Hi! 👋 I'm your Sales AI assistant. How can I help you today?",
        "hello": "Hello! 👋 Ready to help you with your customers, follow-ups and sales performance.",
        "hey": "Hey! What can I help you with today?",
        "good morning": "Good morning! ☀️ Let's make today a productive sales day.",
        "good afternoon": "Good afternoon! How's your day going? Need help with your customers?",
        "good evening": "Good evening! Hope your visits went well. How can I assist?",
        "good night": "Good night! 🌙 Have a great rest.",
        "thanks": "You're welcome! Happy selling. 😊",
        "thank you": "You're most welcome! Let me know if I can help with anything else.",
        "bye": "Goodbye! Wishing you successful sales and great customer meetings."
    }

    if normalized_input in friendly_responses:
        return {
            "messages": [
                SystemMessage(
                    content=friendly_responses[normalized_input]
                )
            ]
        }

    context = build_general_sales_context(state["user_id"])

    # -------------------------------
    # Intent Classification
    # -------------------------------
    intent_prompt = f"""
You are an intent classifier.

Classify the user's query into exactly ONE category.

CRM
Questions requiring the user's CRM or visit data.

Examples:
- today's visits
- summarize my day
- weekly performance
- follow-ups
- customer status
- deal probability
- today's customers
- risky customers
- opportunities

COACHING
General sales guidance that does NOT require CRM data.

Examples:
- negotiation tips
- handling objections
- customer wants discount
- closing techniques
- sales strategy
- how to convince customer
- how to improve conversion

OUT_OF_SCOPE
Anything unrelated to sales.

Reply with exactly one word.

User:
{user_input}
"""

    intent = (
        safe_to_string(
            (
                await llm.ainvoke(
                    [HumanMessage(content=intent_prompt)]
                )
            ).content
        )
        .strip()
        .upper()
    )

    if intent == "OUT_OF_SCOPE":
        return {
            "messages": [
                SystemMessage(
                    content=(
                        "I'm designed to help with sales, customer visits, follow-ups, "
                        "sales strategy, negotiations and CRM insights. "
                        "Please ask me something related to sales."
                    )
                )
            ]
        }

    # -------------------------------
    # Final Prompt
    # -------------------------------
    prompt = f"""
You are a Regional Sales Manager with over 20 years of enterprise B2B sales experience.

Your job is to coach salespeople and help them make better sales decisions.

The user may ask either:

1. CRM questions
2. General sales coaching questions

If it is a CRM question:
- Use ONLY the verified CRM data below.
- Never invent customer names, visits or statistics.
- If the CRM data is insufficient, clearly say so and then provide general sales guidance.

If it is a coaching question:
- Ignore CRM limitations.
- Answer using proven sales best practices.
- Give practical advice.

Always:

• Be friendly and conversational.
• Keep answers concise.
• Explain your reasoning.
• Finish with one practical next action.

Verified CRM Data:

{context}

User Question:

{user_input}
"""

    response = await llm.ainvoke(
        [HumanMessage(content=prompt)]
    )

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