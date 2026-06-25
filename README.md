# SalesBot.AI

An AI-powered sales intelligence chatbot that helps sales representatives analyze visits, prioritize follow-ups, review weekly performance, and get actionable recommendations from real Firestore sales data.

The chatbot uses a multi-agent workflow built with LangGraph and Groq, exposed through a FastAPI backend for integration with Android, React Native, or web applications.

---

## Features

* Multi-agent sales workflow using LangGraph
* Sales summary agent for daily visit analysis
* Follow-up agent for pending customer actions
* Performance agent for weekly sales insights
* General sales expert agent for data-based sales advice
* Firestore integration for real visit data
* FastAPI `/chat` API endpoint
* CORS support for frontend integration
* CLI mode for quick testing
* Greeting support for messages such as:

  * `hi`
  * `hello`
  * `good morning`
  * `good evening`
* Guardrail that prevents unsupported answers about HR, finance, company policy, targets, or unrelated topics

---

## Tech Stack

* Python
* FastAPI
* LangGraph
* LangChain
* Groq
* Firestore
* Pydantic
* Uvicorn
* python-dotenv

---

## Project Structure

```text
multiagent_sales_assistant/
│
├── services/
│   └── firestore.py
│
├── .env
├── .gitignore
├── main.py
├── serviceAccountKey.json
├── requirements.txt
└── README.md
```

---

## How It Works

The chatbot receives a user message and routes it through a LangGraph workflow.

```text
User Message
     ↓
Supervisor Router
     ↓
Summary Agent / Follow-up Agent / Performance Agent / Sales Expert Agent
     ↓
Firestore Sales Data
     ↓
Groq LLM Response
     ↓
FastAPI Response
```

### Available Agents

#### 1. Summary Agent

Used for daily visit summaries and performance reports.

Example trigger from the frontend:

```text
8f3c1a9b2e7d4c6f
```

#### 2. Follow-up Agent

Used for pending follow-ups, customer callbacks, and next-step priorities.

Example trigger from the frontend:

```text
b72e4d1f9a3c8e65
```

#### 3. Performance Agent

Used for weekly performance analysis, customer pain points, and deal probability insights.

Example trigger from the frontend:

```text
c91d5e7a2b4f8c30
```

#### 4. Sales Expert Agent

Handles general sales-related questions such as:

```text
What should I prioritize today?
Which customer has the strongest opportunity?
How should I handle current customer objections?
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd multiagent_sales_assistant
```

### 2. Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install fastapi "uvicorn[standard]" langgraph langchain-groq langchain-core python-dotenv google-cloud-firestore pydantic
```

### 4. Create `.env`

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Add Firebase service account key

Download your Firebase service account JSON file and place it in the project root with this exact name:

```text
serviceAccountKey.json
```

The file must not be uploaded to GitHub.

---

## Run the Chatbot

### CLI mode

Run the chatbot in Terminal:

```bash
python3 main.py
```

Example:

```text
You: hi

AI:
Hello! I’m your Sales AI assistant. How can I help you with your sales work today?
```

Exit CLI mode:

```text
exit
```

---

## Run as FastAPI Backend

Start the server:

```bash
uvicorn main:app --reload
```

Open Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Usage

### Health Check

```http
GET /
```

Response:

```json
{
  "status": "🚀 Agentic Sales AI is running"
}
```

### Chat Endpoint

```http
POST /chat
```

Request body:

```json
{
  "message": "What should I prioritize today?",
  "user_id": "iPr3xaUi9DXQruUtefalaDVkAJH2"
}
```

Response:

```json
{
  "reply": "Based on today's visits, prioritize..."
}
```

---

## Firestore Requirements

Your Firestore `visits` collection should contain data similar to:

```json
{
  "salesPersonId": "user_id_here",
  "customerName": "ABC Enterprises",
  "summary": "Customer is interested in the premium plan.",
  "outcomeStatus": "Interested",
  "nextStep": "Schedule product demo next week.",
  "painPoints": "Current solution is expensive.",
  "probability": 75
}
```

---

## Security

Add these files to `.gitignore`:

```gitignore
venv/
.env
serviceAccountKey.json
__pycache__/
*.pyc
```

Never commit:

* Groq API keys
* Firebase service account keys
* `.env` files
* `venv` folder

---

## Future Improvements

* Replace hash-ID routing with natural-language LLM routing
* Add Redis caching for Firestore queries
* Add chat history with LangGraph MemorySaver
* Add JWT authentication
* Add role-based access for sales representatives and managers
* Add streaming responses
* Deploy backend using Render, Railway, or Google Cloud Run
* Connect to Android and React Native frontend

---

## Author

Built as an Agentic AI sales intelligence project using FastAPI, LangGraph, Groq, and Firestore.
