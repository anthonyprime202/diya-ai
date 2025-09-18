import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # Import the CORS middleware
from agent import agent  # import your LangGraph graph
from script import main  # import the main() function from script.py

app = FastAPI(title="Diya Analytics Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# Request/Response models
class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    reply: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    # Start graph execution with user message
    events = agent.stream({"messages": [("user", req.message)]}, config, stream_mode="values")

    reply = None
    for event in events:
        if "messages" in event:
            reply = event["messages"][-1].content

    return QueryResponse(reply=reply)


# ✅ New endpoint to run main() from script.py
@app.post("/update")
def run_script():
    try:
        result = main()  # call the main function
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
