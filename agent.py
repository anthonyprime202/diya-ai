import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

load_dotenv()
DB_PATH = Path("db")

### STATE
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    relevant_sheets: List[str]
    data: dict

### TOOLS

@tool
def get_datetime() -> str:
    """
    Returns today's date and time in DD-MM-YYYY HH:MM:SS format.
    """
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")


tools = [get_datetime]

#### NODES

llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY")).bind_tools(tools)

def selector(state: AgentState) -> AgentState:
    """Ask LLM which sheets are relevant"""

    prompt = f"""
The data is organized into sheets. Each sheet has a specific meaning and fields:

1. Checklist — recurring tasks
   Fields: [Timestamp, Task ID, Firm, Given By, Name, Task Description, Task Start Date, Freq, Enable Reminders, Require Attachment, Actual, Delay, Status, Remarks, Uploaded Image]

2. Delegation — task delegation
   Fields: [Timestamp, Task ID, Firm, Given By, Name, Task Description, Task Start Date, Freq, Enable Reminders, Require Attachment, Planned Date, Actual, Delay, Status, Update Date, Reasons, Total Extent]

3. Purchase Intransit — material not yet received
   Fields: [Timestamp, LN-Lift Number, Type, Po Number, Bill No., Party Name, Product Name, Qty, Area Lifting, Lead Time To Reach Factory, Truck No., Driver No., Transporter Name, Bill Image, Bilty No., Type Of Rate, Rate, Truck Qty, Material Rate, Bilty Image, Expected Date To Reach]

4. Purchase Receipt — material received
   Fields: [Timestamp, Lift Number, PO Number, Bill Number, Party Name, Product Name, Date Of Receiving, Total Bill Quantity, Actual Quantity, Qty Difference, Physical Condition, Moisture, Physical Image Of Product, Image Of Weight Slip, Bilty Image, Bilty No., Qty Difference Status, Difference Qty, Type]

5. Orders Pending — pending sales orders
   Fields: [Timestamp, DO-Delivery Order No., PARTY PO NO (As Per Po Exact), Party PO Date, Party Names, Product Name, Quantity, Rate Of Material, Type Of Transporting, Upload SO, Is This Order Through Some Agent, Order Received From, Type Of Measurement, Contact Person Name, Contact Person WhatsApp No., Alumina%, Iron%, Type Of PI, Lead Time For Collection Of Final Payment, Quantity Delivered, Order Cancel, Pending Qty, Material Return, Status]

6. Sales Invoices — delivery details
   Fields: [Timestamp, Bill Date, Delivery Order No., Party Name, Product Name, Quantity Delivered., Bill No., Logistic No., Rate Of Material, Type Of Transporting, Transporter Name, Vehicle Number.]

7. Collection Pending — collections to be received
   Fields: [Party Names, Total Pending Amount, Expected Date Of Payment, Collection Remarks]

8. Production Orders — production orders
   Fields: [Timestamp, Delivery Order No., Party Name, Product Name, Order Quantity, Expected Delivery Date, Order Cancel, Actual Production Planned, Actual Production Done, Stock Transfered, Quantity Delivered, Quantity In Stock, Planning Pending, Production Pending, Status]

9. Job Card Production — job card details
   Fields: [Timestamp, Do Number, Party Name, Machine Name, Job Card No., Date Of Production, Name Of Supervisor, Product Name, Quantity Of FG]

Your job is to find the relevant sheet for the User's Query.
User: {state["messages"][-1]}
Give a list of only relevant sheets as a JSON array. Don't use markup. Don't use `.
If no relevant sheets found return empty JSON array

### Rules for answering

1. Only use the fields listed for those sheets. Do not invent fields or values. 
2. If the user asks a vague question like *"How many are pending?"*:  
   - Look for the sheet(s) where a `"Status"` or `"Pending"` column exists.  
   - If only one sheet contains pending status, assume that's what they mean.  
   - If multiple sheets could apply, politely ask for clarification.  
    """

    response = llm.invoke(prompt)
    try:
        sheets = json.loads(response.content.strip())
    except:
        sheets = []

    return {**state, "relevant_sheets": sheets}

def loader(state: AgentState) -> AgentState:
    """Load selected sheets' data"""
    data = {}
    for sheet in state["relevant_sheets"]:
        file_name = sheet.replace(" ", "_") + ".json"
        file_path = DB_PATH / file_name
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data[sheet] = json.load(f)
    return {**state, "data": data}

def model(state: AgentState) -> AgentState:
    """Calls LLM to create response for user's query"""

    system_prompt = SystemMessage(
            content=f"""
Your name is Diya. You are an AI agent for Botivate LLP, that resolves user queries regarding there process status and real time analytics. 
Your concisely reply the user with crisp and meaning full response. Your response should reflect Professionalism.

For any queries relating to the today's date and time use the date time use the `get_datetime` tool.

Reply the user according the following data

{json.dumps(state["data"], indent=2)}

### Rules for answering
1. Always decide which sheet(s) are relevant before answering.  
2. Only use the fields listed for those sheets. Do not invent fields or values.  
3. If the user asks a vague question like *"How many are pending?"*:  
   - Look for the sheet(s) where a `"Status"` or `"Pending"` column exists.  
   - If only one sheet contains pending status, assume that's what they mean.  
   - If multiple sheets could apply, politely ask for clarification.  
4. If the required information does not exist in the data, respond with:  
   **"The data does not contain this information."**  
5. Never fabricate rows or totals. Only count or extract from the provided JSON files.  
6. Always answer in a clear, concise, professional tone as Diya.  
"""
    )

    response = llm.invoke([system_prompt] + state["messages"])

    return {**state, "messages": [response]}

tool_node = ToolNode(tools)


### SETUP

memory = MemorySaver()

agent = (StateGraph(AgentState)
         .add_node("selector", selector)
         .add_node("loader", loader)
         .add_node("model", model)
         .add_node("tools", tool_node)
         
         .set_entry_point("selector")
         .add_edge("selector", "loader")
         .add_edge("loader", "model")
         .add_edge("tools", "model")
         .add_conditional_edges("model", tools_condition)
         .compile(checkpointer=memory))

if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("You: ")
        if user_input == "q":
            print("Goodbye!")
            break

        for event in agent.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values",
        ):
            message = event["messages"][-1]
            if not isinstance(message, tuple):
                message.pretty_print()