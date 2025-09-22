import os
import json
import uuid
import requests
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, List, Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

load_dotenv()
DB_PATH = Path("db")

### UTILS

def filter_fields(data: dict, fields_to_keep: dict) -> dict:
    """
    Keep only specific fields for each heading's rows.

    Parameters
    ----------
    data : dict
        Input dictionary with structure like:
        {
            "Heading1": {"rows": [ {field1: val, field2: val}, ... ]},
            "Heading2": {"rows": [ ... ]}
        }

    fields_to_keep : dict
        Dictionary mapping each heading to the list of fields to keep:
        {
            "Heading1": ["Field1"],
            "Heading2": ["Fields34"]
        }

    Returns
    -------
    dict
        New dictionary with only the requested fields kept.
    """
    result = {}

    for heading, content in data.items():
        # Get the fields we need for this heading (default to empty list if not provided)
        keep_fields = fields_to_keep.get(heading, [])

        # Filter each row to only keep those fields
        filtered_rows = []
        for row in content.get("rows", []):
            filtered_row = {k: row[k] for k in keep_fields if k in row}
            filtered_rows.append(filtered_row)
        result[heading] = {}
        result[heading]["rows"] = filtered_rows
        result[heading]["total_rows"] = len(content.get("rows", []))

    return result


### STATE
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    relevant_sheets: Dict[str, List[str]]
    data: str | dict

### TOOLS

@tool
def get_datetime() -> str:
    """
    Returns today's date and time in DD-MM-YYYY HH:MM:SS format.
    """
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")

@tool
def count_rows(rows: List) -> int:
    """Given a list of rows, it returns a the count"""
    return len(rows)

@tool
def get_summary(sheet_name: str, id_field: str, id_value: str, state: Annotated[AgentState, InjectedState]) -> Dict:
    """
    Fetch a specific row from a Google Sheet tab via Apps Script
    by matching an identification field and its value.

    Parameters
    ----------
    sheet_name : str
        Name of the sheet (e.g., "PO Pending", "Checklist").
    id_field : str
        The identification field/column name (e.g., "Indent Number", "Task ID").
    id_value : str | int
        The value to match inside that field (e.g., "PO12345").

    Returns
    -------
    dict | None
        Row data as JSON (Python dict). If not found, returns None.
    """
    data = state["data"][sheet_name]
    rows = data.get(sheet_name, {}).get("rows", [])

    # Find the row where id_field matches id_value
    for row in rows:
        if str(row.get(id_field, "")).strip() == str(id_value).strip():
            return row  # already dict/JSON

    return None



tools = [get_datetime, count_rows, get_summary]

#### INITIALIZATION

llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY")).bind_tools(tools)
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

#### NODES

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

10. PO Pending - pending purchase orders
   Fields: ["Timestamp","Indent Number","Have To Make Po","Party Name","Product Name","Quantity","Rate","Alumina %","Iron %","Lead Time To Lift Total Qty","PO Copy","Total Amount","Advance To Be Paid","To Be Paid Amount","When To Be Paid","Notes","Total Lifted","Pending Qty","Order Cancel Qty","Status"]

Your job is to
- Provide relevant sheet for the User Query
- At most 4 Relevant Fields/columns from the sheet for the User query
- When user asks only about the total number for rows in a sheet, like
       "Give me total number rows in X Sheet."
       "How many rows are in X sheet."
    Only Return the identity related field for such questions.
- When user asks total number of fields return all fields
- Aways have the identity related field
- But when user asks number of rows with some filter or condition, provide relevant fields for the condition. For Example,
        "Give me total number for pending rows in Checklist"
    Would have "Status" as the field for the "Checklist" Sheet

User: {state["messages"][-1]}
Give a list of only relevant sheets as a JSON. Don't use markup. Don't use `. Use the following format strictly
{{
    "Sheet Name": ["Field1", Field2, ...],
    ...
}}
If no relevant sheets found return empty JSON, when no relevant fields return empty array

### Rules for answering

1. Only use the fields listed for those sheets. Do not invent fields or values. 
2. If the user asks a vague question like *"How many are pending?"*:  
   - Look for the sheet(s) and fields where a `"Status"` or `"Pending"` column exists.  
   - If only one sheet contains pending status, assume that's what they mean.  
   - If multiple sheets could apply, politely ask for clarification.  
3. For greeting messages no sheets are relevant.
4. Never give more thatn 4 fields for one sheet.
    """

    response = llm.invoke(prompt)
    try:
        print(response.content.strip())
        sheets = json.loads(response.content.strip())
    except:
        sheets = {}

    return {**state, "relevant_sheets": sheets}

def loader(state: AgentState) -> AgentState:
    """Load selected sheets' data"""
    data = ""
    if len(state['relevant_sheets']) != 0:
        response = requests.request("get", (os.getenv("APPS_SCRIPT_URL") + f"?sheetNames={','.join(state['relevant_sheets'].keys())}"))
        data = response.json()
        for key in data.keys():
            data[key]["total_rows"] = len(data[key]["rows"]) - 1
    return {**state, "data": data}

def model(state: AgentState) -> AgentState:
    """Calls LLM to create response for user's query"""
    relevant_data = {}
    if len(state["relevant_sheets"].items()) != 0:
        relevant_data = filter_fields(state["data"], state["relevant_sheets"])

    system_prompt = SystemMessage(
            content=f"""
Your name is Diya. You are an AI agent for Botivate LLP, that resolves user queries regarding there process status and real time analytics. 
Your concisely reply the user with crisp and meaning full response. Your response should reflect Professionalism.

For any queries relating to the today's date and time use the date time use the `get_datetime` tool.
Whenever asked to give count or related queries, use the `count_rows` tool, with the list of rows relevant to user query.
Whenever you need more details about a row of data, use the `get_summary` tool with its arguements, this will return extra details about about the row in json.

Reply the user according the following data

{json.dumps(relevant_data, indent=2)}

### Rules for answering
1. Always decide which sheet(s) are relevant before answering.  
2. Only use the fields listed for those sheets. Do not invent fields or values.  
3. If user asks tell me the Total Rows, without any aggregation, just the count of total rows, Get it from the `total_rows` field of the sheet only, not the `count_tool`.
4. If the user asks a vague question like *"How many are pending?"*:  
   - Look for the sheet(s) where a `"Status"` or `"Pending"` column exists.  
   - If only one sheet contains pending status, assume that's what they mean.  
   - If multiple sheets could apply, politely ask for clarification.  
5. If the required information does not exist in the data, respond with:  
   **"The data does not contain this information."**  
6. Never fabricate rows or totals. Only count or extract from the provided JSON files.  
7. Always answer in a clear, concise, professional tone as Diya.
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