from typing import TypedDict, Annotated, List
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from prompt import CLASSIFICATION_PROMPT, COT_PROMPT

load_dotenv()

# database connection
sqlite_conn = sqlite3.connect("DataBase/checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# llm
llm = ChatGroq(model="llama3-70b-8192")

# Agent State
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# Separate routing logic for conditional edges
def route_to_node(state: AgentState) -> str:
    last_user_message = state["messages"][-1].content
    prompt = CLASSIFICATION_PROMPT.format(query=last_user_message)

    result = llm.invoke([HumanMessage(content=prompt)])
    response = result.content.lower()

    if "reasoning" in response:
        return "cot_chatbot_node"
    else:
        return "chatbot_node"

# Router node -- for state updates
def router_node(state: AgentState):
    # Simply pass through the state without modification
    return {"messages": state["messages"]}
    
# ----------- Simple Chatbot Node -----------
def chatbot_node(state: AgentState):
    # get full message history
    messages = state["messages"]
    # Keep only the last 10 messages (5 Human + 5 AI)
    recent_messages = messages[-10:]
    # Generate response from LLM using recent context
    response = llm.invoke(recent_messages)
    
    response.content = f"AI: {response.content.strip()}"

    return {"messages": recent_messages + [response]}

# ----------- CoT Chatbot Node ----------
def cot_chatbot_node(state: AgentState):
    messages = state["messages"]
    recent_messages = messages[-10:]
    last_user_message = recent_messages[-1]

    cot_prompt = COT_PROMPT.format(query=last_user_message.content)
    # Replace last user message with COT-styled one
    cot_messages = recent_messages[:-1] + [HumanMessage(content=cot_prompt)]
    # Get CoT response from LLM
    cot_response = llm.invoke(cot_messages)
    # Append AI message to history
    updated_messages = recent_messages + [AIMessage(content="AI (Chain-of-Thought): " + cot_response.content)]
    
    return {"messages": updated_messages}

# ----------- Graph Definition -----------
graph = StateGraph(AgentState)

# Nodes
graph.add_node("router", router_node) 
graph.add_node("chatbot_node", chatbot_node)
graph.add_node("cot_chatbot_node", cot_chatbot_node)

# Edges
graph.set_entry_point("router")
graph.add_edge("chatbot_node", END)
graph.add_edge("cot_chatbot_node", END)
graph.add_conditional_edges("router", route_to_node)

# Compile
app = graph.compile(checkpointer=memory)

# ----------- CLI Loop -----------
config = {"configurable": {"thread_id": "1" }}

if __name__ == "__main__":
    print("Smart ChatBot with Memory and CoT is Ready")
    print("=" * 80)

    while True:
        
        user_query = input("User: ")
        print("=" * 80)

        if user_query == "exit":
            print('GoodBay See You soon :)')
            break

        if user_query.lower() == 'reset':
            memory.delete_thread("1")  # Match the thread_id string
            print("Memory Has been Reset")
            continue

        result = app.invoke({
            'messages': [HumanMessage(content=user_query)],
        }, config = config)

        final_message = result['messages'][-1]
        print(final_message.content)
        print("=" * 80)