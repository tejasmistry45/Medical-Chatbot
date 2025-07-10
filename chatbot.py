from typing import TypedDict, Annotated, List
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

# database connection
sqlite_conn = sqlite3.connect("DataBase/checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# llm
llm = ChatGroq(model="llama3-70b-8192")

# Agent State
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# create a node
def chatbot(state: AgentState):

    # get full message history
    messages = state["messages"]

    # Keep only the last 10 messages (5 Human + 5 AI)
    recent_messages = messages[-10:]

    # Generate response from LLM using recent context
    response = llm.invoke(recent_messages)

    # add new AI message to the messages
    updated_messages = recent_messages + [response]

    return {
        "messages": updated_messages
    }


graph = StateGraph(AgentState)

graph.set_entry_point("chatbot_node")
graph.add_node("chatbot_node", chatbot)
graph.add_edge("chatbot_node", END)

app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1" }}

if __name__ == "__main__":
    print("Simple ChatBot is Ready")
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
        print("AI Response: ", final_message.content)
        print("=" * 80)