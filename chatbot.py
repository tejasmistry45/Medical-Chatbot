from typing import TypedDict, Annotated, List
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# llm
llm = ChatGroq(model="llama3-70b-8192")

# Agent State
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# create a node
def chatbot(state: AgentState):

    messages = state['messages']
    response = llm.invoke(messages)
    updated_messages = messages + [response]

    return {
        "messages": updated_messages
    }

graph = StateGraph(AgentState)

graph.set_entry_point("chatbot_node")
graph.add_node("chatbot_node", chatbot)
graph.add_edge("chatbot_node", END)

app = graph.compile()

config = {"configurable": {"thread_id":1}}

if __name__ == "__main__":
    print("Simple ChatBot is Ready")
    print("=" * 80)

    while True:
        user_query = input("User: ")
        print("=" * 80)
        if user_query == "exit":
            print("GoodBay See You soon :)")
            break
        
        result = app.invoke({
            'messages': [HumanMessage(content=user_query)],
        }, config = config)

        final_message = result['messages'][-1]
        print("AI Response: ", final_message.content)
        print("=" * 80)