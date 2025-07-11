from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_tavily import TavilySearch
import re
import json
from prompt import CLASSIFICATION_PROMPT, COT_PROMPT, REACT_PROMPT

load_dotenv()

# database connection
sqlite_conn = sqlite3.connect("DataBase/checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# llm
llm = ChatGroq(model="llama3-70b-8192")

# Initialize tools
tavily_tool = TavilySearch(max_results=3)

# Tool registry for easy extension
AVAILABLE_TOOLS = {
    "search_web": {
        "tool": tavily_tool,
        "description": "Search the web for current information date, time, news, weather, stock prices, etc."
    }
    # Add more tools here in the future
}

# Agent State
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# Separate routing logic for conditional edges
def route_to_node(state: AgentState) -> str:
    last_user_message = state["messages"][-1].content
    prompt = CLASSIFICATION_PROMPT.format(query=last_user_message)

    result = llm.invoke([HumanMessage(content=prompt)])
    response = result.content.lower().strip()

    # print(f"🔍 Classification: {response}")  # Debug info
    
    if "reasoning" in response:
        return "cot_chatbot_node"
    elif "search" in response:
        return "react_chatbot_node"
    else:
        return "chatbot_node"

# Router node, for state updates
def router_node(state: AgentState):
    # Simply pass through the state without modification
    return {"messages": state["messages"]}
    
# ----------- Simple Chatbot Node -----------
def chatbot_node(state: AgentState):
    messages = state["messages"]   
    recent_messages = messages[-10:]   # (5 Human + 5 AI)
    response = llm.invoke(recent_messages)  
    final_response = AIMessage(content=response.content.strip(), name="simple")
    return {"messages": recent_messages + [final_response]}

# ----------- CoT Chatbot Node ----------
def cot_chatbot_node(state: AgentState):
    messages = state["messages"]
    recent_messages = messages[-10:]
    last_user_message = recent_messages[-1]

    cot_prompt = COT_PROMPT.format(query=last_user_message.content)
    cot_messages = recent_messages[:-1] + [HumanMessage(content=cot_prompt)]
    cot_response = llm.invoke(cot_messages)  
    
    final_response = AIMessage(content=cot_response.content.strip(), name="cot")  
    
    return {"messages": recent_messages + [final_response]}

# ----------- ReAct Chatbot Node with Enhanced Tool Support -----------
def react_chatbot_node(state: AgentState):
    messages = state["messages"]
    recent_messages = messages[-10:]
    last_query = recent_messages[-1].content

    print(f"\n🔄 Starting ReAct for query: {last_query}")

    # Initialize ReAct state
    react_state = {
        "messages": recent_messages,
        "history": [],
        "iteration": 0,
        "final_answer": "",
        "max_iterations": 3
    }

    def parse_llm_response(response_text: str) -> Dict[str, str]:
        """Enhanced parsing of LLM response"""
        result = {
            "thought": "",
            "action": "",
            "action_input": "",
            "final_answer": ""
        }
        
        # Parse Final Answer first
        final_answer_pattern = r"Final Answer:\s*(.+?)(?:\n\n|$)"
        final_match = re.search(final_answer_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result
        
        # Parse Thought
        thought_pattern = r"Thought:\s*(.+?)(?=\nAction:|$)"
        thought_match = re.search(thought_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Parse Action
        action_pattern = r"Action:\s*([^\n]+)"
        action_match = re.search(action_pattern, response_text, re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # Parse Action Input
        input_pattern = r"Action Input:\s*(.+?)(?=\n\w+:|$)"
        input_match = re.search(input_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if input_match:
            result["action_input"] = input_match.group(1).strip()
        
        return result

    def execute_tool(tool_name: str, tool_input: str) -> str:
        """Execute tool with proper error handling"""
        if tool_name not in AVAILABLE_TOOLS:
            available = ", ".join(AVAILABLE_TOOLS.keys())
            return f"Error: Unknown tool '{tool_name}'. Available tools: {available}"
        
        try:
            tool = AVAILABLE_TOOLS[tool_name]["tool"]
            result = tool.invoke(tool_input)
            
            # Format the result nicely
            if isinstance(result, str):
                return result
            elif isinstance(result, list):
                return "\n".join([str(item) for item in result])
            else:
                return str(result)
                
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def build_context(history: List[Dict]) -> str:
        """Build formatted context from history"""
        if not history:
            return ""
        
        context = "\nPrevious steps:\n"
        for i, step in enumerate(history, 1):
            context += f"\nStep {i}:\n"
            context += f"Thought: {step['thought']}\n"
            context += f"Action: {step['action']}\n"
            context += f"Action Input: {step['action_input']}\n"
            context += f"Observation: {step['observation']}\n"
        
        return context

    def get_tools_description() -> str:
        """Get formatted description of available tools"""
        tools_desc = []
        for tool_name, tool_info in AVAILABLE_TOOLS.items():
            tools_desc.append(f"- {tool_name}: {tool_info['description']}")
        return "\n".join(tools_desc)

    # ReAct loop
    while react_state["iteration"] < react_state["max_iterations"]:
        react_state["iteration"] += 1
        
        print(f"\n🧠 ReAct Iteration {react_state['iteration']}")
        
        # Build context and prompt
        context = build_context(react_state["history"])
        tools_desc = get_tools_description()
        
        # Create the prompt
        prompt_text = REACT_PROMPT.format(
            question=last_query,
            context=context,
            tools=tools_desc,
            iteration=react_state["iteration"],
            max_iterations=react_state["max_iterations"]
        )
        
        # Get LLM response
        response = llm.invoke([SystemMessage(content=prompt_text)])
        response_text = response.content.strip()
        
        print(f"📝 LLM Response:\n{response_text}")
        
        # Parse the response
        parsed = parse_llm_response(response_text)
        
        # Check for final answer
        if parsed["final_answer"]:
            react_state["final_answer"] = parsed["final_answer"]
            # print(f"✅ Final Answer Found: {parsed['final_answer']}")
            break
        
        # Execute action if present
        observation = ""
        if parsed["action"] and parsed["action_input"]:
            # print(f"🔧 Executing: {parsed['action']} with input: {parsed['action_input']}")
            observation = execute_tool(parsed["action"], parsed["action_input"])
            # print(f"🔍 Observation: {observation}")
        else:
            observation = "No valid action found in response."
            print(f"❌ {observation}")
        
        # Add to history
        react_state["history"].append({
            "thought": parsed["thought"],
            "action": parsed["action"],
            "action_input": parsed["action_input"],
            "observation": observation
        })
        
        # Check if we should stop due to successful action
        if parsed["action"] and "Error" not in observation:
            # Let the next iteration decide if we have enough info
            continue
    
    # Generate final answer if not found
    if not react_state["final_answer"]:
        print("🎯 Generating final answer from collected information...")
        
        
        context = build_context(react_state["history"])
        final_prompt = f"""Based on the information gathered through the ReAct process, provide a comprehensive final answer to the user's question.

Question: {last_query}
{context}

Please provide a clear, informative, and well-structured final answer:"""
        
        final_response = llm.invoke([SystemMessage(content=final_prompt)])
        react_state["final_answer"] = final_response.content.strip()
    
    # print(f"\n✨ Final Answer: {react_state['final_answer']}")
    
    # Return the final response
    final_response = AIMessage(content=react_state["final_answer"], name="react")
    return {"messages": recent_messages + [final_response]}

# ----------- Graph Definition -----------
graph = StateGraph(AgentState)

# Nodes
graph.add_node("router", router_node) 
graph.add_node("chatbot_node", chatbot_node)
graph.add_node("cot_chatbot_node", cot_chatbot_node)
graph.add_node("react_chatbot_node", react_chatbot_node)

# Edges
graph.set_entry_point("router")
graph.add_edge("chatbot_node", END)
graph.add_edge("cot_chatbot_node", END)
graph.add_edge("react_chatbot_node", END)
graph.add_conditional_edges("router", route_to_node)

# Compile
app = graph.compile(checkpointer=memory)

# ----------- CLI Loop -----------
config = {"configurable": {"thread_id": "1"}}

if __name__ == "__main__":
    print("🤖 Smart ChatBot with Enhanced ReAct is Ready!")
    print("=" * 80)
    print("Commands: 'exit' to quit, 'reset' to clear memory")
    print("=" * 80)

    while True:
        try:
            user_query = input("User: ").strip()
            
            if not user_query:
                continue
                
            print("=" * 80)

            if user_query.lower() == "exit":
                print('👋 Goodbye! See you soon!')
                break

            if user_query.lower() == 'reset':
                memory.delete_thread("1")
                print("🔄 Memory has been reset!")
                continue

            # Process the query
            result = app.invoke({
                'messages': [HumanMessage(content=user_query)],
            }, config=config)

            final_message = result['messages'][-1]
            
            # Display response with appropriate prefix
            if getattr(final_message, "name", "") == "cot":
                prefix = "🧠 AI (Chain-of-Thought):"
            elif getattr(final_message, "name", "") == "react":
                print("=" * 80)
                prefix = "🔍 AI (ReAct + Web Search):"
            else:
                prefix = "💬 AI:"

            print(f"{prefix} {final_message.content}")
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye! See you soon!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again.")
            continue