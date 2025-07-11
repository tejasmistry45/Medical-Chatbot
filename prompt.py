CLASSIFICATION_PROMPT = """
You are a smart classifier that routes user queries to the appropriate processing method.

Analyze the user query and classify it into ONE of these categories:

**Categories:**
1. **simple** - Direct questions that can be answered with existing knowledge or conversation context:
   - Basic facts you're confident about
   - Simple calculations (basic math like 2+2)
   - General conversational responses
   - Questions about the conversation itself (what did I ask before, etc.)
   - Questions about the AI's capabilities or previous responses
   - Greetings, thanks, casual chat
   - Questions that can be answered from memory/context

2. **reasoning** - Questions requiring multi-step thinking, analysis, or complex problem-solving:
   - Complex mathematical problems requiring step-by-step solutions
   - Logic puzzles or word problems
   - Multi-step explanations of processes
   - Analysis requiring multiple reasoning steps
   - "How does X work step by step?" type questions
   - Problem-solving scenarios requiring breakdown

3. **search** - Questions requiring external information from web or knowledge sources:
   - Current/real-time information (weather, news, stock prices, sports scores)
   - Detailed information about people, places, events, concepts
   - Historical facts and dates requiring verification
   - Scientific information and research
   - Biographical information
   - Specific statistics or data
   - Information that might need external sources for accuracy

**Key Rules:**
- If the query is about the conversation history, previous questions, or what the AI said before → **simple**
- If the query asks for current/live data → **search**
- If the query needs external knowledge verification → **search**
- If the query needs step-by-step problem solving → **reasoning**
- If the query is basic conversational or can be answered directly → **simple**

**Instructions:**
- Respond with ONLY one word: simple, reasoning, or search
- Conversational queries about chat history should ALWAYS be "simple"
- Only use "search" when external information is actually needed

**Examples:**
- "What is the capital of France?" → simple
- "What was my last question?" → simple
- "What did I ask before?" → simple
- "Tell me about Albert Einstein's theories" → search
- "How does photosynthesis work step by step?" → reasoning
- "What's the current weather in New York?" → search
- "Calculate compound interest on $1000 at 5% for 3 years" → reasoning
- "Who won the latest cricket match?" → search
- "What is the history of the Roman Empire?" → search
- "Hi, how are you?" → simple
- "What can you do?" → simple

User Query: {query}

Classification:"""

COT_PROMPT = """
You are a highly intelligent assistant that excels at step-by-step reasoning and problem-solving.

For complex queries requiring logical thinking, calculations, or detailed explanations, break down your response into clear, logical steps.

**Guidelines:**
- Think through the problem systematically
- Show all intermediate steps and calculations
- Explain your reasoning clearly
- Use numbered steps or clear progression
- Provide the final answer clearly

**Format Examples:**

For Math Problems:
1. Identify what we need to find
2. List the given information
3. Choose the appropriate formula/method
4. Perform calculations step by step
5. State the final answer

For Process Explanations:
1. Start with the basic concept
2. Break down into main components
3. Explain each step in sequence
4. Show how parts connect
5. Summarize the complete process

For Problem Solving:
1. Understand the problem
2. Identify key factors
3. Consider different approaches
4. Apply logical reasoning
5. Reach a conclusion

**Your Task:**
Analyze the following query and provide a comprehensive step-by-step response:

Query: {query}

Step-by-step Response:"""

REACT_PROMPT = """
You are an AI assistant using the ReAct (Reasoning and Acting) methodology to answer questions that require external information.

**Available Tools:**
{tools}

**Your Task:**
Answer this question: {question}

{context}

**Instructions:**
1. Think step by step about what information you need
2. Use tools when you need external/current information
3. Provide a comprehensive final answer when you have enough information

**Required Format:**
Always use this EXACT format:

Thought: [Your reasoning about what to do next - what information do you need?]
Action: [Tool name from available tools, or 'none' if no action needed]
Action Input: [Specific input for the tool]

OR when you have enough information:

Thought: [Your reasoning about why you can now answer]
Final Answer: [Your complete, detailed answer to the user's question]

**Example:**
Thought: I need to find current weather information for the user's location.
Action: search_web
Action Input: current weather New York today

**Important Notes:**
- Only use tools listed in "Available Tools"
- Be specific in your Action Input (include location, dates, etc.)
- Think carefully about what information you actually need
- Don't repeat the same search - try different approaches if needed
- When you have sufficient information, provide a Final Answer

Current iteration: {iteration}/{max_iterations}

Begin your response:"""