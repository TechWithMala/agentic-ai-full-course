# Build an agentic AI application using LangChain

# Install required packages:
# python -m pip install -U langchain langchain-core langchain-openai langchain-classic langchain-community ddgs python-dotenv numexpr

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_classic.agents import AgentExecutor, create_react_agent

import numexpr


# 1. Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. "
        "Add it to your .env file."
    )


# 2. Initialize the OpenAI model
openai_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)


# 3. Define the calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate basic mathematical expressions.
    Example: (25 * 4) + 10.
    Use only arithmetic expressions with numbers and operators.
    """
    try:
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# 4. Define the DuckDuckGo search tool
search = DuckDuckGoSearchResults(
    max_results=5
)

tools = [
    calculator,
    search
]


# 5. Create the ReAct prompt
react_template = """Answer the following question as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: decide what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate(
    template=react_template,
    input_variables=[
        "tools",
        "tool_names",
        "input",
        "agent_scratchpad"
    ]
)


# 6. Create the ReAct agent
agent = create_react_agent(
    llm=openai_llm,
    tools=tools,
    prompt=prompt
)


# 7. Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8
)


# 8. Invoke the agent
question = (
    "What is the latest iPhone model and its price in USD? "
    "Search for the latest available USD-to-GBP exchange rate, "
    "calculate the approximate price in British pounds, "
    "and explain the result. Include the sources found by search."
)

response = agent_executor.invoke({
    "input": question
})


# 9. Print the final answer
print("\nFinal Answer:")
print(response["output"])
