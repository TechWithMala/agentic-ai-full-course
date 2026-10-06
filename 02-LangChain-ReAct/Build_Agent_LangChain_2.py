# Build an agentic AI application using LangChain

# pip install -U langchain
# pip install -U langchain-openai

# import libraries
import openai
import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Load the environment variables from .env file
load_dotenv()

# Read the API keys
openai_api_key = os.getenv('OPENAI_API_KEY')
serper_api_key = os.getenv('SERPER_API_KEY')

os.environ["OPENAI_API_KEY"] =openai_api_key   
os.environ["SERPER_API_KEY"] =serper_api_key   

 # Initialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# List top3 Medicare providers in the United States. Also, assume a 50-day joint replacement program for a patient costs were: Hospital $21500, Post-acute care $8400, Readmission $3200, Target price $31000. Are we above or below target?

# Define tools
tools = load_tools(["google-serper", "llm-math"], llm=llm)

# Create the ReAct prompt
react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate(
    template=react_template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)

# Initialize Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

result = agent_executor.invoke({"input": "List top3 Medicare providers in the United States. Also, assume a 50-day joint replacement program for a patient costs were: Hospital $21500, Post-acute care $8400, Readmission $3200, Target price $31000. Are we above or below target?"})
print("\n" + "="*50)
print("FINAL RESULT:")
print("="*50)
print(result.get("output", result))
