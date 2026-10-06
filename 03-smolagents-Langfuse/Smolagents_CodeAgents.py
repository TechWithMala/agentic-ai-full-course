# pip install smolagents[telemetry] opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents
# pip install smolagents -U duckduckgo-search ddgs huggingface_hub

import base64
import os
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Read the API keys
langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

os.environ["LANGFUSE_PUBLIC_KEY"] =langfuse_public_key   
os.environ["LANGFUSE_SECRET_KEY"] =langfuse_secret_key   

# check if the keys are set
if not langfuse_public_key or not langfuse_secret_key:
    raise ValueError("Langfuse public or secret keys are missing!")

LANGFUSE_AUTH = base64.b64encode(f"{langfuse_public_key}:{langfuse_secret_key}".encode()).decode()

#os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # EU data region
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://us.cloud.langfuse.com/api/public/otel" # US data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

print("Langfuse environment variables set successfully!")

from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
# Connected to Langfuse...

from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN") # Access the HuggingFace Token
os.environ["HF_TOKEN"] =hf_token 

# Check if the Hugging Face token is set
if hf_token:
    login(hf_token)
    print("Successfully logged in to Hugging Face!")
else:
    print("Token not found or notebook access not enabled.")
    
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
# Set the InferenceClientModel
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-7B-Instruct"
)

# Initialize Smolagents CodeAgent by passing tools and model
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

# Run the Smolagents CodeAgent
agent.run("Research the top real-world use cases of AI in the workplace and summarize how companies are implementing them.")