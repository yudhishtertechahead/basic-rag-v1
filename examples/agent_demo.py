import os
from dotenv import load_dotenv
from langchain.agents import create_agent

# Load environment variables from .env
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def run_agent(query: str):
    agent = create_agent(
        model="google_genai:gemini-3.5-flash",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result
