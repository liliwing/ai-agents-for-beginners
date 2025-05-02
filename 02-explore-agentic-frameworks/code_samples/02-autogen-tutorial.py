import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.models import UserMessage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load Environment variables. Required Variables: GITHUB_TOKEN
load_dotenv()

client = AzureAIChatCompletionClient(
    model="gpt-4o-mini",
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN")),
    model_info={
        "structured_output": True,
        "json_output": True,
        "function_calling": True,
        "vision": True,
        "family": "unknown",
    },
)

async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."

agent = AssistantAgent(
    name="weather_agent",
    model_client=client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,
)


async def main():

    result = await client.create([UserMessage(content="What is the capital of France?", source="user")]) 
    print(result)   
    
    await Console(agent.run_stream(task="What is the weather in New York?"))
    # Close the connection to the model client.
    await client.close()

    
    
if __name__ == "__main__":
    asyncio.run(main())