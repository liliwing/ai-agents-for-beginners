import asyncio
import logging
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Load Environment variables. Required Variables: GITHUB_TOKEN
load_dotenv()

# Set up logging
setup_logging()
logging.basicConfig(
    format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("kernel").setLevel(logging.DEBUG)

####################################################################################
# Plugins: These are the components that are used by your AI services
# Plugins are named function containers. Each can contain one or more functions.
# Plugins can be registered with the kernel, which allows the kernel to use them
####################################################################################
class Light(BaseModel):
    id: int
    name: str
    on: bool

class LightsPlugin:

    def __init__(self):
        light_1 = Light(id=1, name="Table Lamp", on=False)
        light_2 = Light(id=2, name="Porch light", on=False)
        light_3 = Light(id=3, name="Chandelier", on=True)
        self.lights = [light_1, light_2, light_3]
    
    @kernel_function(name="get_lights", description="Gets a list of lights and their current state")
    def get_state(self):
        for light in self.lights: print(f"{light.name} is {'on' if light.on else 'off'}")
        return self.lights
    
    @kernel_function(name="change_state", description="Changes the state of the light")
    def change_state(self, id: int, on: bool):
        for light in self.lights:
            if light.id == id: 
                light.on = on
                print(f"Turned {'on' if on else 'off'} {light.name}")

async def main():

    ####################################################################################
    # Kernel: The kernel is the central component of Semantic Kernel.
    ####################################################################################
    # At its simplest, the kernel is a Dependency Injection container 
    # that manages all of the services and plugins necessary to run your AI application.
    kernel = Kernel()

    ####################################################################################
    # Services:
    # One of the main features of Semantic Kernel is its ability 
    # to add different AI services to the kernel. Supported services include 
    # Chat Completion, Text Generation, Embedding Generation, Text to Image,
    # Image to Text, Text to Audio and Audio to Text.
    ####################################################################################
    client = AsyncOpenAI(
        api_key=os.environ.get("GITHUB_TOKEN"), 
        base_url="https://models.inference.ai.azure.com/")

    # Add Chat completion service to kernel
    kernel.add_service(OpenAIChatCompletion(
        ai_model_id="gpt-4o-mini",
        async_client=client))
    
    # Retrieve service from kernel
    chat_completion_service: OpenAIChatCompletion = kernel.get_service(type=ChatCompletionClientBase)
    
    # Add 'lights' plugin to kernel
    kernel.add_plugin(LightsPlugin(), plugin_name="Lights")

    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Create a history of the conversation
    history = ChatHistory()

    user_input = None
    while True:
        user_input = input("# User: ")
        if user_input == "exit": break

        # Add user input to the history
        history.add_user_message(user_input)

        # Get the response from the AI
        result = await chat_completion_service.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel)

        # Print the results
        print("# Assistant: " + str(result))

        # Add the message from the agent to the chat history
        history.add_message(result)

if __name__ == "__main__":
    asyncio.run(main())