import datetime as dt
import os
import sys
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeInterpreterTool
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pathlib import Path
from pprint import pprint
from typing import Any

load_dotenv()

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(), 
    conn_str=os.environ["PROJECT_CONNECTION_STRING"])

with project_client:

    code_interpreter = CodeInterpreterTool()

    agent = project_client.agents.create_agent(
        model="gpt-4o-mini",
        name="my-agent",
        instructions="You are helpful agent",
        tools=code_interpreter.definitions,
        tool_resources=code_interpreter.resources)
    
    thread = project_client.agents.create_thread()

    user_query = "Could you please create a bar chart for the operating profit using the following data and provide the file to me? Bali: 100 Travelers, Paris: 356 Travelers, London: 900 Travelers, Tokyo: 850 Travellers"
    project_client.agents.create_message(thread_id=thread.id, role="user", content=user_query)
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
    
    if run.status == "failed":
        print(run.last_error)
        sys.exit(1)

    messages = project_client.agents.list_messages(thread_id=thread.id)
    for message in messages.data[::-1]:
        role = message.get("role", "assistant")
        for content in message.content:
            if content.type == "text":
                output = content.text.value
                # annotations = content.text.annotations
                # if len(annotations):
                #     for annotation in annotations:
                #         if annotation.type == "file_path":
                #             output += f"\n[File {annotation.file_path.file_id}]"
                #         else:
                #             print(f"Unknown annotation type: {annotation.type}")
                #             print(annotation)

                print(role + ": " + output)
            elif content.type == "image_file":
                print(f"[IMAGE FILE ID {content.image_file.file_id}]")
            else:
                print(f"Unknown content type: {content.type}")
                print(content)
        print()

    print("Files:")
    for file_path_annotation in messages.file_path_annotations:
        file_name = Path(file_path_annotation.text).name
        file_id = file_path_annotation.file_path.file_id
        project_client.agents.save_file(file_id=file_id, file_name=file_name)
        print(f"{file_id}: {file_name}")


    project_client.agents.delete_agent(agent.id)
    