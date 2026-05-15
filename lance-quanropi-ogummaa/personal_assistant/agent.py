import os
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types
from personal_assistant.tools import search_knowledge_base

# Load environment variables from the .env file in the current directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

root_agent = Agent(
    model=Gemini(
        model='gemini-2.5-flash',
        retry_options=types.HttpRetryOptions(
            initial_delay=2,  # Start with 2 seconds
            max_delay=30,     # Max wait of 30 seconds
            attempts=5        # Try up to 5 times
        )
    ),
    name='root_agent',
    description='A helpful assistant that can search a LanceDB knowledge base.',
    instruction='''Answer user questions. 
If the question is about LanceDB, ADK, or the project architecture, search the knowledge base FIRST. 
If the knowledge base returns no relevant information, or if the question is about any other topic, answer to the best of your own general knowledge.''',
    tools=[search_knowledge_base]
)
