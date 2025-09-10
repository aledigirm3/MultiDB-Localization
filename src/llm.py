from groq import Groq
import os

#==============.env for llm=====================
from pathlib import Path
from dotenv import load_dotenv

env_path = "../.env"
load_dotenv(dotenv_path=env_path)
#===============================================

api_key = os.environ.get("GROQ_API_KEY")
client = Groq(
    api_key=api_key,
)

def query_groq(messages: list, model: str = "openai/gpt-oss-120b", temperature: float = 0.0, maxTokens: int = 100):
    """
    Function to query the GROQ API with a list of messages.
    
    Args:
        messages (list): The list of messages to send to the model, e.g.
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a short poem about rain."}
            ]
        model (str): The model to use for the completion.
        temperature (float): The temperature to use for the completion.
        max_tokens (int): Maximum number of tokens in the output.
        
    Returns:
        str: The completion from the model.
    """
    chat_completion = client.chat.completions.create(
        messages=messages,
        
        model=model,

        # Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become deterministic and repetitive.
        temperature=temperature,

        max_tokens=maxTokens
    )

    return chat_completion.choices[0].message.content

