import os
import botocore.session
from ansi_colors import *

#==============.env for llm=====================
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
#===============================================

DEFAULT_REGION = "eu-central-1"
DEFAULT_MAX_TOKENS = 20000


def get_bedrock_client():
    session = botocore.session.get_session()
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or session.get_config_variable("region")
        or DEFAULT_REGION
    )

    return session.create_client("bedrock-runtime", region_name=region)


def convert_messages_for_bedrock(messages: list):
    system_text_parts = []
    bedrock_messages = []

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if isinstance(content, list):
            text_content = "\n".join(
                str(part.get("text", part))
                if isinstance(part, dict)
                else str(part)
                for part in content
            )
        else:
            text_content = str(content)

        if role == "system":
            system_text_parts.append(text_content)
            continue

        if role not in {"user", "assistant"}:
            role = "user"

        bedrock_messages.append(
            {
                "role": role,
                "content": [
                    {
                        "text": text_content,
                    }
                ],
            }
        )

    system = [
        {
            "text": "\n\n".join(system_text_parts),
        }
    ] if system_text_parts else None

    return system, bedrock_messages


def extract_text_from_response(response: dict) -> str:
    content_blocks = response.get("output", {}).get("message", {}).get("content", [])
    text_parts = [
        block["text"]
        for block in content_blocks
        if isinstance(block, dict) and "text" in block
    ]

    return "".join(text_parts).strip()


def log_token_usage(response: dict, model_id: str, max_tokens: int):
    usage = response.get("usage", {})
    stop_reason = response.get("stopReason", "UNKNOWN")
    normalized_stop_reason = str(stop_reason).lower()

    if normalized_stop_reason not in {"max_tokens", "max_tokens_reached"}:
        return

    print(
        "[Bedrock token saturation] "
        f"model={model_id} "
        f"input={usage.get('inputTokens', 'NA')} "
        f"output={usage.get('outputTokens', 'NA')} "
        f"total={usage.get('totalTokens', 'NA')} "
        f"max_output={max_tokens} "
        f"stopReason={stop_reason}",
        flush=True,
    )

def query_bedrock(messages: list, model_id: str = "openai.gpt-oss-120b-1:0", temperature: float = 0.1):
    """
    Function to query AWS Bedrock with a list of OpenAI/Groq-style messages.
    
    Args:
        messages (list): The list of messages to send to the model, e.g.
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a short poem about rain."}
            ]
        model_id (str): The model to use for the completion.
        temperature (float): The temperature to use for the completion.
        
    Returns:
        str: The completion from the model.
    """
    system, bedrock_messages = convert_messages_for_bedrock(messages)
    client = get_bedrock_client()

    request = {
        "modelId": model_id,
        "messages": bedrock_messages,
        "inferenceConfig": {
            "maxTokens": DEFAULT_MAX_TOKENS,
            "temperature": temperature,
        },
    }

    if system:
        request["system"] = system

    response = client.converse(**request)
    log_token_usage(response, model_id, DEFAULT_MAX_TOKENS)

    return extract_text_from_response(response)
