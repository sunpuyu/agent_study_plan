import os, json, time
from openai import OpenAI 
from pydantic import BaseModel, ValidationError

from dotenv import load_dotenv
from pathlib import Path

current_file_dir = Path(__file__).resolve().parent
env_file = current_file_dir.parent / ".env"

load_dotenv(env_file)

qwen_client = OpenAI(api_key=os.getenv("QWEN_API_KEY"), base_url=os.getenv("QWEN_API_BASE"))
glm_client = OpenAI(api_key=os.getenv("GLM_API_KEY"), base_url=os.getenv("GLM_API_BASE"))
deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_API_BASE"))

client_dict = {
    "qwen3.8-flash": qwen_client,
    "glm-5.3-flash": glm_client,
    "deepseek-flash": deepseek_client
}

class Review(BaseModel):
    sentiment: str
    score: int
    tags: list[str]

def call(client, mode, text, json_model=False):
    kwargs = {"response_format": {"type": "json_object"}} if json_model else {}
    r = client.chat.completions.create(
        model=mode,
        messages=[{"role": "user", "content": text}],
        **kwargs
    )
    return r.choices[0].message.content, r.usage

if __name__ == "__main__":
    for model_name, client in client_dict.items():
        print(model_name)
        text = "请告诉我，我这个问题是几个字，带上你的回答总共是几个字？每个标点符号算一个字，每一位数字算一个字。"
        response, usage = call(client, model_name, text)
        print(response)
        #print(usage)