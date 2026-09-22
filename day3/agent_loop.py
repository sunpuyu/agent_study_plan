import os
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
import subprocess

envpath = Path(__file__).parent.parent / ".env"
load_dotenv(envpath)

client = Anthropic(api_key=os.getenv("QWEN_API_KEY"), base_url=os.getenv("QWEN_API_BASE_FOR_ANTHROPIC"))

SYS_PROMPT = f"你是一个Agent，工作在 {os.getcwd()} 目录下，你的任务是根据用户指令，调用LLM API，完成用户任务。"

# tool定义
bash_tool = {
    "name": "bash", 
    "description": "执行bash命令",
    "input_schema":
    {
        "type": "object",
        "properties" : {"command":{"type":"string"}},
        "required" : ["command"]
    }
}

# tool执行函数
def run_bash(command:str):
    r = subprocess.run(command, shell=True, cwd=Path.cwd(),
                           capture_output=True, text=True, errors="replace",
                           timeout=120)
    out = (r.stdout + r.stderr).strip()
    return out[0:50000] if out else "(no output)"

# tool列表，LLM调用时的参数
TOOLS = [bash_tool]

# tool处理函数，LLM返回结果需要调用tool时，需要harness去执行tool调用函数
TOOL_HANDLERS = {"bash": run_bash}


# agent loop —— agent最核心的地方
def agent_loop(messages:list):
    while True:
        response = client.messages.create(
            model="qwen3.8-flash",
            messages=messages,
            system=SYS_PROMPT,
            max_tokens=1024,
            tools=TOOLS,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [ block for block in response.content if block.type == "tool_use"]
        if not tool_calls:
            break

        results = []
        for block in tool_calls:
            tool_handle = TOOL_HANDLERS.get(block.name)
            output = tool_handle(**block.input) if tool_handle else f"unknown: {block.name}"
            results.append({"type":"tool_result", "tool_use_id":block.id, "content":output})

        messages.append({"role": "user", "content": results})

    print(messages)
        

if __name__ == "__main__":
    messages = [{"role": "user", "content": "当前文件的目录是什么"}]
    agent_loop(messages)
