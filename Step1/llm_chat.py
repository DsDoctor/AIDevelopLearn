from openai import OpenAI
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ============ 常量管理 ============
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# ============ 公共客户端 ============
def get_client():
    if not NVIDIA_API_KEY or NVIDIA_API_KEY.startswith("替换成"):
        raise RuntimeError("请先在 .env 文件中配置 NVIDIA_API_KEY")
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

# ============ GLM 思考输出颜色控制 ============
_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR = "\033[0m" if _USE_COLOR else ""


def chat_ds(message):
    """DeepSeek V4 Flash"""
    client = get_client()
    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-v4-flash",
        messages=[{"role": "user", "content": message}],
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        extra_body={"chat_template_kwargs": {"thinking": False, "reasoning_effort": "high"}},
        stream=True
    )
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        reasoning = getattr(chunk.choices[0].delta, "reasoning", None) or getattr(chunk.choices[0].delta, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="")
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


def chat_glm(message):
    """GLM 4.7"""
    client = get_client()
    completion = client.chat.completions.create(
        model="z-ai/glm4.7",
        messages=[{"role": "user", "content": message}],
        temperature=1,
        top_p=1,
        max_tokens=16384,
        extra_body={"chat_template_kwargs": {"enable_thinking": False, "clear_thinking": False}},
        stream=True
    )
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            print(f"{_REASONING_COLOR}{reasoning}{_RESET_COLOR}", end="")
        if getattr(delta, "content", None) is not None:
            print(delta.content, end="")


def chat_kimi(message):
    """Kimi K2.5"""
    client = get_client()
    completion = client.chat.completions.create(
        model="moonshotai/kimi-k2.5",
        messages=[{"role": "user", "content": message}],
        max_tokens=16384,
        temperature=1.00,
        top_p=1.00,
        extra_body={"chat_template_kwargs": {"thinking": False}},
        stream=True
    )
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


def chat_minimax(message):
    """MiniMax M2.7"""
    client = get_client()
    completion = client.chat.completions.create(
        model="minimaxai/minimax-m2.7",
        messages=[{"role": "user", "content": message}],
        temperature=1,
        top_p=0.95,
        max_tokens=8192,
        stream=True
    )
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


def chat_qwen(message):
    """Qwen3 Coder 480B"""
    client = get_client()
    completion = client.chat.completions.create(
        model="qwen/qwen3-coder-480b-a35b-instruct",
        messages=[{"role": "user", "content": message}],
        temperature=0.7,
        top_p=0.8,
        max_tokens=4096,
        stream=True
    )
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


if __name__ == '__main__':
    question = " ".join(sys.argv[1:]) or "测试下连接"
    chat_qwen(question)
