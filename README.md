# AI Develop Learn

这是一个用于学习大模型 API 调用和 AI 应用开发演进的练习项目。

## 学习主线

本项目会按“大模型应用能力演进”的方式推进：

API 调用 → Prompt 工程 → 结构化输出 → 工具调用 → RAG → 记忆 → Agent → 评测 → 服务化。

完整路线见：[大模型应用演进学习路线](docs/LLM_EVOLUTION_ROADMAP.md)。

## 本地环境

项目使用 Python 3.10 和 OpenAI SDK，当前示例通过 NVIDIA NIM 的 OpenAI 兼容接口调用模型。

首次使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

然后打开 `.env`，把 `NVIDIA_API_KEY` 改成你的真实 Key。

## 运行示例

```bash
source .venv/bin/activate
python Step1/llm_chat.py "你好，介绍一下你自己"
```

默认会调用 `chat_qwen`。如果想尝试其他模型，可以在 `Step1/llm_chat.py` 里把入口处的函数改成 `chat_ds`、`chat_glm`、`chat_kimi` 或 `chat_minimax`。
