# 基于 LlamaIndex 与 DeepSeek 的本地文档 RAG 问答系统

## 项目简介
本项目是一个端到端的本地知识库问答 Demo。旨在解决大语言模型（LLM）在私域数据上的“幻觉”问题。通过引入 RAG（检索增强生成）技术，让 AI 能够基于本地私有文档进行精准问答。

## 技术栈
- **框架**：LlamaIndex
- **大模型**：DeepSeek API (通过 OpenAILike 适配)
- **向量模型**：BAAI/bge-small-zh-v1.5 (本地部署，零成本)
- **环境**：Python 3.11, PyCharm, Virtualenv

## 核心亮点
1. 完整跑通“数据加载 -> 切块 -> 向量化 -> 检索 -> 生成”的 RAG 全链路。
2. 采用本地开源向量模型，实现零额外 Embedding 成本与数据隐私隔离。
3. 成功解决第三方模型 API 兼容性问题，以及国内网络环境下的 SSL 证书报错与模型下载卡顿问题。

## 如何运行
1. 安装依赖：`pip install llama-index llama-index-llms-openai-like llama-index-embeddings-huggingface python-dotenv`
2. 在根目录新建 `.env` 文件，并配置：`DEEPSEEK_API_KEY=你的密钥`
3. 将你的文本放入 `data/` 目录。
4. 运行 `main.py` 即可。