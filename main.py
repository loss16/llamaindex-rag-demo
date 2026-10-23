import os
#RAG：本地模型将文档切块并向量化建索引；提问向量化后检索相关top-k块，连同问题交给大模型生成答案。

# 🌟 国内镜像加速下载，以后遇到包去hf-mirror下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv#保密员，去env拿到我的api_key
#切碎文字整理成文字目录，读文件，配置用哪个大模型
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# ⬇️ 注意这里：导入的是 OpenAILike，而不是 OpenAI，翻译官：把ds伪装成openai的格式
from llama_index.llms.openai_like import OpenAILike
#向量化
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 读取 .env 里的 API Key
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 2. 配置大模型（使用 OpenAILike 适配 DeepSeek）
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=api_key,
    is_chat_model=True,          # 声明这是聊天模型
    context_window=128000,       # 声明模型的上下文窗口大小
)

# 3. 配置向量模型（本地免费模型）
print("正在加载向量模型，请稍候...")
#HuggingFaceEmbedding(...) 是 LlamaIndex 官方写好的一个“工具类”。
# 它的作用是：帮你连上 HuggingFace，把你指定的 bge 模型下载下来，并打包成 LlamaIndex 能直接调用的格式
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 4. 读取 data 文件夹里的文档
print("正在读取文档并建立索引...")
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 5. 提问！
query_engine = index.as_query_engine()
print("正在向 AI 提问...")
resp = query_engine.query("用一句话总结这份文档讲了什么")

print("\n==== AI 的回答 ====\n", resp)