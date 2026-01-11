import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.llms.openai import OpenAI

# 1. Dán API Key của bạn vào đây
os.environ["OPENAI_API_KEY"] = "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"

# Load tài liệu
apple_docs = SimpleDirectoryReader(input_files=["apple.pdf"]).load_data()
ms_docs = SimpleDirectoryReader(input_files=["microsoft.pdf"]).load_data()

# Tạo và lưu Index cho Apple
apple_index = VectorStoreIndex.from_documents(apple_docs)
apple_index.storage_context.persist(persist_dir="apple_docs.DB")

# Tạo và lưu Index cho Microsoft
ms_index = VectorStoreIndex.from_documents(ms_docs)
ms_index.storage_context.persist(persist_dir="ms_docs.DB")

print("Đã tạo xong dữ liệu thành công!")