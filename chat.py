import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent # Sử dụng Agent tối ưu cho OpenAI
from llama_index.llms.openai import OpenAI
from halo import Halo

# 1. API Key của bạn
os.environ["OPENAI_API_KEY"] = "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"

# Khởi tạo bộ não AI
llm = OpenAI(model="gpt-3.5-turbo")

print("Đang tải dữ liệu từ bộ nhớ...")
# 2. Load dữ liệu đã lưu
apple_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="apple_docs.DB"))
ms_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="ms_docs.DB"))

# 3. Tạo công cụ tìm kiếm
apple_tool = QueryEngineTool(
    query_engine=apple_index.as_query_engine(),
    metadata=ToolMetadata(name="apple_report", description="Tìm dữ liệu tài chính của Apple năm 2022")
)
ms_tool = QueryEngineTool(
    query_engine=ms_index.as_query_engine(),
    metadata=ToolMetadata(name="ms_report", description="Tìm dữ liệu tài chính của Microsoft năm 2023")
)

# 4. TẠO AI AGENT (Sử dụng OpenAIAgent để tránh lỗi from_tools)
# Đây là cách khởi tạo chuẩn cho phiên bản LlamaIndex mới nhất
agent = OpenAIAgent.from_tools(tools=[apple_tool, ms_tool], llm=llm, verbose=True)

print('\n--- FINANCIAL AI AGENT READY ---')
while True:
    text = input('\nUSER: ')
    if text.upper() == 'QUIT': break
    
    spinner = Halo(text='Thinking...', spinner='dots')
    spinner.start()
    
    # Agent trả lời
    response = agent.chat(text)
    
    spinner.stop()
    print(f"\nAGENT: {response}")