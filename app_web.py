import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# 1. Cấu hình giao diện Web
st.set_page_config(page_title="AI Financial Agent", layout="wide")
st.title("🏦 Financial Analysis AI Agent")
st.markdown("Upload các báo cáo tài chính (PDF) và yêu cầu AI phân tích, so sánh.")

# 2. Nhập API Key (Có thể dán cứng hoặc nhập trên web)
os.environ["OPENAI_API_KEY"] = "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"

# 3. Khu vực Upload File
uploaded_files = st.file_uploader("Chọn các file báo cáo tài chính (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Tạo thư mục tạm để lưu file upload
    if not os.path.exists("temp_dir"):
        os.makedirs("temp_dir")
    
    tools = []
    with st.spinner("Đang đọc và phân tích dữ liệu..."):
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp_dir", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Đọc file và tạo Index ngay lập tức (không cần embed.py rời)
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            index = VectorStoreIndex.from_documents(documents)
            
            # Tạo công cụ cho từng file
            engine = index.as_query_engine()
            tool = QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name=f"tool_{uploaded_file.name.replace('.pdf', '').replace(' ', '_')}",
                    description=f"Dữ liệu từ file {uploaded_file.name}"
                )
            )
            tools.append(tool)
    
    st.success(f"Đã sẵn sàng phân tích {len(uploaded_files)} tài liệu!")

    # 4. Khởi tạo Agent
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

    # 5. Khu vực Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ô nhập câu hỏi
    if prompt := st.chat_input("Bạn muốn hỏi gì về các báo cáo này?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent đang suy nghĩ..."):
                response = agent.chat(prompt)
                st.markdown(response)
                # Hiển thị Thought của Agent lên console để bạn theo dõi
                print(f"Agent Thought: {response}") 
        
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
else:
    st.info("Vui lòng upload ít nhất một file PDF để bắt đầu.")