import streamlit as st
import os
import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & BẢO MẬT
# ==========================================
st.set_page_config(page_title="Financial AI Agent", layout="wide", page_icon="🏦")

# Lấy API Key từ Streamlit Secrets (Tránh bị khóa Key do leak lên GitHub)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("❌ Không tìm thấy OPENAI_API_KEY trong mục Secrets của Streamlit!")
    st.info("Hướng dẫn: Vào Settings -> Secrets trên Streamlit Cloud và dán: OPENAI_API_KEY = 'sk-...'")
    st.stop()

st.title("🏦 Financial Analysis AI Agent")
st.markdown("---")

# ==========================================
# 2. THANH SIDEBAR (GIỚI THIỆU & QUẢN LÝ)
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình dự án")
    st.write("**Người thực hiện:** Phương Vũ")
    st.write("**Công nghệ:** LlamaIndex + ReAct Agent")
    
    if st.button("🗑️ Xóa dữ liệu tạm"):
        if os.path.exists("temp_dir"):
            shutil.rmtree("temp_dir")
            st.success("Đã xóa các file tạm!")
            st.rerun()

# ==========================================
# 3. XỬ LÝ UPLOAD FILE & TẠO INDEX
# ==========================================
uploaded_files = st.file_uploader(
    "Tải lên báo cáo tài chính PDF (Ví dụ: VCB, Apple, Microsoft...)", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    # Tạo thư mục tạm an toàn
    if not os.path.exists("temp_dir"):
        os.makedirs("temp_dir")
    
    tools = []
    with st.spinner("🔄 Đang phân tích dữ liệu PDF..."):
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp_dir", uploaded_file.name)
            
            # Lưu file vào thư mục tạm trên server
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Đọc và lập chỉ mục dữ liệu (RAG)
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            index = VectorStoreIndex.from_documents(documents)
            
            # Đóng gói Index thành Công cụ (Tool) cho Agent
            engine = index.as_query_engine(similarity_top_k=3)
            tool = QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name=f"tool_{uploaded_file.name.replace('.pdf', '').replace(' ', '_').replace('-', '_')}",
                    description=f"Truy xuất số liệu tài chính chi tiết từ file: {uploaded_file.name}"
                )
            )
            tools.append(tool)
    
    st.toast(f"✅ Đã nạp thành công {len(uploaded_files)} tài liệu!")

    # ==========================================
    # 4. KHỞI TẠO AI AGENT (REMARKABLE REASONING)
    # ==========================================
    # Sử dụng mô hình GPT-3.5-Turbo tiết kiệm và hiệu quả
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

    # ==========================================
    # 5. GIAO DIỆN CHAT (CHAT INTERFACE)
    # ==========================================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử hội thoại
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Xử lý câu hỏi mới
    if prompt := st.chat_input("Nhập câu hỏi về báo cáo tài chính tại đây..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 AI Agent đang phân tích..."):
                try:
                    # Agent thực hiện suy luận (Thought) và hành động (Action)
                    response = agent.chat(prompt)
                    st.markdown(response.response)
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")

else:
    st.info("👋 Chào Phương Vũ! Hãy upload báo cáo tài chính của Vietcombank để bắt đầu phân tích.")
    st.image("https://www.vietcombank.com.vn/images/vcb-logo.png", width=200)