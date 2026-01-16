import streamlit as st
import os
import shutil
import platform
import pytesseract
from pdf2image import convert_from_path
import pypdf
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent 
from llama_index.llms.openai import OpenAI

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & MÔI TRƯỜNG
# ==========================================
st.set_page_config(page_title="Universal Financial AI Agent", layout="wide", page_icon="📊")

# Tự động nhận diện đường dẫn Tesseract (Hỗ trợ cả Windows và Linux/Cloud)
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # Trên Linux (Streamlit Cloud), Tesseract thường nằm ở đường dẫn mặc định
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# BẢO MẬT: Lấy API Key từ Streamlit Secrets thay vì dán trực tiếp
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    # Nếu chạy local mà chưa có secrets.toml, bạn có thể nhập tạm ở đây hoặc báo lỗi
    st.error("❌ Thiếu OPENAI_API_KEY! Vui lòng cấu hình trong Secrets.")
    st.stop()

st.title("📊 Universal Financial AI Agent")
st.markdown("---")

# ==========================================
# 2. THANH SIDEBAR (QUẢN LÝ DỰ ÁN)
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    st.write("**Người thực hiện:** Phương Vũ")
    model_choice = st.selectbox("Chọn mô hình AI:", ["gpt-4o", "gpt-3.5-turbo"])
    
    if st.button("🗑️ Làm mới bộ nhớ tạm"):
        if os.path.exists("temp_dir"):
            shutil.rmtree("temp_dir")
            st.success("Đã xóa dữ liệu cũ!")
            st.rerun()

# ==========================================
# 3. PIPELINE OCR ĐA NĂNG (DPI=400, PSM=1)
# ==========================================
uploaded_files = st.file_uploader("Tải lên báo cáo tài chính (PDF):", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if not os.path.exists("temp_dir"):
        os.makedirs("temp_dir")
    
    all_docs = []
    with st.spinner("🤖 AI đang tự động nhận diện bố cục và quét dữ liệu..."):
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp_dir", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Quét trang 8-10 để tập trung vào bảng cân đối kế toán chính (trang 9)
            # Lưu ý: Trên Cloud cần cài poppler thông qua packages.txt
            pages = convert_from_path(file_path, dpi=400, first_page=8, last_page=10)
            
            content = ""
            for i, page in enumerate(pages):
                page_text = pytesseract.image_to_string(page, lang='vie', config='--psm 1')
                content += f"\n--- TRANG {i+1} ---\n{page_text}"

            all_docs.append(Document(text=content, metadata={"file_name": uploaded_file.name}))

    # Khởi tạo Index và Agent
    index = VectorStoreIndex.from_documents(all_docs)
    query_engine = index.as_query_engine(similarity_top_k=10)
    
    finance_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="financial_scanner", 
            description="Truy xuất dữ liệu thô từ báo cáo tài chính."
        )
    )

    llm = OpenAI(model=model_choice, temperature=0)
    agent = OpenAIAgent.from_tools(
        tools=[finance_tool], 
        llm=llm, 
        verbose=True, 
        system_prompt=(
            "Bạn là một chuyên gia phân tích tài chính cao cấp. "
            "Ưu tiên trích xuất số liệu tại cột 'Hợp nhất' (Consolidated). "
            "Hãy đối chiếu kỹ dòng 'Tổng cộng tài sản' tại trang 9 của báo cáo."
        )
    )

    # ==========================================
    # 4. GIAO DIỆN HỘI THOẠI
    # ==========================================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Nhập câu hỏi về báo cáo tài chính..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Đang trích xuất dữ liệu thực tế..."):
                try:
                    response = agent.chat(prompt)
                    st.markdown(response.response)
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
                except Exception as e:
                    st.error(f"Lỗi: {e}")
else:
    st.info("👋 Chào Phương Vũ! Hãy tải lên báo cáo tài chính để AI bắt đầu phân tích.")