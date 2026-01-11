import streamlit as st
import os
import shutil
import platform
import pytesseract
from pdf2image import convert_from_path
import pypdf
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & MÔI TRƯỜNG
# ==========================================
st.set_page_config(page_title="Universal Financial AI Agent", layout="wide", page_icon="📊")

# Tự động nhận diện đường dẫn Tesseract
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Lấy API Key từ Secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("❌ Thiếu OPENAI_API_KEY trong cấu hình Secrets!")
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
            
            # Quét trang 1-20 để bao quát toàn bộ các bảng cân đối kế toán và kết quả kinh doanh
            pages = convert_from_path(file_path, dpi=400, first_page=1, last_page=20)
            
            content = ""
            for i, page in enumerate(pages):
                # Sử dụng PSM 1 để tự động phân tích bố cục trang bất kỳ
                page_text = pytesseract.image_to_string(page, lang='vie', config='--psm 1')
                content += f"\n--- TRANG {i+1} ---\n{page_text}"

            all_docs.append(Document(text=content, metadata={"file_name": uploaded_file.name}))

    # Khởi tạo Index từ dữ liệu quét thực tế
    index = VectorStoreIndex.from_documents(all_docs)
    query_engine = index.as_query_engine(similarity_top_k=10)
    
    # Tool tổng quát không còn nhắc đến VCB
    finance_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="financial_scanner", 
            description="Truy xuất dữ liệu thô từ báo cáo tài chính được tải lên."
        )
    )

    # ==========================================
    # 4. KHỞI TẠO AI AGENT (CẬP NHẬT PHIÊN BẢN MỚI)
    # ==========================================
    llm = OpenAI(model=model_choice, temperature=0)
    
    # Cách khởi tạo an toàn cho mọi phiên bản LlamaIndex
    from llama_index.core.agent import ReActAgent
    
    agent = ReActAgent.from_tools(
        tools=[finance_tool], 
        llm=llm, 
        verbose=True,
        max_iterations=10 # Thêm giới hạn để tránh Agent chạy vòng lặp vô tận
    )
    
    # Thiết lập Context cho Agent (phần này tách riêng để tránh lỗi khởi tạo)
    agent.update_prompts({"agent_worker:system_prompt": (
        "Bạn là một chuyên gia phân tích tài chính cao cấp. "
        "Nhiệm vụ của bạn là đọc dữ liệu từ công cụ financial_scanner và trả lời câu hỏi. "
        "Hãy luôn kiểm tra tiêu đề trang và tiêu đề bảng biểu để xác định đúng loại báo cáo. "
        "Khi trả lời về số liệu, hãy chỉ rõ số liệu đó thuộc về đơn vị nào và thời điểm nào."
    )})

    # ==========================================
    # 5. GIAO DIỆN HỘI THOẠI
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
    st.info("👋 Chào Phương Vũ! Hãy tải lên bất kỳ báo cáo tài chính nào (PDF scan hoặc text) để bắt đầu phân tích.")
    # Sử dụng ảnh minh họa tổng quát
    st.markdown("### 🏦 Hệ thống hỗ trợ đọc:")
    st.markdown("- Báo cáo tài chính Ngân hàng\n- Báo cáo thường niên Doanh nghiệp\n- Các tài liệu dạng ảnh quét (OCR)")