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
# 1. CẤU HÌNH GIAO DIỆN & TỰ ĐỘNG NHẬN DIỆN MÔI TRƯỜNG
# ==========================================
st.set_page_config(page_title="Financial AI Agent - VCB Analysis", layout="wide", page_icon="🏦")

# Tự động cấu hình đường dẫn Tesseract dựa trên hệ điều hành
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Trên Streamlit Cloud (Linux), lệnh 'tesseract' sẽ tự động được nhận diện qua packages.txt

# Kiểm tra API Key
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("❌ Thiếu OPENAI_API_KEY! Hãy kiểm tra Settings trên Streamlit Cloud.")
    st.stop()

st.title("🏦 Financial Analysis AI Agent")
st.markdown("---")

# ==========================================
# 2. THANH SIDEBAR (QUẢN LÝ DỰ ÁN)
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình dự án")
    st.write("**Người thực hiện:** Phương Vũ")
    
    model_choice = st.selectbox("Chọn mô hình AI:", ["gpt-4o", "gpt-3.5-turbo"])
    
    if st.button("🗑️ Xóa dữ liệu tạm"):
        if os.path.exists("temp_dir"):
            shutil.rmtree("temp_dir")
            st.success("Đã xóa các file tạm!")
            st.rerun()

# ==========================================
# 3. PIPELINE XỬ LÝ TÀI LIỆU HYBRID (TEXT + OCR)
# ==========================================
uploaded_files = st.file_uploader(
    "Tải lên báo cáo tài chính PDF (Ví dụ: VCB...)", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    if not os.path.exists("temp_dir"):
        os.makedirs("temp_dir")
    
    tools = []
    with st.spinner("🔄 Hệ thống đang trích xuất dữ liệu (Hybrid Mode)..."):
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp_dir", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # --- BƯỚC 1: TRÍCH XUẤT VĂN BẢN KỸ THUẬT SỐ ---
            reader = pypdf.PdfReader(file_path)
            raw_text = ""
            for i in range(min(15, len(reader.pages))): # Quét 15 trang đầu
                raw_text += reader.pages[i].extract_text() or ""
            
            # --- BƯỚC 2: TỰ ĐỘNG BẬT OCR NẾU LÀ FILE QUÉT (SCAN) ---
            # Nếu lượng văn bản lấy được quá ít, hệ thống coi là file ảnh
            if len(raw_text.strip()) < 200:
                st.info(f"🔍 Phát hiện file scan cho {uploaded_file.name}. Đang kích hoạt OCR...")
                # Quét trang quan trọng: trang 5 (quản trị) và trang 9 (tài sản)
                pages = convert_from_path(file_path, first_page=1, last_page=15)
                final_content = ""
                for i, page in enumerate(pages):
                    final_content += f"\n--- TRANG {i+1} ---\n" + pytesseract.image_to_string(page, lang='vie')
            else:
                final_content = raw_text

            # Tạo dữ liệu cho Agent
            documents = [Document(text=final_content)]
            index = VectorStoreIndex.from_documents(documents)
            engine = index.as_query_engine(similarity_top_k=5)
            
            tool = QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name="tool_vcb", 
                    description="Truy xuất dữ liệu tài chính từ báo cáo Vietcombank"
                )
            )
            tools.append(tool)
    
    st.toast("✅ Đã sẵn sàng phân tích dữ liệu!")

    # ==========================================
    # 4. KHỞI TẠO REACT AGENT
    # ==========================================
    llm = OpenAI(model=model_choice, temperature=0)
    agent = ReActAgent.from_tools(
        tools=tools, 
        llm=llm, 
        verbose=True, 
        context="Bạn là chuyên gia tài chính. Hãy tìm kiếm kỹ trong dữ liệu tool_vcb. "
                "Trả lời chính xác số liệu có trong văn bản (ví dụ: Tổng tài sản)."
    )

    # ==========================================
    # 5. GIAO DIỆN HỘI THOẠI
    # ==========================================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hỏi về số liệu tài chính..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Agent đang phân tích bảng biểu..."):
                try:
                    response = agent.chat(prompt)
                    st.markdown(response.response)
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
                except Exception as e:
                    st.error(f"Lỗi hệ thống: {e}")
else:
    st.info("👋 Chào Phương Vũ, hãy tải báo cáo tài chính lên để Agent bắt đầu quét OCR.")
    st.image("https://www.vietcombank.com.vn/images/vcb-logo.png", width=200)