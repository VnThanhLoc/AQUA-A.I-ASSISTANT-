import base64
from email.mime import message
import importlib
import json
from logging import config
from logging import config
import os
from typing import Optional

import pandas as pd
import streamlit as st


try:
    dotenv = importlib.import_module("dotenv")
    load_dotenv = dotenv.load_dotenv
except Exception:
    # Provide a noop fallback if python-dotenv isn't available in the environment
    def load_dotenv(path: str = ".env") -> None:
        return None

try:
    genai = importlib.import_module("google.genai")
    types = importlib.import_module("google.genai.types")
except Exception:
    genai = None
    types = None

MODEL_NAME = "gemini-2.5-flash"

def get_streamlit_secret(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_data
def load_config() -> dict:
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_menu() -> pd.DataFrame:
    return pd.read_csv(
        "danh_sach_100_loai_ca_canh_co_gia.csv",
        encoding="utf-8-sig"
    )


def load_api_key() -> Optional[str]:
    load_dotenv()
    return (
        os.getenv("GEMINI_API_KEY")
        or get_streamlit_secret("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or get_streamlit_secret("GOOGLE_API_KEY")
    )


@st.cache_resource
def create_client(api_key: Optional[str]):
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


def build_menu_context(menu_df: pd.DataFrame) -> str:
    def clean_text(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def add_detail(item_text: str, label: str, value) -> str:
        value = clean_text(value)
        if not value:
            return item_text

        suffix = "" if value.endswith((".", "!", "?")) else "."
        return f"{item_text} {label}: {value}{suffix}"

    progress_lines = []
    for _, row in menu_df.iterrows():
        item_text = f"- {clean_text(row['Fish_Name'])}: {clean_text(row['Feeding_Habit'])}."
        item_text = add_detail(item_text, "Tính cách", row.get("Character", ""))
        item_text = add_detail(item_text, "Môi trường phù hợp", row.get("Environment", ""))
        progress_lines.append(item_text)
    return "\n".join(progress_lines)


def build_system_instruction(config: dict) -> str:
    functions = ", ".join(config.get("functions", []))
    shop_name = config.get("shop_name", "AquaShop")
    shop_address = config.get("shop_address", "Tổ 311, đường 112 Trần Hưng Đạo, Phường 67, TP.Vũng Tàu, Việt Nam")
    out_of_scope_message = config.get("out_of_scope_message", "Xin liên hệ nhân viên cửa hàng để được trợ giúp.")
    return "\n".join([
        f"Bạn tên là AquaBot, một trợ lý AI hỗ trợ khách hàng của cửa hàng {shop_name}.",
        f"Địa chỉ cửa hàng: {shop_address}.",
        f"Các chức năng được hỗ trợ: {functions}.",
        "",
        "Nguyên tắc trả lời:",
        f"1. Trả lời chi tiết, thân thiện, dễ hiểu và chính xác dựa trên dữ liệu sản phẩm đã cho.",
        f"2. Chỉ trả lời các câu hỏi liên quan đến cửa hàng, sản phẩm trong dữ liệu được cung cấp.",
        f"3. Nếu câu hỏi nằm ngoài phạm vi hỗ trợ, trả lời đúng câu sau: \"{out_of_scope_message}\"",
        f"4. Không bịa thông tin nếu dữ liệu của cửa hàng không có câu trả lời.",
    ])


def build_history_text(messages: list[dict], max_messages: int = 6) -> str:
    
    recent_messages = messages[-max_messages:]
    lines = []
    for message in recent_messages:
        role = "Khách hàng" if message["role"] == "user" else "AquaBot"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def mock_response(prompt: str, products_df: pd.DataFrame, config: dict) -> str:
    prompt_lower = prompt.lower()

    # Hỏi về sản phẩm chung
    if any(keyword in prompt_lower for keyword in [
        "sản phẩm", "san pham", "cây", "cay",
        "cá", "ca", "phân nền", "phan nen",
        "lọc", "loc", "đèn", "den", "thủy sinh", "thuy sinh"
    ]):
        # Kiểm tra xem có hỏi cụ thể loại cá nào không, nếu không mới trả lời chung
        has_fish_name = False
        for _, row in products_df.iterrows():
            if str(row['Fish_Name']).lower() in prompt_lower:
                has_fish_name = True
                break
        
        if not has_fish_name and any(k in prompt_lower for k in ["kích thước", "size", "giá", "ăn gì", "tính cách", "môi trường"]):
            pass # Để trôi xuống kiểm tra các thuộc tính cụ thể
        elif not has_fish_name:
            return (
                "Ở đây chúng tôi có khoảng 100 loại cá cảnh khác nhau, cùng với các loại cây thủy sinh, phân nền và phụ kiện lọc nước, ngoài ra chúng tôi cũng có thể tư vấn thêm cho bạn về 1 số loài cá cảnh, môi trường phù hợp,... Bạn có thể hỏi tôi về các loại cá cảnh, cây thủy sinh, phân nền hoặc phụ kiện lọc nước mà bạn quan tâm nhé!")

    # 1. Kiểm tra kích thước
    size_keywords = ["kích thước", "kich thuoc", "size", "lớn cỡ nào", "lon co nao", "dài bao nhiêu", "dai bao nhieu"]
    if any(keyword in prompt_lower for keyword in size_keywords):
        for _, row in products_df.iterrows():
            fish_name = str(row['Fish_Name'])      
            if fish_name.lower() in prompt_lower:
                # Đề phòng trường hợp cột size không có dữ liệu
                size_val = row.get('Min-Max_Size_cm', 'Chưa cập nhật')
                return f"Kích thước tối đa của **{fish_name}** là: {size_val} cm."

    # 2. Kiểm tra tính cách
    characters_keywords = ["hiền","hien", "dễ nuôi", "de nuoi", "hung dữ", "hung du", "khó nuôi", "kho nuoi","dễ thương", "de thuong","săn mồi", "san moi", "lãnh thổ cao", "lanh tho cao","năng động", "nang dong", "tính cách", "tinh cach"]
    if any(keyword in prompt_lower for keyword in characters_keywords):
        for _, row in products_df.iterrows():
            fish_name = str(row['Fish_Name'])      
            if fish_name.lower() in prompt_lower:
                return f"Đặc điểm tính cách của **{fish_name}**: {row['Character']}."

    # 3. Kiểm tra môi trường
    environment_keywords = ["môi trường", "moi truong", "độ ph", "do ph", "độ cứng nước", "do cung nuoc", "đèn chiếu sáng", "den chieu sang"]
    if any(keyword in prompt_lower for keyword in environment_keywords):
        for _, row in products_df.iterrows():
            fish_name = str(row['Fish_Name'])      
            if fish_name.lower() in prompt_lower:
                return f"Môi trường phù hợp cho **{fish_name}**: {row['Environment']}, có độ pH nước là {row['Water_pH']}."
    # 4. Kiểm tra nhiệt độ nước
    temperature_keywords = ["nhiệt độ", "nhiet do", "nhiệt độ nước", "nhiet do nuoc", "độ ấm", "do am", "độ lạnh", "do lanh"]
    if any(keyword in prompt_lower for keyword in temperature_keywords):
        for _, row in products_df.iterrows():
            fish_name = str(row['Fish_Name'])      
            if fish_name.lower() in prompt_lower:
                return f"Nhiệt độ nước phù hợp cho **{fish_name}**: {row['Water_Temperature']} °C."

    # 5. Kiểm tra chế độ ăn
    feeding_habit_keywords = ["chế độ ăn", "che do an", "ăn gì", "an gi", "thức ăn", "thuc an", "ăn bao nhiêu lần", "an bao nhieu lan", "đồ ăn", "do an"]
    if any(keyword in prompt_lower for keyword in feeding_habit_keywords):
        for _, row in products_df.iterrows():
            fish_name = str(row['Fish_Name'])      
            if fish_name.lower() in prompt_lower:
                return f"Chế độ ăn của **{fish_name}**: {row['Feeding_Habit']}."

    # 6. Kiểm tra giá cả
    price_keywords = ["giá", "gia", "bao nhiêu", "bao nhieu", "cost", "price","bao nhiêu tiền", "bao nhieu tien"]
    if any(keyword in prompt_lower for keyword in price_keywords):
        for _, row in products_df.iterrows():
            fish_name = str(row['Fish_Name'])      
            if fish_name.lower() in prompt_lower:
                price_val = row.get('Price_VND', 'Liên hệ')
                return f"Giá của **{fish_name}** là: {price_val} VND."
    # 7. Khách hàng đặt cọc 
    deposit_keywords = ["đặt cọc", "dat coc", "mua trước", "mua truoc", "pre-order", "preorder", "đặt trước", "dat truoc", 'đặt cọc trước', "dat coc truoc","đặt"]
    if any(keyword in prompt_lower for keyword in deposit_keywords):
        for _, row in products_df.iterrows():
            fish_name = str(row['Fish_Name'])      
            if fish_name.lower() in prompt_lower:
                return f"Bạn có thể đặt cọc trước cho **{fish_name}**. Vui lòng liên hệ cửa hàng qua số {config.get('hotline')} để biết chi tiết về quy trình đặt cọc và thanh toán."

    # Dự phòng nếu hỏi chung chung về cá mà ko rõ loại cá
    if any(k in prompt_lower for k in ["cá", "ca"]) and any(k in prompt_lower for k in ["kích thước", "giá", "tính cách", "ăn gì", "môi trường", "nhiệt độ", "chế độ ăn", "đặt cọc", "kich thuoc", "gia", "tinh cach", "an gi", "moi truong", "nhiet do", "che do an", "dat coc"]):
        return "Bạn muốn hỏi thông tin của loài cá nào ạ? (Ví dụ: Cá lóc cảnh, Cá bảy màu, Cá sặc gấm...)"

    # Hỏi địa chỉ cửa hàng
    if any(keyword in prompt_lower for keyword in ["địa chỉ", "dia chi", "ở đâu", "o dau", "address"]):
        return f"Cửa hàng {config.get('shop_name')} nằm tại {config.get('shop_address')}."

    # Chào hỏi
    if any(keyword in prompt_lower for keyword in ["xin chào", "hello", "hi", "chào", "chao", "hey", "alo"]):
        return (
            f"Chào bạn! Tôi là trợ lý của {config.get('shop_name', 'cửa hàng')}. "
            "Bạn có thể hỏi tôi về cây thủy sinh, cá cảnh, hồ thủy sinh và các phụ kiện nhé."
        )

    # Hỏi giờ mở cửa
    if any(keyword in prompt_lower for keyword in ["giờ mở cửa", "gio mo cua", "mấy giờ", "may gio", "open"]):
        return f"{config.get('shop_name', 'Cửa hàng')} mở cửa từ {config.get('opening_hours', '08:00 - 21:00')}."

    return config.get(
        "out_of_scope_message",
        "Xin lỗi, tôi chưa hiểu câu hỏi của bạn. Bạn có thể hỏi về sản phẩm thủy sinh, cá cảnh hoặc thông tin cửa hàng."
    )


def ask_bot(prompt: str, messages: list[dict], client, menu_df: pd.DataFrame, config: dict, use_mock: bool) -> str:
    if use_mock or client is None or types is None:
        return mock_response(prompt, menu_df, config)

    system_instruction = build_system_instruction(config)
    menu_context = build_menu_context(menu_df)
    history_text = build_history_text(messages)
    user_content = (
        "Dữ liệu cửa hàng:\n" + menu_context +
        "\n\nLịch sử trò chuyện gần đây:\n" + history_text +
        "\n\nCâu hỏi mới của khách hàng:\n" + prompt
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
            ),
        )
        return response.text or "Xin lỗi, tôi chưa tạo được câu trả lời phù hợp."
    except Exception as exc:
        return (
            "Xin lỗi, hiện tại tôi chưa kết nối được với Gemini API. "
            "Bạn có thể kiểm tra lại API key, kết nối mạng hoặc dùng chế độ Mock để tiếp tục demo.\n\n"
            f"Chi tiết lỗi: `{exc}`"
        )


def restaurant_chatbot():
    # SỬA LỖI: Chỉ gọi set_page_config DUY NHẤT 1 lần ở đầu hàm
    st.set_page_config(
        page_title="AquaBot - Aquatic Assistant",
        page_icon="./ChatGPT Image 16_28_55 13 thg 6, 2026.png",
        layout="wide"
    )

    config = load_config()
    menu_df = load_menu()
    api_key = load_api_key()
    client = create_client(api_key)

    st.markdown("""
        <div class="main-title">
        🐠 Aqua A.I Assistant
        </div>

        <div class="sub-title">
        Khám phá thế giới cá cảnh và thủy sinh cùng AI
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Cấu hình")
        st.caption("Streamlit Chat UI + Gemini API")

        has_sdk = genai is not None

        st.write(
            "Google GenAI SDK:",
            "✅ Đã sẵn sàng" if has_sdk else "⚠️ Chưa cài google-genai"
        )

        st.write(
            "API key:",
            "✅ Đã tìm thấy" if api_key else "⚠️ Chưa có"
        )

        use_mock = st.toggle(
            "Dùng Mock Response",
            value=not bool(api_key and has_sdk),
            help="Bật nếu chưa có API key."
        )

        st.markdown("---")

        st.markdown("### 💡 Gợi ý câu hỏi")
        st.markdown("• Địa chỉ của cửa hàng ở đâu?")
        st.markdown("• Cửa hàng có những sản phẩm gì?")
        st.markdown("• Các loại cá cảnh có những đặc điểm gì?")
        

    # Load video background
    with open("./shipwreck-moewalls-com.mp4", "rb") as f:
            video_bytes = f.read()

    st.markdown(
        f"""
            <style>

            #bg-video {{
                position: fixed;
                right: 0;
                bottom: 0;
                min-width: 100%;
                min-height: 100%;
                object-fit: cover;
                z-index: -1;
                opacity: 0.5;
            }}

            .stApp {{
                background: transparent !important;
            }}

            /* Header */
            .main-title{{
                text-align:center;
                color:white;
                font-size:42px;
                font-weight:700;
                margin-bottom:5px;
                text-shadow: 2px 2px 10px black;
            }}

            .sub-title{{
                text-align:center;
                color:#e2e8f0;
                margin-bottom:25px;
                text-shadow: 2px 2px 10px black;
            }}


            </style>

            <video autoplay muted loop id="bg-video">
                <source src="data:video/mp4;base64,{base64.b64encode(video_bytes).decode("utf-8")}" type="video/mp4">
            </video>
            
            """,
            unsafe_allow_html=True)
        
    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = [
            {"role": "assistant", "content": config.get("initial_bot_message", "Xin chào! Bạn cần hỗ trợ gì?")}
        ]


   # Khởi tạo lịch sử chat
    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = [
            {
                "role": "assistant",
                "content": config.get(
                    "initial_bot_message",
                    "Xin chào! Bạn cần hỗ trợ gì?"
                )
            }
        ]

    # Hiển thị lịch sử chat
    for message in st.session_state.conversation_log:

        # USER
        if message["role"] == "user":
            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:flex-end;
                    margin:12px 0;
                ">
                    <div style="
                        background:rgba(37,99,235,0.9);
                        color:white;
                        padding:12px 18px;
                        border-radius:20px 20px 5px 20px;
                        max-width:100%;
                        backdrop-filter:blur(10px);
                        box-shadow:0 4px 15px rgba(0,0,0,0.3);
                    ">
                        <span style="font-size:25px; color:white;">🤵</span> {message["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ASSISTANT
        else:
            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:flex-start;
                    margin:12px 0;
                ">
                    <div style="
                        background:rgba(30,41,59,0.85);
                        color:white;
                        padding:12px 18px;
                        border-radius:20px 20px 20px 5px;
                        max-width:100%;
                        backdrop-filter:blur(10px);
                        box-shadow:0 4px 15px rgba(0,0,0,0.3);
                    ">
                    <span style="font-size:25px;">🐟</span> {message["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True)
            

    # Xử lý input chat của user
    prompt = st.chat_input("Nhập yêu cầu của bạn tại đây...")
    if prompt:
        # 1. Append câu hỏi của user
        st.session_state.conversation_log.append({"role": "user", "content": prompt})
        
        # 2. Sinh câu trả lời từ Bot (hiển thị spinner trong khi chờ)
        with st.spinner("🐠 Aqua AI đang suy nghĩ..."):
            bot_reply = ask_bot(
                prompt=prompt,
                messages=st.session_state.conversation_log,
                client=client,
                menu_df=menu_df,
                config=config,
                use_mock=use_mock,
            )

        # 3. Append câu trả lời của Bot (dọn ký tự markdown thừa)
        bot_reply = (
            bot_reply
            .replace("**", "")
            .replace("* ", "")
        )
        st.session_state.conversation_log.append({"role": "assistant", "content": bot_reply})
        
        # 4. Kích hoạt render lại giao diện ngay lập tức để đồng bộ tin nhắn mới
        st.rerun()

    with st.expander("Checklist bảo mật API key"):
        st.markdown(
            """
- Không dán API key trực tiếp vào `app.py`.
- Local: lưu key trong `.env` với tên `GEMINI_API_KEY`.
- Deploy Streamlit Cloud: lưu key trong **Secrets**.
- Không commit file `.env` lên GitHub.
- Nếu thiếu key, app vẫn có thể chạy bằng Mock Response để demo giao diện.
"""
        )


if __name__ == "__main__":
    restaurant_chatbot()