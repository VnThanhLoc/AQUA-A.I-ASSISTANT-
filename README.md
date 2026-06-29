# LLM Sample Hotfix - CSI Bài 7 & Bài 8

Sample này dùng cho hotfix Buổi 7-8 của Computer Scientist Intensive.

## Điểm cập nhật chính

- Chuyển từ SDK cũ `google.generativeai` sang Google GenAI SDK mới: `google-genai`.
- Chuẩn hóa API key bằng biến môi trường `GEMINI_API_KEY`.
- Model mặc định: `gemini-2.5-flash`.
- Thêm chế độ Mock Response để học viên vẫn làm được giao diện nếu chưa tạo được API key.
- Không hard-code API key trong source code.

## Chạy local

```bash
pip install -r requirements.txt
cp .env.example .env
```

Mở file `.env` và thay:

```bash
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

Chạy app:

```bash
streamlit run Fish_shop.py
```

## Nếu chưa có API key

App vẫn chạy được bằng cách bật **Dùng Mock Response** ở sidebar. Cách này dùng để học viên hoàn thiện UI, chat history và luồng xử lý trước, sau đó thay bằng API thật khi có key.

## Deploy Streamlit Cloud

Không upload file `.env`. Vào **App settings > Secrets** và thêm:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

## File trong sample

- `Fish_shop.py`: ứng dụng Streamlit chatbot.
- `ls7_llm.ipynb`: notebook mẫu/GV cho Buổi 7.
- `students_ls7_llm.ipynb`: notebook học viên cho Buổi 7.
- `config.json`: cấu hình chatbot.
- `./danh_sach_100_loai_ca_canh_co_gia.csv.csv`: dữ liệu menu nhà hàng.
- `.env.example`: mẫu biến môi trường.
- `requirements.txt`: thư viện cần cài.
