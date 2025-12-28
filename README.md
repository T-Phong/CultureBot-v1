---
title: CultureBot
emoji: 🏃
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
---
# Vietnam Heritage RAG API

REST API cho hệ thống hỏi đáp về di sản văn hóa Việt Nam sử dụng RAG (Retrieval-Augmented Generation).

## Tính năng

- 🔍 Tìm kiếm ngữ cảnh với FAISS
- 🤖 LLM-powered Q&A với Groq API
- 🎯 Reranking với Cross-Encoder
- 📝 Query rewriting thông minh
- 🌐 REST API với Flask

## Cài đặt

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate

# Cài dependencies
pip install -r requirements_rag.txt

# Tạo file .env
echo "GROQ_API_KEY=your_key_here" > .env
```

## Chạy

```bash
python app.py
```

Server sẽ chạy tại `http://localhost:5000`

## API Endpoints

- `POST /ask` - Hỏi đáp
- `GET /health` - Health check
- `GET /` - API documentation

## Ví dụ

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Nguyễn Trãi là ai?"}'


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
