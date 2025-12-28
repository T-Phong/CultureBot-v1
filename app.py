"""
REST API cho Vietnam Heritage RAG System
"""
import uuid
import json
import os
import sys
from datetime import datetime

import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add service directory to sys.path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'service'))
from rewrite import QueryRewriter

app = Flask(__name__)
CORS(app)
# Khởi tạo QueryRewriter (chứa ask_with_context)
rewriter = QueryRewriter()
islog = os.getenv('islog')
metrics_log = []  # Lưu lại các lần đánh giá để dựng biểu đồ

GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)


def _safe_json_parse(text):
    """Parse chuỗi JSON, cố gắng trích block {} đầu tiên nếu có thêm text."""
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def evaluate_answer_llm(question: str, answer: str, history_message):
    """Gọi LLM để chấm điểm mức liên quan, độ chính xác và mức độ lan man."""
    if not GENAI_API_KEY:
        return {
            "status": "skipped",
            "reason": "missing_gemini_api_key",
        }

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        history_text = "\n".join([m.get("content", "") for m in history_message]) if history_message else ""
        prompt = (
            "You are an evaluator for a RAG chatbot."
            " Return JSON with keys: rag_relevance (0-1), answer_accuracy (0-1), hallucination (bool), notes (string)."
            " Evaluate strictly from question and answer (and chat history if provided)."
            " rag_relevance measures how well retrieved context seems relevant to the question."
            " answer_accuracy measures factual correctness and completeness."
            " hallucination is true if the answer includes unrelated, fabricated, or off-topic info."
            f"\nQuestion: {question}\nAnswer: {answer}\nHistory: {history_text}\nReturn JSON only."
        )
        resp = model.generate_content(prompt)
        parsed = _safe_json_parse(resp.text)
        if not parsed:
            raise ValueError("LLM did not return valid JSON")

        rag_rel = float(parsed.get("rag_relevance", 0))
        acc = float(parsed.get("answer_accuracy", 0))
        halluc = bool(parsed.get("hallucination", False))

        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "rag_relevance": max(0.0, min(1.0, rag_rel)),
            "answer_accuracy": max(0.0, min(1.0, acc)),
            "hallucination": halluc,
            "notes": parsed.get("notes", "") or "",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }

@app.route('/v1/chat/completions', methods=['POST'])
def ask_api():
    """
    Main endpoint - Gọi ask_with_context
    
    Request body:
    {
        "question": "Câu hỏi của bạn"
    }
    
    Response:
    {
        "question": "Câu hỏi",
        "answer": "Câu trả lời từ RAG"
    }
    """    
    try:
        data = request.get_json()
        all_messages = data.get("messages", [])
        history_message = all_messages[-6:-1]
        # if islog == "1":
        #     for f in history_message:
        #         print(f)
        

        

        question = all_messages[-1]["content"]
        
        if not question:
            return jsonify({
                "error": "'question' cannot be empty"
            }), 400
        
        # Gọi ask_with_context
        answer = rewriter.ask_with_context(question, history_message)

        # Đánh giá tự động bằng LLM
        # evaluation = evaluate_answer_llm(question, answer, history_message)
        # if evaluation:
        #     metrics_log.append({
        #         "question": question,
        #         "answer": answer,
        #         "evaluation": evaluation,
        #     })
        #     # Giữ kích thước log vừa phải để hiển thị biểu đồ
        #     if len(metrics_log) > 200:
        #         del metrics_log[:-200]

        return jsonify({
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ],
            "evaluation": "evaluation"
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/v1/models', methods=['GET'])
def lstmodel():
    return jsonify({
        "object": "list",
        "data": [
            {"id": "Model-1", "object": "model", "owned_by": "owner"},
            {"id": "Model-2", "object": "model", "owned_by": "owner"}
            ]
    }), 200
        

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Vietnam Heritage RAG API"
    }), 200

@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        "message": "Vietnam Heritage AI REST API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ask": {
                "description": "Ask a question about Vietnamese heritage",
                "body": {
                    "question": "Your question here"
                }
            },
            "GET /health": "Health check endpoint",
            "GET /": "API documentation",
            "GET /lstmodel": "List available models"
        },
        "example": {
            "url": "/ask",
            "method": "POST",
            "body": {
                "question": "Nguyễn Trãi là ai?"
            }
        }
    }), 200


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Trả về log đánh giá để dựng biểu đồ ở frontend."""
    # Tính trung bình nhanh để tiện hiển thị
    rag_scores = [m["evaluation"].get("rag_relevance", 0) for m in metrics_log if m.get("evaluation", {}).get("status") == "ok"]
    acc_scores = [m["evaluation"].get("answer_accuracy", 0) for m in metrics_log if m.get("evaluation", {}).get("status") == "ok"]
    halluc_counts = [m["evaluation"].get("hallucination", False) for m in metrics_log if m.get("evaluation", {}).get("status") == "ok"]

    summary = {
        "total": len(metrics_log),
        "avg_rag_relevance": sum(rag_scores) / len(rag_scores) if rag_scores else 0,
        "avg_answer_accuracy": sum(acc_scores) / len(acc_scores) if acc_scores else 0,
        "hallucination_rate": (sum(1 for h in halluc_counts if h) / len(halluc_counts)) if halluc_counts else 0,
    }

    return jsonify({
        "summary": summary,
        "data": metrics_log,
    }), 200

@app.route('/reset', methods=['POST'])
def reset_history():
    """Reset conversation history"""
    global history
    history = []
    return jsonify({
        "message": "History reset successfully",
        "status": "success"
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print(f"🚀 Vietnam Heritage RAG API")
    print("=" * 60)
    print(f"📍 Server: http://localhost:{port}")
    print(f"📝 Endpoints:")
    print(f"   POST http://localhost:{port}/ask")
    print(f"   GET  http://localhost:{port}/health")
    print(f"   GET  http://localhost:{port}/")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=True)
