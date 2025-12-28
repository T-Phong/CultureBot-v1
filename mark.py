import google.generativeai as genai
import json
import pandas as pd

# Cấu hình API Key (Lấy tại aistudio.google.com)
genai.configure(api_key="AIzaSyBpGsHI4GR_APcMVSIcNRDt-rupEOOUDJc")
print("Danh sách các model bạn có thể dùng:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")
def evaluate_rag_pipeline(question, retrieved_contexts, model_answer, ground_truth):
    """
    Hàm này dùng Gemini để chấm điểm 3 khía cạnh của RAG.
    """
    
    # Gom các đoạn văn tìm được thành 1 chuỗi để Gemini đọc
    context_text = "\n---\n".join(retrieved_contexts)
    
    prompt = f"""
    Bạn là chuyên gia đánh giá hệ thống RAG (Retrieval-Augmented Generation).
    Hãy phân tích và chấm điểm các yếu tố sau dựa trên dữ liệu cung cấp:

    1. INPUT DATA:
    - Câu hỏi (Query): "{question}"
    - Ngữ cảnh tìm được (Retrieved Contexts): 
    "{context_text}"
    - Câu trả lời của Bot (Generated Answer): "{model_answer}"
    - Đáp án chuẩn (Ground Truth): "{ground_truth}"

    2. NHIỆM VỤ ĐÁNH GIÁ (Thang điểm 0 hoặc 1):
    
    A. Context_Relevance (0/1): 
       - 1: Nếu trong 'Ngữ cảnh tìm được' CÓ chứa thông tin cần thiết để trả lời câu hỏi.
       - 0: Nếu ngữ cảnh không liên quan hoặc thiếu thông tin quan trọng.
       
    B. Faithfulness (0/1):
       - 1: Nếu 'Câu trả lời của Bot' chỉ dựa trên thông tin trong 'Ngữ cảnh tìm được', không bịa đặt.
       - 0: Nếu Bot đưa ra thông tin không có trong ngữ cảnh (hallucination) hoặc mâu thuẫn với ngữ cảnh.
       
    C. Answer_Correctness (0/1):
       - 1: Nếu 'Câu trả lời của Bot' trùng khớp ý nghĩa với 'Đáp án chuẩn'.
       - 0: Nếu sai lệch ý nghĩa.

    3. OUTPUT FORMAT (JSON ONLY):
    Trả về đúng định dạng JSON này, không thêm lời dẫn:
    {{
        "context_score": int,
        "faithfulness_score": int,
        "correctness_score": int,
        "reason": "Giải thích ngắn gọn tại sao chấm điểm như vậy"
    }}
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"Lỗi API: {e}")
        return {"context_score": 0, "faithfulness_score": 0, "correctness_score": 0, "reason": "Error"}