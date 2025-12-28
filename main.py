"""
Main entry point để test RAG system
"""
import pandas as pd
from mark import evaluate_rag_pipeline
from rag import retrieve_context
import json

from reranking import advanced_search
from rewrite import QueryRewriter


# --- VÍ DỤ THỰC TẾ ---


def main():
    rewriter = QueryRewriter()
    """Hàm main để test retrieve_context"""
    print("=" * 60)
    print("Vietnam Heritage RAG System - Test")
    print("=" * 60)

    
    
    # Test queries
    test_queries = [
        "Bánh xèo và bánh chưng khác nhau ở điểm nào?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] Query: {query}")
        print("-" * 60)
        
        try:
            # Gọi retrieve_context từ rag.py
            temp = advanced_search(query,['Bánh xèo','bánh chưng'])
            # In kết quả
            print(temp)
        
        except Exception as e:
            print(f"  ❌ Lỗi: {e}")
    
    print("\n" + "=" * 60)
    print("Test hoàn thành!")
    print("=" * 60)


def interactive_mode():
    rewriter = QueryRewriter()
    """Chế độ tương tác - hỏi đáp liên tục"""
    print("\n🎯 Chế độ tương tác (Gõ 'exit' để thoát)")
    print("-" * 60)
    
    while True:
        query = input("\n❓ Câu hỏi: ").strip()
        keyword = []
        print("Nhập chuỗi (nhập 'done' để kết thúc):")
        while True:
            nhap = input()
            if nhap.lower() == 'done':
                break
            keyword.append(nhap)
        print("Mảng của bạn:", keyword)

        if query.lower() in ['exit', 'quit', 'q']:
            print("👋 Tạm biệt!")
            break
        
        if not query:
            print("⚠️  Vui lòng nhập câu hỏi!")
            continue
        
        try:
            
            # rewrite question with key word
            q_rewrite = rewriter.rewrite(query,keyword)
            print(f"\n--- q_rewrite: {q_rewrite} ---")
            # get top 30 RAG and reranking by question rewrite and keyword then get 5
            p = advanced_search(q_rewrite,keyword)
            print("\n📝 Kết quả RAG + Reranking:", p)
        except Exception as e:
            print(f"❌ Lỗi: {e}")

def mark():
    # Giả lập dữ liệu từ hệ thống RAG của bạn
    test_cases = [
        # CASE 1: TỐT TOÀN DIỆN (Tìm đúng, Trả lời đúng)
        {
            "question": "Lễ hội Gióng được UNESCO công nhận năm nào?",
            "retrieved_contexts": [
                "Lễ hội Gióng ở đền Phù Đổng và đền Sóc được UNESCO công nhận là Di sản văn hóa phi vật thể đại diện của nhân loại vào năm 2010.",
                "Thánh Gióng là một trong tứ bất tử."
            ],
            "model_answer": "Lễ hội Gióng được UNESCO công nhận vào năm 2010.",
            "ground_truth": "Năm 2010."
        },
        
        # CASE 2: RETRIEVAL KÉM (Tìm sai tài liệu -> Bot không trả lời được hoặc bịa)
        {
            "question": "Nguyên liệu chính làm quả cầu trong lễ hội gieo cầu là gì?",
            "retrieved_contexts": [
                "Lễ hội Gióng tái hiện trận đánh giặc Ân.", # <--- Context không liên quan gì đến quả cầu
                "Đền Hùng nằm ở Phú Thọ."
            ],
            "model_answer": "Quả cầu được làm bằng nhựa.", # <--- Bot bịa (Hallucination) do không có context
            "ground_truth": "Gỗ hoặc da."
        },
        
        # CASE 3: RETRIEVAL TỐT NHƯNG BOT BỊA (Hallucination)
        {
            "question": "Ý nghĩa của Lễ hội Đền Hùng?",
            "retrieved_contexts": [
                "Lễ hội Đền Hùng thể hiện lòng biết ơn sâu sắc đối với các Vua Hùng đã có công dựng nước."
            ],
            "model_answer": "Lễ hội Đền Hùng là để cầu mưa thuận gió hòa cho miền Tây sông nước.", # <--- Sai, không dựa vào context
            "ground_truth": "Tưởng nhớ công lao dựng nước của các Vua Hùng."
        }
    ]

    # --- VÒNG LẶP ĐÁNH GIÁ ---
    results = []
    print("Đang đánh giá hệ thống RAG...")

    for i, case in enumerate(test_cases):
        print(f"Processing case {i+1}...")
        scores = evaluate_rag_pipeline(
            case["question"], 
            case["retrieved_contexts"], 
            case["model_answer"], 
            case["ground_truth"]
        )
        
        # Gộp kết quả
        case_result = {**case, **scores} # Merge dict
        results.append(case_result)

    # --- HIỂN THỊ KẾT QUẢ ---
    df = pd.DataFrame(results)

    # Chỉ hiện các cột quan trọng
    display_cols = ["question", "context_score", "faithfulness_score", "correctness_score", "reason"]
    print("\nBẢNG ĐIỂM CHI TIẾT:")
    print(df[display_cols].to_string())

    # Tính điểm tổng kết
    print("\n--- TỔNG KẾT HIỆU SUẤT ---")
    print(f"Độ chính xác tìm kiếm (Retrieval Accuracy): {df['context_score'].mean() * 100:.1f}%")
    print(f"Độ trung thực (Faithfulness): {df['faithfulness_score'].mean() * 100:.1f}%")
    print(f"Độ chính xác câu trả lời (End-to-End Accuracy): {df['correctness_score'].mean() * 100:.1f}%")

if __name__ == "__main__":
    import sys
    
    # Kiểm tra argument
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        mark()
