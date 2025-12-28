import ast
import concurrent.futures
from sentence_transformers import CrossEncoder

from helper import format_metadata_list_to_context
from rag import hf_rag_service, local_rag_service

# Load model Reranker (nhẹ, chạy CPU được)
#reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
reranker = CrossEncoder('Alibaba-NLP/gte-multilingual-reranker-base', trust_remote_code=True)

def reciprocal_rank_fusion(search_results_lists: list, k_rrf: int = 60):
    """
    Áp dụng Reciprocal Rank Fusion (RRF) để kết hợp thông minh nhiều danh sách kết quả.
    Tài liệu nào có thứ hạng cao ở nhiều danh sách sẽ được đẩy lên đầu.
    """
    fused_scores = {}
    doc_map = {} # Lưu lại object của tài liệu để không cần truy xuất lại

    # Lặp qua từng danh sách kết quả (từ mỗi lần search)
    for doc_list in search_results_lists:
        for rank, doc in enumerate(doc_list):
            # Tạo một khóa định danh duy nhất cho mỗi tài liệu
            unique_key = doc['metadata'].get('ten', '').strip().lower() or doc['metadata'].get('group', '').strip().lower()
            if not unique_key:
                # Nếu không có tên, dùng 100 ký tự đầu của nội dung làm khóa
                unique_key = doc['metadata'].get('content', '')[:100].strip().lower()
            
            if not unique_key:
                continue # Bỏ qua nếu không có khóa

            if unique_key not in doc_map:
                doc_map[unique_key] = doc

            if unique_key not in fused_scores:
                fused_scores[unique_key] = 0
            # Công thức RRF: cộng điểm dựa trên thứ hạng nghịch đảo
            fused_scores[unique_key] += 1 / (rank + k_rrf)

    reranked_results = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return [doc_map[key] for key in reranked_results]

def advanced_search(query, keyword):
    """
    Tìm kiếm nâng cao kết hợp cả 2 nguồn dữ liệu, rerank và định dạng riêng biệt.
    """
    try:
        result_wiki = []
        result_hf = []
        
        # 1. Tạo danh sách các cụm từ tìm kiếm
        search_terms = [query]
        if isinstance(keyword, list):
            search_terms.extend(keyword)
        elif isinstance(keyword, str) and keyword:
            search_terms.append(keyword)

        # Tối ưu: Tìm kiếm song song trên cả 2 nguồn với mỗi cụm từ
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_source = {}
            for r in search_terms:
                # HF RAG
                future_hf = executor.submit(hf_rag_service.search, r.lower(), k=15)
                future_to_source[future_hf] = 'hf'
                # Local RAG
                future_local = executor.submit(local_rag_service.search, r.lower(), top_k=15)
                future_to_source[future_local] = 'local'
            
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    docs = future.result()
                    if docs:
                        if source == 'hf':
                            result_hf.append(docs)
                        else:
                            result_wiki.append(docs)
                except Exception as e:
                    print(f"Lỗi tìm kiếm {source}: {e}")

        if not result_hf and not result_wiki:
            return "Không tìm thấy thông tin phù hợp."

        # 2. Kết hợp kết quả từ các lần tìm kiếm (Fusion) và Rerank cho từng nguồn
        # Xử lý nguồn HF RAG
        fused_results_hf = reciprocal_rank_fusion(result_hf)
        # Tối ưu: Giảm số lượng ứng viên rerank xuống 15 để tăng tốc độ
        candidates_for_rerank_hf = fused_results_hf[:15]
        pairs_to_score_hf = []
        for item in candidates_for_rerank_hf:
            meta = item['metadata']
            name = meta.get('ten','') 
            desc = meta.get('mo_ta','')
            loai_hinh = meta.get('loai_hinh', '')
            chu_de = meta.get('chu_de', '')
            y_nghia = meta.get('y_nghia', '')
            constructed_text = f"Tên: {name}. Loại hình: {loai_hinh}. Chủ đề: {chu_de}. Mô tả: {desc}. Ý nghĩa: {y_nghia}"
            pairs_to_score_hf.append([query, constructed_text])
        
        sorted_docs_hf = []
        if pairs_to_score_hf:
            scores_hf = reranker.predict(pairs_to_score_hf)
            for i, doc in enumerate(candidates_for_rerank_hf):
                doc['rerank_score'] = scores_hf[i]
            sorted_docs_hf = sorted(candidates_for_rerank_hf, key=lambda x: x['rerank_score'], reverse=True)

        # Xử lý nguồn Local (Wiki) RAG
        fused_results_wiki = reciprocal_rank_fusion(result_wiki)
        # Tối ưu: Giảm số lượng ứng viên rerank xuống 40
        candidates_for_rerank_wiki = fused_results_wiki[:40]
        print("candidates_for_rerank_wiki:", len(candidates_for_rerank_wiki))
        pairs_to_score_wiki = []
        for item in candidates_for_rerank_wiki:
            meta = item['metadata']
            name = meta.get('group','')
            desc = meta.get('content','')
            constructed_text = f"Tên: {name}. Mô tả: {desc}"
            pairs_to_score_wiki.append([query, constructed_text])
                   
        sorted_docs_wiki = []
        if pairs_to_score_wiki:
            scores_wiki = reranker.predict(pairs_to_score_wiki)
            for i, doc in enumerate(candidates_for_rerank_wiki):
                doc['rerank_score'] = scores_wiki[i]
            sorted_docs_wiki = sorted(candidates_for_rerank_wiki, key=lambda x: x['rerank_score'], reverse=True)
        
        # 3. Lấy Top 3 từ mỗi loại và định dạng riêng biệt
        top_3_hf = sorted_docs_hf[:3]
        top_3_wiki = sorted_docs_wiki[:3]

        # Định dạng kết quả từ HF RAG (sử dụng helper, định dạng đầy đủ)
        hf_context = format_metadata_list_to_context(top_3_hf)

        # Định dạng kết quả từ Wiki RAG (chỉ tên + mô tả)
        wiki_context_parts = []
        for doc in top_3_wiki:
            metadata = doc.get('metadata', {})
            name = metadata.get('group', 'Không rõ tên')
            section = metadata.get('section', '')
            description = metadata.get('content', 'Không có mô tả.')
            wiki_context_parts.append(f"[Nguồn Wiki - Tên]: {name}\n[Nguồn Wiki - Mô tả]: {description}" + (f"\n[Nguồn Wiki - Section]: {section}" if section else ""))
        
        wiki_context = "\n\n".join(wiki_context_parts)

        # 4. Kết hợp hai ngữ cảnh thành một chuỗi duy nhất
        final_context_parts = []
        
        if wiki_context:
            final_context_parts.append(wiki_context)


        if hf_context and hf_context.strip() != "Không có dữ liệu ngữ cảnh.": 
            final_context_parts.append(hf_context)
        
        if not final_context_parts:
            return "Không tìm thấy thông tin phù hợp."

        return "\n\n".join(final_context_parts).strip()
    except Exception as e:
        print(f"Lỗi trong advanced_search: {e}")
        return "Đã xảy ra lỗi trong quá trình tìm kiếm nâng cao."
