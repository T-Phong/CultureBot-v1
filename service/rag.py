import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from typing import List, Dict, Any, Optional

from helper import format_metadata_list_to_context

# ==============================================================================
# HỆ THỐNG RAG 1: SỬ DỤNG HUGGING FACE DATASET
# ==============================================================================
class HuggingFaceRAGService:
    _instance: Optional['HuggingFaceRAGService'] = None
    
    # Singleton Pattern: Đảm bảo chỉ có một instance của lớp này được tạo ra
    def __new__(cls):
        if cls._instance is None:
            print("Khởi tạo HuggingFaceRAGService...")
            cls._instance = super(HuggingFaceRAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # Cấu hình
        self.MODEL_NAME = "all-MiniLM-L6-v2"
        self.DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        self.FAISS_PATH = os.path.join(self.DATA_DIR, "heritage.faiss")
        self.METADATA_PATH = os.path.join(self.DATA_DIR, "metadata.json")
        self.IDS_PATH = os.path.join(self.DATA_DIR, "ids.json")
        
        # Tải model và dữ liệu
        self._load_model()
        self._load_data()
        self._initialized = True
        print("✅ HuggingFaceRAGService đã sẵn sàng.")

    def _load_model(self):
        print(f"🤖 [HF RAG] Đang tải model: {self.MODEL_NAME}...")
        self.model = SentenceTransformer(self.MODEL_NAME)

    def _load_data(self):
        self.index, self.metadata, self.ids = self._load_cache()
        if self.index and self.metadata and self.ids:
            print(f"💾 [HF RAG] Sử dụng cache FAISS index và metadata (items: {len(self.ids)})")
        else:
            print("💾 [HF RAG] Cache không tồn tại. Tải dataset và xây dựng FAISS index...")
            dataset = load_dataset("synguyen1106/vietnam_heritage_embeddings_v4", split="train")
            vectors = np.array(dataset['embedding']).astype("float32")
            self.metadata = [{k: v for k, v in dataset[i].items() if k not in ['embedding', 'id', 'slug']} for i in range(len(dataset))]
            self.ids = [dataset[i]['id'] for i in range(len(dataset))]
            print(f"💾 [HF RAG] Đã tải {len(self.ids)} mục từ dataset.")
            
            d = vectors.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(vectors)
            print("🔨 [HF RAG] Số lượng vector trong FAISS index:", self.index.ntotal)
            
            self._save_cache(self.index, self.metadata, self.ids)
            print(f"💾 [HF RAG] Đã lưu cache tại: {self.FAISS_PATH}")

    def _save_cache(self, faiss_index, metadata_list, ids_list):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        faiss.write_index(faiss_index, self.FAISS_PATH)
        with open(self.METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False)
        with open(self.IDS_PATH, "w", encoding="utf-8") as f:
            json.dump(ids_list, f, ensure_ascii=False)

    def _load_cache(self):
        if not (os.path.exists(self.FAISS_PATH) and os.path.exists(self.METADATA_PATH) and os.path.exists(self.IDS_PATH)):
            return None, None, None
        idx = faiss.read_index(self.FAISS_PATH)
        with open(self.METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(self.IDS_PATH, "r", encoding="utf-8") as f:
            ids_local = json.load(f)
        return idx, meta, ids_local

    def search(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        _, indices = self.index.search(query_vec, k)
        results = [{"metadata": self.metadata[int(idx)]} for idx in indices[0]]
        return results

# ==============================================================================
# HỆ THỐNG RAG 2: SỬ DỤNG LOCAL DISK DATASET
# ==============================================================================
class LocalDiskRAGService:
    _instance: Optional['LocalDiskRAGService'] = None

    def __new__(cls):
        if cls._instance is None:
            print("\nKhởi tạo LocalDiskRAGService...")
            cls._instance = super(LocalDiskRAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # Cấu hình
        self.MODEL_NAME = 'AITeamVN/Vietnamese_Embedding_v2'
        # Thay đổi từ đường dẫn local sang ID của dataset trên Hugging Face Hub
        self.DATASET_ID = "phongnt251199/Wiki_Culture_Vec"
        self.MIN_CONTENT_LENGTH = 200
        self.CANDIDATE_MULTIPLIER = 5
        
        # Tải model và dữ liệu
        self._load_model()
        self._load_data()
        self._initialized = True
        print("✅ LocalDiskRAGService đã sẵn sàng.")

    def _load_model(self):
        print(f"🤖 [Local RAG] Đang tải model AI: {self.MODEL_NAME}...")
        self.model = SentenceTransformer(self.MODEL_NAME)

    def _load_data(self):
        print(f"💾 [Local RAG] Đang tải dữ liệu từ Hugging Face Hub: {self.DATASET_ID}...")
        try:
            # Tải toàn bộ dataset về và lấy đường dẫn local
            # Hugging Face Spaces sẽ tự động sử dụng token trong secrets nếu repo là private
            dataset_path = snapshot_download(repo_id=self.DATASET_ID, repo_type="dataset")
            
            self.dataset = load_from_disk(dataset_path)
            print(f"💾 [Local RAG] Load xong! Tổng số dữ liệu: {len(self.dataset)} dòng.")
            
            print("🔨 [Local RAG] Đang kích hoạt bộ tìm kiếm (Re-indexing)...")
            self.dataset.add_faiss_index(column="embeddings")
            print("🔨 [Local RAG] Đã kích hoạt xong FAISS Index!")
        except Exception as e:
            print(f"❌ Lỗi: Không thể tải dataset từ Hub. Lỗi: {e}")
            self.dataset = None
            return

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.dataset:
            return []
            
        # print(f"\n🔎 [Local RAG] Đang tìm: '{query}'")
        # print("-" * 50)

        query_vector = self.model.encode(query)
        candidate_k = top_k * self.CANDIDATE_MULTIPLIER
        scores, samples = self.dataset.get_nearest_examples("embeddings", query_vector, k=candidate_k)

        results = []
        for i in range(len(samples['original_content'])):
            if len(results) >= top_k:
                break
            
            content = samples['original_content'][i]
            if len(content) < self.MIN_CONTENT_LENGTH:
                continue

            score = scores[i]
            metadata = samples['metadata'][i]
            metadata['content'] = content
            
            results.append({
                "metadata": metadata,
                "score": score
            })
            
            # In ra console để debug như hàm gốc
            # print(f"Top {len(results)} (Độ sai lệch: {score:.2f}):")
            # print(f"Nội dung: {content[:200]}...")
            # print("-" * 50)

        if not results:
            print(f"Không tìm thấy kết quả nào có nội dung dài hơn {self.MIN_CONTENT_LENGTH} ký tự.")
        
        return results

# ==============================================================================
# KHỞI TẠO SERVICE VÀ CUNG CẤP CÁC HÀM GỐC
# ==============================================================================
hf_rag_service = HuggingFaceRAGService()
local_rag_service = LocalDiskRAGService()

def retrieve_context(query: str, k: int = 2) -> str:
    """
    Tìm kiếm ngữ cảnh sử dụng hệ thống RAG từ Hugging Face.
    (Giữ nguyên hàm gốc để tương thích)
    """
    print("\n>>> Sử dụng hệ thống RAG 1 (HuggingFace)...")
    results = hf_rag_service.search(query, k)
    return format_metadata_list_to_context(results)

def search_heritage(query: str, top_k: int = 3) -> str:
    """
    Tìm kiếm di sản sử dụng hệ thống RAG từ ổ đĩa cục bộ.
    (Giữ nguyên hàm gốc để tương thích)
    """
    print("\n>>> Sử dụng hệ thống RAG 2 (Local Disk)...")
    results = local_rag_service.search(query, top_k)
    return format_metadata_list_to_context(results)