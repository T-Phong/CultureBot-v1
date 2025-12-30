import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from typing import List, Dict, Any, Optional
from huggingface_hub import hf_hub_download
from helper import format_metadata_list_to_context

# ==============================================================================
# HỆ THỐNG RAG 1: SỬ DỤNG HUGGING FACE DATASET
# ==============================================================================
class HuggingFaceRAGService:
    _instance: Optional['HuggingFaceRAGService'] = None
    
    # Singleton Pattern
    def __new__(cls):
        if cls._instance is None:
            print("Khởi tạo HuggingFaceRAGService...")
            cls._instance = super(HuggingFaceRAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # --- CẤU HÌNH ---
        self.MODEL_NAME = "all-MiniLM-L6-v2"
        
        # ID của Repo trên Hugging Face chứa file index và data
        # Bạn cần đảm bảo đã upload file .faiss và .json lên repo này (dạng Dataset hoặc Model)
        self.HF_REPO_ID = "synguyen1106/vietnam_heritage_embeddings_v4"
        self.HF_REPO_TYPE = "dataset" # Hoặc "model" hoặc "space" tùy nơi bạn để file
        
        # Tên file trên repo HF
        self.FILENAME_INDEX = "heritage.faiss"
        self.FILENAME_META = "metadata.json"
        # self.FILENAME_IDS = "ids.json" # Nếu bạn gộp vào metadata thì ko cần file này
        
        # Load model & Data
        self._load_model()
        self._load_data()
        
        self._initialized = True
        print("✅ HuggingFaceRAGService đã sẵn sàng.")

    def _load_model(self):
        print(f"🤖 [HF RAG] Đang tải model embedding: {self.MODEL_NAME}...")
        self.model = SentenceTransformer(self.MODEL_NAME)

    def _load_data(self):
        """
        Chiến lược:
        1. Cố gắng tải file index đã build sẵn từ Hugging Face (Nhanh, tránh lỗi LFS).
        2. Nếu không tìm thấy file trên HF, fallback về việc tải Dataset gốc và build lại index (Chậm hơn).
        """
        try:
            print(f"⬇️ [HF RAG] Đang thử tải Index pre-built từ HF Hub: {self.HF_REPO_ID}...")
            
            # 1. Tải file FAISS Index
            # hf_hub_download sẽ tự xử lý caching và LFS pointer
            index_path = hf_hub_download(
                repo_id=self.HF_REPO_ID,
                filename=self.FILENAME_INDEX,
                repo_type=self.HF_REPO_TYPE
            )
            
            # 2. Tải file Metadata
            metadata_path = hf_hub_download(
                repo_id=self.HF_REPO_ID,
                filename=self.FILENAME_META,
                repo_type=self.HF_REPO_TYPE
            )

            # 3. Load vào RAM
            print(f"📂 [HF RAG] Đang đọc file index từ: {index_path}")
            self.index = faiss.read_index(index_path)
            
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                
            print(f"✅ [HF RAG] Load thành công từ Cache HF! (Items: {self.index.ntotal})")

        except Exception as e:
            print(f"⚠️ [HF RAG] Không tải được pre-built index ({e}). \n🔄 Chuyển sang build từ Dataset gốc...")
            self._build_from_dataset()

    def _build_from_dataset(self):
        """
        Hàm fallback: Tải dataset thô và build index tại chỗ (Tốn RAM và CPU lúc khởi động)
        """
        print("💾 [HF RAG] Đang tải dataset và xây dựng FAISS index mới...")
        dataset = load_dataset(self.HF_REPO_ID, split="train")
        
        # Chuẩn bị vectors
        vectors = np.array(dataset['embedding']).astype("float32")
        
        # Chuẩn bị metadata (loại bỏ cột embedding để nhẹ RAM)
        self.metadata = [{k: v for k, v in item.items() if k != 'embedding'} for item in dataset]
        
        # Build Index
        d = vectors.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(vectors)
        
        print(f"🔨 [HF RAG] Đã build xong index. Số lượng vector: {self.index.ntotal}")
        
        # Mẹo: Ở đây bạn có thể lưu file ra đĩa và upload ngược lên HF để lần sau dùng cách 1

    def search(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        # Encode câu hỏi
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        
        # Search FAISS
        distances, indices = self.index.search(query_vec, k)
        
        # Map kết quả
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # Kiểm tra nếu tìm thấy
                item = {
                    "score": float(distances[0][i]), # Distance càng nhỏ càng giống (với L2)
                    "metadata": self.metadata[int(idx)]
                }
                results.append(item)
                
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