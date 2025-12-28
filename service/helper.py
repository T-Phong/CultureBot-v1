from typing import List, Dict, Any

def format_metadata_list_to_context(search_results: List[Dict[str, Any]]) -> str:
    """
    Chuyển đổi danh sách metadata (List[Dict]) thành văn bản ngữ cảnh (String).
    Hỗ trợ cả 2 định dạng từ Hugging Face và Local Disk.
    """
    if not search_results:
        return "Không có dữ liệu ngữ cảnh."
    if isinstance(search_results, dict):
        search_results = [search_results]
    
    full_context_text = ""

    # Lặp qua từng hồ sơ trong danh sách
    for result in search_results:
        data = result.get('metadata', {})

        # 1. Trích xuất dữ liệu (Dùng .get để tránh lỗi nếu thiếu trường)
        ten = data.get('ten') or data.get('group', 'Không rõ tên')
        mo_ta = data.get('mo_ta') or data.get('content', '')
        
        # Nhóm thông tin phân loại
        loai_hinh = data.get('loai_hinh', 'N/A')
        chu_de = data.get('chu_de') or data.get('topic', 'N/A')
        dan_toc = data.get('dan_toc', 'N/A')
        
        # Nhóm thời gian & không gian
        nien_dai = data.get('nien_dai', 'N/A')
        thoi_ky = data.get('thoi_ky', 'N/A')
        vung_mien = data.get('vung_mien', 'N/A')
        dia_diem = data.get('dia_diem', 'N/A')
        
        # Nhóm giá trị nội dung
        chat_lieu = data.get('chat_lieu', 'N/A')
        nguyen_lieu_chinh = data.get('nguyen_lieu_chinh', 'N/A')
        # 2. Tạo template văn bản cho TỪNG hồ sơ
        # Dùng dấu phân cách rõ ràng để Model không lẫn lộn giữa Sọ Dừa và Nguyễn Trãi
        profile_text = f"""
        [TỔNG QUAN]
        Tên: {ten}
        Mô tả/Nội dung: {mo_ta}
        
        [THÔNG TIN CHI TIẾT]
        - Phân loại: {loai_hinh} (Chủ đề: {chu_de})
        - Dân tộc: {dan_toc}
        - Thời gian: {nien_dai} ({thoi_ky})
        - Địa danh/Vùng miền: {vung_mien} - {dia_diem}
        - Chất liệu: {chat_lieu} - Nguyên liệu chính: {nguyen_lieu_chinh}
        """
        
        # 3. Ghép vào chuỗi tổng
        full_context_text += profile_text + "\n"

    return full_context_text
