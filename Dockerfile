# 1. Dùng Python 3.10 (tương thích tốt với Faiss/Torch mới)
FROM python:3.10-slim
# 1. Tắt buffering
ENV PYTHONUNBUFFERED=1

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Copy file requirements trước (để tận dụng cache)
# Nếu bạn vẫn dùng tên 'requirements_rag.txt' thì sửa dòng dưới thành:
COPY requirements_rag.txt requirements.txt

# 4. Cài đặt thư viện
RUN pip install --no-cache-dir -r requirements.txt

# 5. Tạo user non-root trước (Bắt buộc đối với Hugging Face Spaces)
RUN useradd -m -u 1000 user

# 6. Copy toàn bộ code vào container và cấp quyền cho user
# Quan trọng: Cấp quyền sở hữu cho user để tránh lỗi Permission denied khi app cần ghi file
COPY --chown=user . .

# 7. Chuyển sang user non-root
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 8. Mở port 7860 (Port mặc định của HF Spaces)
EXPOSE 7860

# 9. Lệnh chạy server
# "app:app" nghĩa là: trong file app.py, tìm object tên là app (biến Flask)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120", "--workers", "2"]