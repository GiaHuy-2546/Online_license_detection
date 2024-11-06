# Sử dụng hình ảnh cơ sở với Python và Tesseract
FROM python:3.11-slim

# Cài đặt tesseract và các công cụ bổ trợ khác
RUN apt-get update && apt-get install -y tesseract-ocr && apt-get clean

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các tệp dự án vào container
COPY . /app

# Cài đặt các phụ thuộc Python
RUN pip install --no-cache-dir -r requirements.txt

# Khởi chạy ứng dụng
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]