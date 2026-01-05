Nhóm 9 môn bigdata 
Hướng dẫn chạy chương trình:
1.  **Chuẩn bị dữ liệu:**
    Tải file dữ liệu đầu vào `VN_housing_dataset.csv`.
2.  **Làm sạch dữ liệu:**
    Mở file `dataclean.ipynb` (Jupyter Notebook) để chạy các bước xử lý và làm sạch dữ liệu.
    > Kết quả: File sạch sẽ được lưu thành `housing_data_cleaned_v3.csv`.
3.  **Upload lên HDFS:**
    Đưa file `housing_data_cleaned_v3.csv` lên thư mục HDFS ( `/user/housing_data/`).
4.  **Huấn luyện mô hình:**
    Mở terminal và chạy lệnh sau để huấn luyện trên Spark:
    spark-submit train_test.py
