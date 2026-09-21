# Hệ thống AI sàng lọc và phân tích hồ sơ nguy cơ trầm cảm ở sinh viên

Dự án xây dựng ứng dụng Streamlit hỗ trợ sàng lọc ban đầu nguy cơ trầm cảm ở sinh viên. Hệ thống **không chẩn đoán y khoa** và **không thay thế chuyên gia tâm lý/bác sĩ**.

## Điểm chính của dự án

- LightGBM là mô hình chính.
- XGBoost được dùng để so sánh nếu cài đặt thành công.
- Dữ liệu được chia đúng thành train/validation/test theo tỷ lệ 70/15/15.
- Threshold được chọn trên validation set, test set chỉ dùng để đánh giá cuối.
- LIME giải thích dự đoán cho từng sinh viên.
- SHAP giải thích toàn cục mô hình bằng tiếng Việt.
- Hồ sơ nguy cơ gồm 5 nhóm: Học tập, Đi làm/công việc, Tài chính, Lối sống/giấc ngủ, Tâm lý/an toàn.
- Clustering/KMeans dùng để tìm nhóm sinh viên tương đồng.
- App tiếng Việt, bố cục gọn hơn, biểu đồ thu nhỏ vừa với web.

## Cấu trúc thư mục

```text
student_depression_risk_profiling/
├── data/raw/
├── data/processed/
├── models/
├── results/figures/
├── src/
├── app.py
├── run_all.py
├── requirements.txt
└── README.md
```

## Cài đặt trên Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn activate, chạy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Đặt dataset

Tải dataset Kaggle, đổi tên file CSV thành:

```text
student_depression_dataset.csv
```

Đặt vào:

```text
data/raw/student_depression_dataset.csv
```

Trong thư mục `data/raw` có file `sample_student_depression_dataset.csv` chỉ để test nhanh. Khi làm chính thức, hãy dùng dataset Kaggle đầy đủ.

## Chạy pipeline

```powershell
python run_all.py
```

Pipeline sẽ chạy:

1. Làm sạch dữ liệu
2. EDA
3. Train model
4. Tối ưu threshold trên validation
5. Đánh giá cuối trên test
6. Tạo LIME
7. Tạo SHAP
8. Clustering

## Chạy app

```powershell
streamlit run app.py
```

App có 3 tab:

1. **Sàng lọc, giải thích & hồ sơ nguy cơ**: nhập thông tin, xem kết quả, LIME, hồ sơ nguy cơ và SHAP.
2. **Nhóm tương đồng & khuyến nghị**: giải thích cluster/KMeans và đưa khuyến nghị cá nhân hóa.
3. **Kết quả mô hình cho báo cáo**: bảng và biểu đồ kỹ thuật.

## Lưu ý đạo đức/y khoa

Không dùng kết quả để kết luận “bị trầm cảm”. Cách diễn đạt đúng là “có dấu hiệu nguy cơ” hoặc “cần theo dõi thêm”. Nếu có tín hiệu tự làm hại bản thân, cần khuyến nghị người dùng liên hệ người thân, cố vấn học đường, chuyên viên tâm lý hoặc dịch vụ khẩn cấp tại địa phương.
