from src.utils import ensure_directories, print_step
from src.data_cleaning import load_and_clean_data
from src.eda import run_eda
from src.train_model import train_models
from src.threshold_optimizer import optimize_threshold
from src.evaluate_model import evaluate_final_model
from src.lime_explainer import build_lime_example
from src.shap_explainer import build_shap_global_explanation
from src.clustering import run_clustering


def main():
    ensure_directories()
    print_step("Bước 1: Làm sạch dữ liệu")
    load_and_clean_data()
    print_step("Bước 2: Khai phá dữ liệu EDA")
    run_eda()
    print_step("Bước 3: Huấn luyện và so sánh mô hình")
    train_models()
    print_step("Bước 4: Tối ưu threshold trên validation")
    optimize_threshold()
    print_step("Bước 5: Đánh giá cuối trên test")
    evaluate_final_model()
    print_step("Bước 6: Tạo LIME local explanation")
    build_lime_example()
    print_step("Bước 7: Tạo SHAP global explanation")
    build_shap_global_explanation()
    print_step("Bước 8: Phân cụm hồ sơ nguy cơ")
    run_clustering()
    print("\nPipeline hoàn tất.")
    print("Chạy app bằng lệnh: streamlit run app.py")


if __name__ == "__main__":
    main()
