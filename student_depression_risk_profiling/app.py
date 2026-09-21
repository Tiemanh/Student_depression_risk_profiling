import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import MODEL_DIR, RESULTS_DIR, FIGURES_DIR, PROFILE_LABELS
from src.utils import load_pickle
from src.recommendation import get_risk_level, detect_safety_flag, generate_recommendation
from src.lime_explainer import explain_instance
from src.risk_profile import score_profiles_from_input, get_main_profiles
from src.clustering import predict_cluster
from src.feature_labels import feature_to_vietnamese, value_to_vietnamese, explanation_direction

st.set_page_config(page_title="Sàng lọc nguy cơ trầm cảm sinh viên", layout="wide")

st.title("Hệ thống AI sàng lọc và phân tích hồ sơ nguy cơ trầm cảm ở sinh viên")
st.caption("LightGBM + Threshold + LIME cá nhân + SHAP toàn cục + Hồ sơ nguy cơ + Nhóm tương đồng")
st.warning("Hệ thống chỉ hỗ trợ sàng lọc ban đầu, không chẩn đoán y khoa và không thay thế chuyên gia tâm lý/bác sĩ.")


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _load_required():
    try:
        pipeline = load_pickle(MODEL_DIR / "full_pipeline.pkl")
        threshold = load_pickle(MODEL_DIR / "best_threshold.pkl")
        metadata = load_pickle(MODEL_DIR / "feature_metadata.pkl")
        return pipeline, threshold, metadata, None
    except Exception as e:
        return None, None, None, str(e)


def _get_options(metadata, col, fallback):
    values = metadata.get("categorical_values", {}).get(col, []) if metadata else []
    values = [v for v in values if str(v).lower() not in {"nan", "none", ""}]
    return values if values else fallback


def _selectbox_vi(label, options, key=None):
    return st.selectbox(label, options, format_func=value_to_vietnamese, key=key)


def _build_input_form(metadata):
    raw_cols = metadata.get("raw_feature_columns", [])
    data = {c: np.nan for c in raw_cols}

    st.markdown("### 1) Nhập thông tin sinh viên")
    st.caption("Các trường City đã được loại khỏi mô hình để app và dữ liệu huấn luyện nhất quán hơn.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Thông tin học tập**")
        if "gender" in raw_cols:
            data["gender"] = _selectbox_vi("Giới tính", _get_options(metadata, "gender", ["Male", "Female"]))
        if "age" in raw_cols:
            data["age"] = st.number_input("Tuổi", min_value=15, max_value=80, value=22)
        if "profession" in raw_cols:
            data["profession"] = _selectbox_vi(
                "Trạng thái hiện tại", _get_options(metadata, "profession", ["Student", "Working Professional"])
            )
        if "degree" in raw_cols:
            data["degree"] = _selectbox_vi(
                "Bậc học / ngành học", _get_options(metadata, "degree", ["Bachelor", "Master", "PhD", "Other"])
            )
        if "academic_pressure" in raw_cols:
            data["academic_pressure"] = st.slider("Áp lực học tập", 0, 5, 3, help="0 là rất thấp, 5 là rất cao")
        if "cgpa" in raw_cols:
            data["cgpa"] = st.number_input("CGPA / điểm trung bình", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        if "study_satisfaction" in raw_cols:
            data["study_satisfaction"] = st.slider("Mức độ hài lòng với việc học", 0, 5, 3, help="0 là rất không hài lòng, 5 là rất hài lòng")

    with c2:
        st.markdown("**Công việc, lối sống và áp lực cá nhân**")
        if "work_pressure" in raw_cols:
            data["work_pressure"] = st.slider("Áp lực công việc / việc làm thêm", 0, 5, 0, help="Dành cho sinh viên có đi làm thêm hoặc vừa học vừa làm")
        if "job_satisfaction" in raw_cols:
            data["job_satisfaction"] = st.slider("Mức độ hài lòng với công việc / việc làm thêm", 0, 5, 0)
        if "work_study_hours" in raw_cols:
            data["work_study_hours"] = st.number_input("Số giờ học/làm mỗi ngày", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
        if "financial_stress" in raw_cols:
            data["financial_stress"] = st.slider("Áp lực tài chính", 0, 5, 2)
        if "sleep_duration" in raw_cols:
            data["sleep_duration"] = _selectbox_vi(
                "Thời lượng ngủ", _get_options(metadata, "sleep_duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"])
            )
        if "dietary_habits" in raw_cols:
            data["dietary_habits"] = _selectbox_vi(
                "Thói quen ăn uống", _get_options(metadata, "dietary_habits", ["Healthy", "Moderate", "Unhealthy", "Others"])
            )
        if "family_history" in raw_cols:
            data["family_history"] = _selectbox_vi(
                "Tiền sử gia đình về vấn đề sức khỏe tinh thần", _get_options(metadata, "family_history", ["No", "Yes"])
            )
        if "suicidal_thoughts" in raw_cols:
            data["suicidal_thoughts"] = _selectbox_vi(
                "Đã từng có suy nghĩ tự làm hại bản thân?", _get_options(metadata, "suicidal_thoughts", ["No", "Yes"])
            )
    return pd.DataFrame([data], columns=raw_cols)


def _show_image(path, caption=None, width=650):
    p = Path(path)
    if p.exists():
        st.image(str(p), caption=caption, width=width)
    else:
        st.info(f"Chưa có file: {p.name}")


def _plot_lime(lime_df):
    if lime_df.empty:
        return None
    plot_df = lime_df.copy().iloc[::-1]
    colors = ["#d62728" if w > 0 else "#2ca02c" for w in plot_df["weight"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh(plot_df["feature_vi"], plot_df["weight"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("LIME: yếu tố ảnh hưởng đến dự đoán cá nhân")
    ax.set_xlabel("Trọng số LIME")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def _plot_profile_scores(score_df):
    if score_df.empty:
        return None
    plot_df = score_df.copy()
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.bar(plot_df["label"], plot_df["score"])
    ax.set_title("Điểm hồ sơ nguy cơ")
    ax.set_ylabel("Điểm tương đối")
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    fig.tight_layout()
    return fig


def _show_shap_global():
    st.markdown("#### SHAP: giải thích toàn cục của mô hình")
    st.caption("SHAP cho biết trên toàn bộ tập kiểm thử, mô hình LightGBM thường dựa nhiều nhất vào những yếu tố nào.")
    shap_csv = RESULTS_DIR / "shap_global_importance.csv"
    if shap_csv.exists():
        shap_df = pd.read_csv(shap_csv)
        if "feature_vi" not in shap_df.columns:
            col = "feature" if "feature" in shap_df.columns else shap_df.columns[0]
            shap_df["feature_vi"] = shap_df[col].apply(feature_to_vietnamese)
        value_col = "mean_abs_shap" if "mean_abs_shap" in shap_df.columns else shap_df.columns[-1]
        show_df = shap_df[["feature_vi", value_col]].head(10).rename(columns={
            "feature_vi": "Yếu tố",
            value_col: "Độ quan trọng SHAP trung bình",
        })
        st.dataframe(show_df, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có bảng SHAP. Hãy chạy lại `python run_all.py`.")
    _show_image(FIGURES_DIR / "shap_global_bar.png", "SHAP: top yếu tố quan trọng nhất", width=620)
    _show_image(FIGURES_DIR / "shap_profile_importance.png", "SHAP theo nhóm hồ sơ nguy cơ", width=560)


# -----------------------------------------------------------------------------
# UI layout
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "1. Sàng lọc, giải thích & hồ sơ nguy cơ",
    "2. Nhóm tương đồng & khuyến nghị",
    "3. Kết quả mô hình cho báo cáo",
])

with tab1:
    with st.expander("Dự án này hoạt động như thế nào?", expanded=False):
        st.markdown(
            """
            **Luồng xử lý:** dữ liệu sinh viên → LightGBM dự đoán xác suất nguy cơ → threshold phân mức → LIME giải thích cá nhân → hồ sơ nguy cơ → SHAP giải thích toàn cục → khuyến nghị.

            **LIME** dùng để giải thích vì sao **một sinh viên cụ thể** có kết quả như vậy.  
            **SHAP** dùng để giải thích **toàn bộ mô hình** thường quan tâm đến yếu tố nào.  
            **Hồ sơ nguy cơ** gom các tín hiệu thành 5 nhóm dễ hiểu: học tập, đi làm/công việc, tài chính, lối sống/giấc ngủ và tâm lý/an toàn.
            """
        )

    pipeline, threshold, metadata, err = _load_required()
    if err:
        st.error("Chưa tìm thấy model hoặc artifacts. Hãy chạy `python run_all.py` trước.")
        st.code(err)
    else:
        raw_input_df = _build_input_form(metadata)
        run_predict = st.button("Dự đoán và giải thích kết quả", type="primary", use_container_width=True)

        if run_predict:
            try:
                risk_probability = float(pipeline.predict_proba(raw_input_df)[:, 1][0])
                predicted_label = bool(risk_probability >= threshold)
                risk_level = get_risk_level(risk_probability)
                safety_flag = detect_safety_flag(raw_input_df)

                st.session_state.raw_input_df = raw_input_df
                st.session_state.risk_probability = risk_probability
                st.session_state.predicted_label = predicted_label
                st.session_state.risk_level = risk_level
                st.session_state.safety_flag = safety_flag

                try:
                    lime_explanation = explain_instance(raw_input_df)
                except Exception as e:
                    st.warning(f"Không tạo được LIME cho mẫu này, hồ sơ nguy cơ sẽ dựa trên dữ liệu nhập: {e}")
                    lime_explanation = []
                st.session_state.lime_explanation = lime_explanation

                profile_scores, profile_reasons = score_profiles_from_input(raw_input_df, lime_explanation)
                profile_info = get_main_profiles(profile_scores, profile_reasons)
                st.session_state.profile_info = profile_info
                st.success("Đã dự đoán và tạo giải thích.")
            except Exception as e:
                st.error(f"Không dự đoán được: {e}")

        if "risk_probability" in st.session_state:
            st.markdown("---")
            st.markdown("### 2) Kết quả sàng lọc")
            c1, c2, c3 = st.columns(3)
            c1.metric("Xác suất nguy cơ", f"{st.session_state.risk_probability:.2%}")
            c2.metric("Kết luận sàng lọc", "Có dấu hiệu nguy cơ" if st.session_state.predicted_label else "Chưa thấy dấu hiệu rõ")
            c3.metric("Mức nguy cơ", st.session_state.risk_level)
            if st.session_state.safety_flag:
                st.error("Nếu bạn đang có suy nghĩ tự làm hại bản thân hoặc cảm thấy không an toàn, hãy liên hệ ngay với người thân, cố vấn học đường, chuyên viên tâm lý hoặc dịch vụ khẩn cấp tại địa phương.")

            st.markdown("### 3) Giải thích cá nhân bằng LIME")
            lime_explanation = st.session_state.get("lime_explanation", [])
            if lime_explanation:
                lime_df = pd.DataFrame(lime_explanation)
                if "feature_vi" not in lime_df.columns:
                    lime_df["feature_vi"] = lime_df["feature"].apply(feature_to_vietnamese)
                lime_df["direction"] = lime_df["weight"].apply(explanation_direction)
                show_lime = lime_df[["feature_vi", "direction", "weight"]].rename(columns={
                    "feature_vi": "Yếu tố",
                    "direction": "Chiều ảnh hưởng",
                    "weight": "Trọng số LIME",
                })
                left, right = st.columns([1.05, 1])
                with left:
                    st.dataframe(show_lime, use_container_width=True, hide_index=True)
                with right:
                    fig = _plot_lime(lime_df)
                    if fig:
                        st.pyplot(fig)
            else:
                st.info("Chưa có LIME cho mẫu này.")

            st.markdown("### 4) Nguồn nguy cơ")
            profile_info = st.session_state.get("profile_info", {})
            pc1, pc2 = st.columns(2)
            pc1.metric("Nguồn nguy cơ chính", profile_info.get("main_profile_label", "Chưa xác định rõ"))
            pc2.metric("Nguồn nguy cơ phụ", profile_info.get("secondary_profile_label") or "Không rõ")
            if profile_info.get("main_reason"):
                st.info(f"**Vì sao:** {profile_info['main_reason']}")

            score_df = pd.DataFrame(profile_info.get("sorted_profiles", []))
            if not score_df.empty:
                show_profile = score_df[["label", "score", "reason"]].rename(columns={
                    "label": "Nhóm hồ sơ",
                    "score": "Điểm tương đối",
                    "reason": "Lý do nổi bật",
                })
                left, right = st.columns([1.05, 1])
                with left:
                    st.dataframe(show_profile, use_container_width=True, hide_index=True)
                with right:
                    fig = _plot_profile_scores(score_df)
                    if fig:
                        st.pyplot(fig)
            st.caption("Nguồn nguy cơ được tính từ thông tin nhập vào và được đối chiếu với LIME, nên dễ hiểu hơn so với chỉ nhìn từng feature rời rạc.")

            with st.expander("Xem thêm: giải thích toàn cục bằng SHAP", expanded=False):
                _show_shap_global()

with tab2:
    if "raw_input_df" not in st.session_state:
        st.info("Vui lòng dự đoán ở Tab 1 trước.")
    else:
        st.markdown("### Nhóm sinh viên tương đồng là gì?")
        st.markdown(
            """
            Đây là kết quả **clustering/KMeans** trên dữ liệu huấn luyện. Hệ thống tìm các nhóm sinh viên có kiểu đặc điểm gần giống nhau, ví dụ: nhóm áp lực học tập, nhóm áp lực đi làm, nhóm áp lực tài chính hoặc nhóm lối sống/giấc ngủ.

            Kết quả này **không phải chẩn đoán**, mà là cách để người dùng hiểu mình đang gần với kiểu hồ sơ nào trong dữ liệu.
            """
        )

        cluster_info = predict_cluster(st.session_state.raw_input_df, st.session_state.risk_probability)
        st.session_state.cluster_info = cluster_info
        c1, c2 = st.columns([0.8, 1.2])
        with c1:
            st.metric("Cluster ID", cluster_info.get("cluster_id") if cluster_info.get("cluster_id") is not None else "?")
            st.success(cluster_info.get("cluster_name", "Chưa xác định"))
        with c2:
            st.info(cluster_info.get("cluster_description", "Không có mô tả chi tiết."))
        _show_image(FIGURES_DIR / "pca_clusters.png", "Trực quan hóa các nhóm sinh viên tương đồng bằng PCA", width=620)

        st.markdown("### Khuyến nghị cá nhân hóa")
        profile_info = st.session_state.get("profile_info", {"main_profile_label": None, "secondary_profile_label": None})
        recs = generate_recommendation(
            st.session_state.risk_probability,
            st.session_state.predicted_label,
            st.session_state.risk_level,
            profile_info.get("main_profile_label"),
            profile_info.get("secondary_profile_label"),
            cluster_info,
            st.session_state.get("lime_explanation", []),
            st.session_state.safety_flag,
        )
        for i, rec in enumerate(recs, 1):
            if st.session_state.safety_flag and i == 1:
                st.error(rec)
            else:
                st.info(rec)
        st.caption("Khuyến nghị chỉ mang tính tham khảo ban đầu, không thay thế tư vấn chuyên môn.")

with tab3:
    st.markdown("### Kết quả mô hình phục vụ báo cáo")
    st.caption("Phần này dành cho báo cáo kỹ thuật, không phải phần người dùng cuối cần đọc.")
    for csv_name in [
        "data_split_summary.csv",
        "model_comparison.csv",
        "final_metrics.csv",
        "threshold_optimization.csv",
        "cluster_summary.csv",
        "shap_profile_importance.csv",
        "shap_global_importance.csv",
    ]:
        p = RESULTS_DIR / csv_name
        if p.exists():
            with st.expander(csv_name, expanded=False):
                df = pd.read_csv(p)
                if csv_name == "shap_global_importance.csv" and "feature_vi" in df.columns:
                    df = df[["feature_vi", "mean_abs_shap"]].rename(columns={"feature_vi": "Yếu tố", "mean_abs_shap": "Độ quan trọng SHAP"})
                st.dataframe(df.head(30), use_container_width=True, hide_index=True)

    st.markdown("#### Biểu đồ báo cáo")
    for img, cap in [
        ("model_comparison.png", "So sánh mô hình"),
        ("confusion_matrix.png", "Confusion matrix"),
        ("roc_curve.png", "ROC curve"),
        ("pr_curve.png", "Precision-Recall curve"),
        ("threshold_analysis.png", "Phân tích threshold"),
        ("cluster_radar_chart.png", "Tóm tắt nhóm tương đồng"),
        ("shap_global_bar.png", "SHAP toàn cục"),
        ("shap_profile_importance.png", "SHAP theo nhóm hồ sơ"),
    ]:
        _show_image(FIGURES_DIR / img, cap, width=620)
