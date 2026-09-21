import pandas as pd


def get_risk_level(risk_probability: float) -> str:
    if risk_probability < 0.35:
        return "Nguy cơ thấp"
    if risk_probability < 0.60:
        return "Cần theo dõi thêm"
    return "Nguy cơ cao"


def _is_yes(v):
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"yes", "y", "true", "1", "có", "co"}


def detect_safety_flag(raw_input_df):
    if "suicidal_thoughts" not in raw_input_df.columns:
        return False
    return _is_yes(raw_input_df.iloc[0]["suicidal_thoughts"])


def generate_recommendation(risk_probability, predicted_label, risk_level, main_profile, secondary_profile=None, cluster_info=None, lime_explanation=None, safety_flag=False):
    recs = []
    if safety_flag:
        recs.append("Nếu bạn đang có suy nghĩ tự làm hại bản thân hoặc cảm thấy không an toàn, hãy liên hệ ngay với người thân, cố vấn học đường, chuyên viên tâm lý hoặc dịch vụ khẩn cấp tại địa phương. Hệ thống này không thay thế chuyên gia.")
    if risk_level == "Nguy cơ cao":
        recs.append("Kết quả cho thấy có dấu hiệu nguy cơ cao. Bạn nên cân nhắc trao đổi thêm với người hỗ trợ hoặc chuyên gia nếu tình trạng kéo dài.")
    elif risk_level == "Cần theo dõi thêm":
        recs.append("Kết quả ở mức cần theo dõi thêm. Bạn nên quan sát trạng thái học tập, giấc ngủ và áp lực cá nhân trong thời gian tới.")
    else:
        recs.append("Kết quả hiện ở mức nguy cơ thấp. Bạn vẫn nên duy trì thói quen học tập, nghỉ ngơi và sinh hoạt ổn định.")

    profiles = [main_profile, secondary_profile]
    for p in profiles:
        if not p:
            continue
        if "Học tập" in p:
            recs.append("Về học tập: nên chia nhỏ kế hoạch học, ưu tiên nhiệm vụ quan trọng và trao đổi với cố vấn/giảng viên nếu áp lực kéo dài.")
        elif "Đi làm" in p or "công việc" in p:
            recs.append("Về việc làm thêm/công việc: nên cân bằng lịch học và lịch làm, tránh kéo dài thời gian làm việc quá mức, và xem xét giảm tải nếu ảnh hưởng đến sức khỏe.")
        elif "Tài chính" in p:
            recs.append("Về tài chính: nên lập kế hoạch chi tiêu, tìm hiểu học bổng hoặc các kênh hỗ trợ tài chính của nhà trường.")
        elif "Lối sống" in p:
            recs.append("Về lối sống/giấc ngủ: nên theo dõi thời lượng ngủ, hạn chế học/làm quá khuya kéo dài và duy trì ăn uống đều đặn.")
        elif "Tâm lý" in p:
            recs.append("Về tâm lý/an toàn: nên tìm sự hỗ trợ từ người thân, cố vấn học đường hoặc chuyên viên tâm lý nếu cảm xúc tiêu cực kéo dài.")
    if cluster_info and cluster_info.get("cluster_name"):
        recs.append(f"Nhóm tương đồng: bạn gần với {cluster_info['cluster_name']}. Hãy xem đây là thông tin tham khảo để hiểu kiểu áp lực chính của bản thân.")
    return list(dict.fromkeys(recs))
