from __future__ import annotations

import math
import pandas as pd
from .config import PROFILE_GROUPS, PROFILE_LABELS
from .feature_labels import PROFILE_EXPLANATIONS


def _safe_float(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _text(v):
    if pd.isna(v):
        return ""
    return str(v).strip().lower()


def _yes(v):
    return _text(v) in {"yes", "y", "true", "1", "có", "co"}


def _sleep_risk(v):
    s = _text(v)
    if "less" in s or "dưới" in s or "<5" in s:
        return 1.0
    if "5-6" in s or "5–6" in s:
        return 0.55
    if "others" in s or "khác" in s:
        return 0.35
    return 0.0


def _diet_risk(v):
    s = _text(v)
    if "unhealthy" in s or "không" in s:
        return 0.9
    if "moderate" in s or "trung" in s:
        return 0.35
    return 0.0


def _low_score_risk(value, max_value=5.0):
    v = _safe_float(value, default=max_value / 2)
    return max(0.0, min(1.0, (max_value - v) / max_value))


def _high_score_risk(value, max_value=5.0):
    v = _safe_float(value, default=0.0)
    return max(0.0, min(1.0, v / max_value))


def _lime_profile_scores(lime_explanation):
    scores = {k: 0.0 for k in PROFILE_GROUPS}
    for item in lime_explanation or []:
        feat = str(item.get("feature", "") or item.get("feature_vi", "")).lower()
        weight = float(item.get("weight", 0.0))
        if weight <= 0:
            continue
        # Map theo cả tên kỹ thuật và tên tiếng Việt để app ổn định hơn.
        for profile_key, keywords in PROFILE_GROUPS.items():
            label = PROFILE_LABELS.get(profile_key, "").lower()
            if label and label in feat:
                scores[profile_key] += weight
                break
            if any(k in feat for k in keywords):
                scores[profile_key] += weight
                break
            if profile_key == "academic" and any(w in feat for w in ["học", "cgpa", "điểm"]):
                scores[profile_key] += weight
                break
            if profile_key == "work" and any(w in feat for w in ["công việc", "việc làm", "nghề nghiệp"]):
                scores[profile_key] += weight
                break
            if profile_key == "financial" and any(w in feat for w in ["tài chính"]):
                scores[profile_key] += weight
                break
            if profile_key == "lifestyle" and any(w in feat for w in ["ngủ", "ăn uống", "số giờ"]):
                scores[profile_key] += weight
                break
            if profile_key == "psychological" and any(w in feat for w in ["tâm lý", "tự làm hại", "tiền sử"]):
                scores[profile_key] += weight
                break
    return scores


def map_lime_to_profiles(lime_explanation):
    """Giữ tương thích với bản cũ: chỉ gom LIME dương vào 5 nhóm hồ sơ."""
    return _lime_profile_scores(lime_explanation)


def score_profiles_from_input(raw_input_df=None, lime_explanation=None):
    """Tính hồ sơ nguy cơ trực quan hơn từ dữ liệu đầu vào, có đối chiếu LIME.

    LIME đôi khi rất cục bộ và tên feature sau one-hot khó đọc, vì vậy app dùng thêm
    các tín hiệu trực tiếp từ form để hồ sơ nguy cơ hợp lý với người dùng hơn.
    """
    scores = {k: 0.0 for k in PROFILE_GROUPS}
    reasons = {k: [] for k in PROFILE_GROUPS}

    if raw_input_df is not None and len(raw_input_df) > 0:
        row = raw_input_df.iloc[0]

        if "academic_pressure" in row.index:
            s = _high_score_risk(row["academic_pressure"])
            scores["academic"] += 1.20 * s
            if s >= 0.6:
                reasons["academic"].append("áp lực học tập cao")
        if "study_satisfaction" in row.index:
            s = _low_score_risk(row["study_satisfaction"])
            scores["academic"] += 0.90 * s
            if s >= 0.6:
                reasons["academic"].append("mức hài lòng với việc học thấp")
        if "cgpa" in row.index:
            cgpa = _safe_float(row["cgpa"], default=7.0)
            # CGPA thấp vừa phải mới tính, tránh áp đặt quá mạnh.
            if cgpa < 6.5:
                s = min(1.0, (6.5 - cgpa) / 3.0)
                scores["academic"] += 0.35 * s
                reasons["academic"].append("điểm trung bình chưa ổn định")

        if "work_pressure" in row.index:
            s = _high_score_risk(row["work_pressure"])
            scores["work"] += 1.10 * s
            if s >= 0.6:
                reasons["work"].append("áp lực công việc/việc làm thêm cao")
        if "job_satisfaction" in row.index:
            s = _low_score_risk(row["job_satisfaction"])
            scores["work"] += 0.75 * s
            if s >= 0.6:
                reasons["work"].append("mức hài lòng với công việc thấp")

        if "financial_stress" in row.index:
            s = _high_score_risk(row["financial_stress"])
            scores["financial"] += 1.15 * s
            if s >= 0.6:
                reasons["financial"].append("áp lực tài chính cao")

        if "sleep_duration" in row.index:
            s = _sleep_risk(row["sleep_duration"])
            scores["lifestyle"] += 0.95 * s
            if s >= 0.5:
                reasons["lifestyle"].append("thời lượng ngủ chưa phù hợp")
        if "dietary_habits" in row.index:
            s = _diet_risk(row["dietary_habits"])
            scores["lifestyle"] += 0.55 * s
            if s >= 0.5:
                reasons["lifestyle"].append("thói quen ăn uống chưa lành mạnh")
        if "work_study_hours" in row.index:
            hours = _safe_float(row["work_study_hours"], default=6.0)
            if hours >= 8:
                s = min(1.0, (hours - 7.0) / 5.0)
                scores["lifestyle"] += 0.60 * s
                reasons["lifestyle"].append("thời gian học/làm mỗi ngày cao")

        if "family_history" in row.index and _yes(row["family_history"]):
            scores["psychological"] += 0.55
            reasons["psychological"].append("có tiền sử gia đình liên quan sức khỏe tinh thần")
        if "suicidal_thoughts" in row.index and _yes(row["suicidal_thoughts"]):
            scores["psychological"] += 1.60
            reasons["psychological"].append("có tín hiệu tâm lý/an toàn cần chú ý")

    lime_scores = _lime_profile_scores(lime_explanation)
    max_lime = max(lime_scores.values()) if lime_scores else 0.0
    if max_lime > 0:
        for k, v in lime_scores.items():
            # LIME đóng vai trò đối chiếu, không lấn át logic form đầu vào.
            scores[k] += 0.45 * (v / max_lime)
            if v > 0:
                reasons[k].append("LIME cũng ghi nhận nhóm này có ảnh hưởng đến dự đoán")

    return scores, reasons


def get_main_profiles(profile_scores, profile_reasons=None, min_score=0.25):
    if not profile_scores:
        return {"main_profile_key": None, "main_profile_label": "Chưa xác định rõ", "secondary_profile_key": None, "secondary_profile_label": None, "sorted_profiles": [], "main_reason": ""}

    sorted_items = sorted(profile_scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_items[0][1] <= min_score:
        main_key = None
        main_label = "Chưa xác định rõ"
    else:
        main_key = sorted_items[0][0]
        main_label = PROFILE_LABELS.get(main_key, main_key)

    secondary_key = sorted_items[1][0] if len(sorted_items) > 1 and sorted_items[1][1] > min_score else None
    secondary_label = PROFILE_LABELS.get(secondary_key) if secondary_key else None

    def reason_for(k):
        if not k:
            return PROFILE_EXPLANATIONS.get("Chưa xác định rõ", "")
        if profile_reasons and profile_reasons.get(k):
            return "; ".join(dict.fromkeys(profile_reasons[k]))
        return PROFILE_EXPLANATIONS.get(PROFILE_LABELS.get(k, k), "")

    return {
        "main_profile_key": main_key,
        "main_profile_label": main_label,
        "secondary_profile_key": secondary_key,
        "secondary_profile_label": secondary_label,
        "main_reason": reason_for(main_key),
        "secondary_reason": reason_for(secondary_key),
        "sorted_profiles": [
            {"profile": k, "label": PROFILE_LABELS.get(k, k), "score": float(v), "reason": reason_for(k)}
            for k, v in sorted_items
        ],
    }
