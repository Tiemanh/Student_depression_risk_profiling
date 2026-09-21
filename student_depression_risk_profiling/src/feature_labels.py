"""Nhãn tiếng Việt và tiện ích hiển thị cho feature/model."""
from __future__ import annotations

import re
from typing import Any

FEATURE_LABELS = {
    "gender": "Giới tính",
    "age": "Tuổi",
    "profession": "Nghề nghiệp/trạng thái",
    "academic_pressure": "Áp lực học tập",
    "work_pressure": "Áp lực công việc/việc làm thêm",
    "cgpa": "CGPA/điểm trung bình",
    "study_satisfaction": "Hài lòng với việc học",
    "job_satisfaction": "Hài lòng với công việc/việc làm thêm",
    "sleep_duration": "Thời lượng ngủ",
    "dietary_habits": "Thói quen ăn uống",
    "degree": "Bậc học/ngành học",
    "suicidal_thoughts": "Suy nghĩ tự làm hại bản thân",
    "work_study_hours": "Số giờ học/làm mỗi ngày",
    "financial_stress": "Áp lực tài chính",
    "family_history": "Tiền sử gia đình về sức khỏe tinh thần",
    "risk_probability": "Xác suất nguy cơ",
}

VALUE_LABELS = {
    "Male": "Nam",
    "Female": "Nữ",
    "Other": "Khác",
    "Yes": "Có",
    "No": "Không",
    "Student": "Sinh viên",
    "Working Professional": "Sinh viên đi làm / đang đi làm",
    "Less than 5 hours": "Dưới 5 giờ",
    "5-6 hours": "5–6 giờ",
    "7-8 hours": "7–8 giờ",
    "More than 8 hours": "Trên 8 giờ",
    "Others": "Khác",
    "Healthy": "Lành mạnh",
    "Moderate": "Trung bình",
    "Unhealthy": "Không lành mạnh",
}

PROFILE_EXPLANATIONS = {
    "Học tập": "Nguy cơ nổi bật liên quan đến áp lực học tập, điểm số hoặc mức hài lòng với việc học.",
    "Đi làm/công việc": "Nguy cơ nổi bật liên quan đến áp lực việc làm thêm/công việc hoặc mức hài lòng với công việc.",
    "Tài chính": "Nguy cơ nổi bật liên quan đến áp lực tài chính.",
    "Lối sống/giấc ngủ": "Nguy cơ nổi bật liên quan đến giấc ngủ, ăn uống hoặc thời gian học/làm quá dài.",
    "Tâm lý/an toàn": "Nguy cơ nổi bật liên quan đến tín hiệu tâm lý nhạy cảm hoặc tiền sử gia đình.",
    "Chưa xác định rõ": "Các nhóm yếu tố chưa có chênh lệch đủ rõ trong mẫu hiện tại.",
}


def value_to_vietnamese(value: Any) -> str:
    if value is None:
        return "Thiếu"
    s = str(value).strip()
    if s.lower() in {"nan", "none", ""}:
        return "Thiếu"
    return VALUE_LABELS.get(s, s)


def feature_to_vietnamese(feature: Any) -> str:
    """Chuyển tên feature kỹ thuật/one-hot/LIME sang nhãn tiếng Việt dễ hiểu."""
    s = str(feature).strip()
    s = s.replace("num__", "").replace("cat__", "")
    s = re.sub(r"\s+", " ", s)

    # LIME có thể trả dạng: feature = value
    if " = " in s:
        left, right = s.split(" = ", 1)
        return f"{FEATURE_LABELS.get(left, left)} = {value_to_vietnamese(right)}"

    # OneHotEncoder thường tạo feature dạng base_value.
    # Ưu tiên base dài trước để tránh match nhầm.
    bases = sorted(FEATURE_LABELS.keys(), key=len, reverse=True)
    for base in bases:
        if s == base:
            return FEATURE_LABELS.get(base, base)
        if s.startswith(base + "_"):
            val = s[len(base) + 1 :]
            return f"{FEATURE_LABELS.get(base, base)} = {value_to_vietnamese(val)}"

    return FEATURE_LABELS.get(s, s)


def explanation_direction(weight: float) -> str:
    return "Làm tăng nguy cơ" if weight > 0 else "Làm giảm nguy cơ"


def shorten_label(label: str, max_len: int = 42) -> str:
    label = str(label)
    return label if len(label) <= max_len else label[: max_len - 1] + "…"
