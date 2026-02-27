import re
from typing import Any, Callable, Dict, List, Optional, Tuple


MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


FEATURE_RANGES = {
    "Year": (1900.0, 2100.0),
    "Month": (1.0, 12.0),
    "Day": (1.0, 31.0),
    "Inflation Rate": (-20.0, 40.0),
    "Unemployment Rate": (0.0, 50.0),
    "Real GDP (Percent Change)": (-1_000_000.0, 1_000_000.0),
}


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip().lower()
    if not s:
        return None

    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("usd", "")
    s = s.replace("percent", "")
    s = s.replace("%", "")
    s = re.sub(r"\b(trillion|billion|million|tn|bn|mn)\b", "", s)

    match = re.search(r"-?\d+(?:\.\d+)?", s)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_features_from_text(user_message: str, required_features: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {f: None for f in required_features}
    text = user_message.lower()

    year_match = re.search(r"\b(19\d{2}|20\d{2}|2100)\b", text)
    if year_match and "Year" in out:
        out["Year"] = float(year_match.group(1))

    month_match = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        text,
    )
    if month_match and "Month" in out:
        out["Month"] = float(MONTH_MAP[month_match.group(1)])

    if "Day" in out:
        day_match = re.search(r"\bday\s*(?:of\s*month\s*)?(\d{1,2})\b", text)
        if not day_match and month_match:
            tail = text[month_match.end() :]
            day_match = re.search(r"^\s*[,\s]+(\d{1,2})\b", tail)
        if not day_match:
            day_match = re.search(r"\b(3[01]|[12]\d|[1-9])\b", text)
        if day_match:
            day_value = int(day_match.group(1))
            if 1 <= day_value <= 31:
                out["Day"] = float(day_value)

    gdp_match = re.search(
        r"(?:real\s+)?gdp[^\d$-]*([$]?\s*-?\d[\d,]*(?:\.\d+)?)\s*(?:trillion|billion|million|tn|bn|mn)?",
        text,
    )
    if not gdp_match:
        gdp_match = re.search(r"([$]\s*-?\d[\d,]*(?:\.\d+)?)\s*(?:trillion|billion|million|tn|bn|mn)", text)
    if gdp_match and "Real GDP (Percent Change)" in out:
        out["Real GDP (Percent Change)"] = safe_float(gdp_match.group(0))

    inflation_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%?\s*(?:inflation|cpi)", text)
    if inflation_match and "Inflation Rate" in out:
        out["Inflation Rate"] = float(inflation_match.group(1))

    unemployment_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%?\s*(?:unemployment|jobless)", text)
    if unemployment_match and "Unemployment Rate" in out:
        out["Unemployment Rate"] = float(unemployment_match.group(1))

    pct_values = [float(m.group(1)) for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", text)]
    if pct_values:
        if "Inflation Rate" in out and out["Inflation Rate"] is None:
            out["Inflation Rate"] = pct_values[-1]
        if "Unemployment Rate" in out and out["Unemployment Rate"] is None:
            if len(pct_values) > 1:
                out["Unemployment Rate"] = pct_values[-2]
            else:
                out["Unemployment Rate"] = pct_values[-1]

    return out


def merge_feature_updates(current: Dict[str, Any], update: Dict[str, Any], prompt_text: str) -> Dict[str, Any]:
    merged = dict(current)
    prompt_lc = prompt_text.lower()
    for key, value in update.items():
        if value is None:
            continue

        # Avoid overwriting previously captured values with ambiguous unlabeled percentages.
        if key == "Unemployment Rate" and merged.get(key) is not None:
            if "unemployment" not in prompt_lc and "jobless" not in prompt_lc:
                continue
        merged[key] = value
    return merged


def validate_features(features: Dict[str, Any], required_features: List[str]) -> Tuple[Dict[str, float], List[str]]:
    normalized: Dict[str, float] = {}
    missing: List[str] = []
    for feature in required_features:
        value = safe_float(features.get(feature))
        if value is None:
            missing.append(feature)
            continue
        low, high = FEATURE_RANGES.get(feature, (-1_000_000.0, 1_000_000.0))
        if value < low or value > high:
            raise ValueError(f"Feature '{feature}' out of range [{low}, {high}].")
        normalized[feature] = value
    return normalized, missing


def run_prompt_sequence(prompts: List[str], required_features: List[str]) -> Tuple[Dict[str, float], List[str]]:
    merged: Dict[str, Any] = {f: None for f in required_features}
    for prompt in prompts:
        parsed = parse_features_from_text(prompt, required_features)
        merged = merge_feature_updates(merged, parsed, prompt)
    return validate_features(merged, required_features)


def infer_from_prompt_sequence(
    prompts: List[str],
    required_features: List[str],
    predictor: Callable[[Dict[str, float]], float],
) -> float:
    features, missing = run_prompt_sequence(prompts, required_features)
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return predictor(features)
