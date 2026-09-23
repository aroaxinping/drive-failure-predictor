"""Constants and utilities for feature engineering."""

CRITICAL_SMART = [
    "smart_5_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
]

SMART_RAW_COLS = [
    "smart_1_raw",
    "smart_3_raw",
    "smart_4_raw",
    "smart_5_raw",
    "smart_7_raw",
    "smart_9_raw",
    "smart_10_raw",
    "smart_12_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_190_raw",
    "smart_192_raw",
    "smart_193_raw",
    "smart_194_raw",
    "smart_197_raw",
    "smart_198_raw",
    "smart_199_raw",
    "smart_240_raw",
    "smart_241_raw",
    "smart_242_raw",
]


def extract_manufacturer(model: str) -> str:
    """Extract manufacturer from drive model name."""
    model_lower = str(model).lower()
    if model_lower.startswith("st") or "seagate" in model_lower:
        return "Seagate"
    if model_lower.startswith("wdc") or "western" in model_lower:
        return "WDC"
    if "hgst" in model_lower or model_lower.startswith("hus") or model_lower.startswith("hms"):
        return "HGST"
    if "toshiba" in model_lower:
        return "Toshiba"
    return "Other"
