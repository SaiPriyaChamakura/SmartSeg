import os
import requests
from typing import List, Dict

HATZ_API_BASE = os.getenv("HATZ_API_BASE", "https://ai.hatz.ai")
HATZ_API_KEY = os.getenv("HATZ_API_KEY", "")

def generate_recommendations(items: List[Dict], model: str = "gpt-4o-mini") -> List[str]:
    """
    items: list of dicts with keys: SKU, Segment, Annual_Value, CV.
    Returns: list of recommendation strings aligned with items order.
    """
    if not HATZ_API_KEY:
        return ["[Set HATZ_API_KEY to enable AI recommendations]"] * len(items)

    header = "You are an inventory expert. For each row, propose policy: review frequency, safety stock approach, supplier strategy, and a 1-2 line rationale."
    rows = ["SKU | Segment | Annual_Value | CV"] + [
        f"{x['SKU']} | {x['Segment']} | {x['Annual_Value']:.2f} | {x['CV']:.2f}" for x in items
    ]
    prompt = header + "\n" + "\n".join(rows) + "\nReturn results as bullet points numbered 1..N aligned to the rows."

    url = f"{HATZ_API_BASE}/v1/chat/completions"
    headers = {"X-API-Key": HATZ_API_KEY, "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("choices",[{}])[0].get("message",{}).get("content","").strip()
    lines = [ln.strip("- ").strip() for ln in text.split("\n") if ln.strip()]
    return lines if lines else [text]