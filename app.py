# backend/app.py
import os
import re
import tempfile
from typing import List, Dict, Any
from collections import OrderedDict

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

print("🔧 Starting Vyoora backend...")

# --- OpenAI (new SDK) for chat + classification ---
from openai import OpenAI

# --- EasyOCR for OCR ---
try:
    import easyocr
    OCR_IMPORT_OK = True
except Exception as e:
    OCR_IMPORT_OK = False
    OCR_IMPORT_ERR = str(e)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("🔑 OPENAI_API_KEY present?", bool(OPENAI_API_KEY))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Clients
oai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

reader = None
if OCR_IMPORT_OK:
    try:
        reader = easyocr.Reader(["en"], gpu=False)
        print("✅ EasyOCR ready")
    except Exception as e:
        OCR_IMPORT_OK = False
        OCR_IMPORT_ERR = f"EasyOCR init failed: {e}"
        print("❌", OCR_IMPORT_ERR)
else:
    print("❌ EasyOCR import failed:", OCR_IMPORT_ERR)

# ----------------------------
# OCR → Ingredient extraction helpers
# ----------------------------
IRRELEVANT_HEADINGS = [
    "directions", "usage", "how to use", "warnings", "caution", "attention",
    "store in a cool", "external use only", "keep out of reach", "stop use",
    "if irritation occurs", "discontinue use", "not intended", "consult a physician",
    "avoid contact with eyes", "manufactured by", "distributed by"
]

def slice_after_ingredients(text: str) -> str:
    m = re.search(r"\bingredients?\b\s*[:\-]?", text, flags=re.IGNORECASE)
    return text[m.end():] if m else text

def normalize_token(tok: str) -> str:
    tok = tok.strip(" .:;-_")
    tok = re.sub(r"[^A-Za-z0-9 ,()\-\u2013\u2014/]", "", tok)  # keep typical chars
    tok = re.sub(r"\s+", " ", tok).strip()
    if not tok or len(tok) < 2:
        return ""

    # Title-case alphabetic words; keep acronyms/numbers (BHT, PEG-10) as-is
    words = []
    for w in tok.split():
        if re.fullmatch(r"[A-Z0-9\-(),/]+", w):
            words.append(w)
        else:
            words.append(w.capitalize())
    tok = " ".join(words).strip(", ")
    return tok

def extract_ingredients_from_text(text: str) -> List[str]:
    """Aggressive splitter: commas, semicolons, bullets, middots, newlines, ' and ' and hyphen lists."""
    if not text:
        return []
    # Prefer content after “ingredients”
    text = slice_after_ingredients(text)

    # Cut off non-ingredient sections
    low = text.lower()
    for phrase in IRRELEVANT_HEADINGS:
        p = low.find(phrase)
        if p != -1:
            text = text[:p]
            break

    parts = re.split(r"[;,•·\u2022\u00B7]|\s-\s|\n|\band\b", text, flags=re.IGNORECASE)
    parts = [normalize_token(p) for p in parts]
    parts = [p for p in parts if p and len(p.split()) <= 12]

    # De-dup preserve order
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

# Canonical synonyms
CANON = {
    "Aqua": "Water",
    "Aqua/Water": "Water",
    "Parfum": "Fragrance",
    "Parfum/Fragrance": "Fragrance",
    "Alcohol Denat": "Alcohol Denat.",
    "Tocopherol Acetate": "Tocopheryl Acetate",
}
def canonicalize(name: str) -> str:
    return CANON.get(name, name)

# ----------------------------
# Local rules (fast, $0) + cache
# ----------------------------
LOCAL_RULES = {
    "Water": ("Safe", "Solvent/base in most formulas."),
    "Aqua": ("Safe", "Solvent/base in most formulas."),
    "Glycerin": ("Safe", "Humectant that draws moisture to the skin."),
    "Hyaluronic Acid": ("Safe", "Hydrating humectant; generally well-tolerated."),
    "Niacinamide": ("Safe", "Supports barrier and brightening with low irritation risk."),
    "Fragrance": ("Irritant", "Common sensitizer; may irritate sensitive skin."),
    "Parfum": ("Irritant", "Common sensitizer; may irritate sensitive skin."),
    "Alcohol Denat.": ("Irritant", "Can be drying/irritating at higher levels."),
    "Coconut Oil": ("Comedogenic", "Can clog pores for acne-prone skin."),
    "Isopropyl Myristate": ("Comedogenic", "Associated with pore clogging."),
    "Cetyl Alcohol": ("Safe", "Fatty alcohol; generally non-irritating."),
    "Stearyl Alcohol": ("Safe", "Fatty alcohol; generally non-irritating."),
    "Phenoxyethanol": ("Limited Data", "Preservative; considered safe at low levels."),
    "Parabens": ("Limited Data", "Preservatives; debated but low irritation risk."),
    "Tocopheryl Acetate": ("Safe", "Vitamin E derivative; antioxidant."),
    "Tocopherol": ("Safe", "Vitamin E; antioxidant."),
    "Salicylic Acid": ("Safe", "BHA exfoliant; may irritate sensitive skin."),
    "Glycolic Acid": ("Safe", "AHA exfoliant; may irritate sensitive skin."),
}

AI_CACHE_MAX = 500
AI_CACHE: "OrderedDict[str, tuple[str,str]]" = OrderedDict()

def cache_get(name: str):
    key = name.lower().strip()
    if key in AI_CACHE:
        AI_CACHE.move_to_end(key)
        return AI_CACHE[key]
    return None

def cache_put(name: str, cls: str, reason: str):
    key = name.lower().strip()
    AI_CACHE[key] = (cls, reason)
    AI_CACHE.move_to_end(key)
    if len(AI_CACHE) > AI_CACHE_MAX:
        AI_CACHE.popitem(last=False)

# ----------------------------
# AI classification (no Unknowns)
# ----------------------------
def ai_clean_and_classify(ingredients: List[str]) -> List[Dict[str, Any]]:
    """Local rules + cache first. Remaining items -> single OpenAI call. No 'Unknowns'."""
    if not ingredients:
        return []

    # 1) Apply local rules + cache
    results: List[Dict[str, Any]] = []
    to_ai: List[str] = []
    for ing in ingredients:
        if ing in LOCAL_RULES:
            cls, rsn = LOCAL_RULES[ing]
            cache_put(ing, cls, rsn)
            results.append({"ingredient": ing, "classification": cls, "reason": rsn})
            continue
        cached = cache_get(ing)
        if cached:
            cls, rsn = cached
            results.append({"ingredient": ing, "classification": cls, "reason": rsn})
            continue
        to_ai.append(ing)

    # 2) If nothing left or no client/key, return what we have + Limited Data for rest
    if not to_ai:
        # Keep original order
        order = {name: i for i, name in enumerate(ingredients)}
        results.sort(key=lambda x: order.get(x["ingredient"], 10**6))
        return results

    if not (oai_client and OPENAI_API_KEY):
        for ing in to_ai:
            results.append({"ingredient": ing, "classification": "Limited Data", "reason": "Not enough reliable data."})
        order = {name: i for i, name in enumerate(ingredients)}
        results.sort(key=lambda x: order.get(x["ingredient"], 10**6))
        return results

    # 3) Ask OpenAI once for the remaining items
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ingredient": {"type": "string"},
                        "classification": {"type": "string", "enum": ["Safe", "Irritant", "Comedogenic", "Limited Data"]},
                        "reason": {"type": "string"}
                    },
                    "required": ["ingredient", "classification", "reason"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["items"],
        "additionalProperties": False
    }

    sys = (
        "You are Vyoora, a concise skincare ingredient expert. "
        "Return ONLY real cosmetic ingredients with a 1-sentence reason. "
        "If evidence is mixed or unclear, use 'Limited Data' (do NOT invent)."
    )
    user = "Classify these:\n" + "\n".join(f"- {x}" for x in to_ai[:60])

    try:
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ingredient_classifications", "schema": schema, "strict": True},
            },
        )
        import json
        data = resp.choices[0].message.content
        parsed = data if isinstance(data, dict) else json.loads(data)
        items = parsed.get("items", [])

        for it in items:
            ing = str(it.get("ingredient", "")).strip()
            cls = str(it.get("classification", "Limited Data")).strip()
            rsn = str(it.get("reason", "Brief reason unavailable.")).strip()
            if not ing:
                continue
            if cls not in {"Safe", "Irritant", "Comedogenic", "Limited Data"}:
                cls = "Limited Data"
            cache_put(ing, cls, rsn)
            results.append({"ingredient": ing, "classification": cls, "reason": rsn})
    except Exception as e:
        # Fallback: Limited Data for those we couldn't classify
        for ing in to_ai:
            cls, rsn = "Limited Data", "Automatic fallback. Not enough reliable data."
            cache_put(ing, cls, rsn)
            results.append({"ingredient": ing, "classification": cls, "reason": rsn})

    # Keep original order
    order = {name: i for i, name in enumerate(ingredients)}
    results.sort(key=lambda x: order.get(x["ingredient"], 10**6))
    return results

# ----------------------------
# Routes
# ----------------------------
@app.get("/ping")
def ping():
    return jsonify({"ok": True, "service": "flask", "chat": "/chat"}), 200

@app.post("/upload")
def upload():
    """
    Multipart form-data: 'image' or 'file'
    -> OCR -> aggressive split -> canonicalize/dedup -> local rules/cache -> OpenAI classify
    Returns: {"ingredients": [{"ingredient","classification","reason"}], "count": N}
    """
    if not OCR_IMPORT_OK or reader is None:
        return jsonify({"error": f"OCR unavailable: {OCR_IMPORT_ERR}"}), 500

    f = request.files.get("image") or request.files.get("file")
    if not f:
        return jsonify({"error": "No image provided (expected field 'image' or 'file')"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.filename)[-1] or ".jpg") as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Two OCR passes for robustness
        lines_line = reader.readtext(tmp_path, detail=0, paragraph=False)
        lines_para = reader.readtext(tmp_path, detail=0, paragraph=True)
        raw_text = ("\n".join(lines_line or []) + "\n" + " ".join(lines_para or [])).strip()

        extracted = extract_ingredients_from_text(raw_text)

        # If extraction too small, fallback to comma splitting on paragraph text
        if len(extracted) <= 1 and lines_para:
            fallback_text = slice_after_ingredients(" ".join(lines_para))
            more = [normalize_token(p) for p in re.split(r"[,;\n]+", fallback_text)]
            extracted = [p for p in more if p and len(p.split()) <= 12]

        # Dedup + canonicalize
        dedup, seen = [], set()
        for x in extracted:
            cx = canonicalize(x)
            if cx and cx not in seen:
                dedup.append(cx)
                seen.add(cx)

        # Classify (local rules/cache + AI)
        results = ai_clean_and_classify(dedup)

        return jsonify({"ingredients": results, "count": len(results)}), 200

    except Exception as e:
        return jsonify({"error": f"OCR failed: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Missing 'message'"}), 400

    if not oai_client or not OPENAI_API_KEY:
        return jsonify({"reply": "Server is missing OPENAI_API_KEY. Add it to .env and restart."}), 200

    try:
        completion = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=[
                {"role": "system",
                 "content": "You are Vyoora, a concise skincare assistant. Be practical, avoid medical claims, and suggest patch testing."},
                {"role": "user", "content": user_message},
            ],
        )
        reply_text = completion.choices[0].message.content.strip()
        return jsonify({"reply": reply_text}), 200
    except Exception:
        return jsonify({"reply": "Sorry, I couldn’t process that just now. Try again in a moment."}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
