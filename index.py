import json
import openai
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import easyocr

# Load .env for OpenAI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup Flask
app = Flask(__name__)
CORS(app)

# EasyOCR setup
reader = easyocr.Reader(['en'])

# Load ingredient safety data
with open('ingredient_safety_dataset.json') as f:
    safety_data = json.load(f)

# GPT fallback with explanation
def gpt_guess_safety_with_reason(ingredient):
    try:
        prompt = (
            f"Classify the ingredient '{ingredient}' for skincare as Safe, Irritant, Comedogenic, or Unknown.\n"
            f"Then explain the reasoning behind the classification in 1 sentence.\n"
            f"Respond in JSON with keys 'safety' and 'reason'."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response['choices'][0]['message']['content']
        parsed = json.loads(content)
        return parsed.get('safety', 'Unknown'), parsed.get('reason', 'No explanation provided.')
    except Exception as e:
        print(f"⚠️ GPT fallback error for '{ingredient}':", str(e))
        return "Unknown", "Failed to generate explanation."

# Main ingredient analysis
def analyze_ingredient_safety(ingredients):
    results = []
    for ingredient in ingredients:
        label = safety_data.get(ingredient.lower())
        if label:
            results.append({
                'ingredient': ingredient,
                'safety': label,
                'reason': 'Matched from database.'
            })
        else:
            gpt_label, reason = gpt_guess_safety_with_reason(ingredient)
            results.append({
                'ingredient': ingredient,
                'safety': gpt_label,
                'reason': reason
            })
    return results

# 🧠 Chatbot endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '')

        prompt = (
            f"You are an AI skincare assistant. Answer user queries clearly and concisely.\n"
            f"User: {question}\nAI:"
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        answer = response['choices'][0]['message']['content'].strip()
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"❌ Chatbot error: {e}")
        return jsonify({"answer": "Sorry, I couldn’t process your question."})

# 📷 OCR Upload route
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)

    print(f"🔍 Image saved at: {image_path}")
    raw_text = reader.readtext(image_path, detail=0)
    print(f"📷 EasyOCR Raw Result: {raw_text}")

    # Basic ingredient cleaning
    text = ' '.join(raw_text)
    ingredients = [
        i.strip(' ,;:')
        for i in text.upper().split()
        if 3 < len(i.strip(' ,;:')) <= 30  # basic length check
    ]

    results = analyze_ingredient_safety(ingredients)
    return jsonify({'ingredients': results})

# ✅ Start server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
