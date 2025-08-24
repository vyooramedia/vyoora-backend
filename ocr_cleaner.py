import re

def clean_ocr_text(text):
    """
    Cleans raw OCR output by:
    - Removing non-ingredient sections (like instructions)
    - Standardizing spacing, punctuation, etc.
    """
    # Remove newline artifacts and convert to lowercase
    text = text.replace('\n', ' ').lower()

    # Remove phrases commonly not part of ingredient list
    irrelevant_phrases = [
        'directions', 'usage', 'warnings', 'store in a cool', 'external use only',
        'keep out of reach', 'stop use', 'if irritation occurs', 'discontinue use',
        'not intended', 'consult a physician', 'avoid contact with eyes'
    ]
    for phrase in irrelevant_phrases:
        text = text.split(phrase)[0]

    # Remove extra punctuation and characters
    text = re.sub(r'[^\w\s,()-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_ingredients(cleaned_text):
    """
    Extracts ingredients by:
    - Splitting by commas
    - Removing numbers or leftover symbols
    - Capitalizing each for consistency
    """
    raw_ingredients = [i.strip() for i in cleaned_text.split(',') if i.strip()]
    
    # Basic filtering and formatting
    ingredients = []
    for ing in raw_ingredients:
        ing = re.sub(r'[^a-zA-Z\s-]', '', ing).strip()
        if len(ing) > 1:
            ingredients.append(ing.title())

    return ingredients
