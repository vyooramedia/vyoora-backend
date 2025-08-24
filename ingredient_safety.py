import json
import os

# Load the safety dataset once
with open('ingredient_safety_dataset.json', 'r') as f:
    safety_data = json.load(f)

def analyze_ingredient_safety(ingredient_list, skin_type='normal'):
    results = []

    for ing in ingredient_list:
        ing_lower = ing.lower()
        matched = False

        for key, entry in safety_data.items():
            if key.lower() in ing_lower:
                matched = True
                results.append({
                    "ingredient": ing,
                    "safety": entry.get("overall_safety", "Unknown"),
                    "skin_type_note": entry.get("skin_type_specific", {}).get(skin_type, "Unknown")
                })
                break

        if not matched:
            results.append({
                "ingredient": ing,
                "safety": "Unknown",
                "skin_type_note": "Unknown"
            })

    return results
