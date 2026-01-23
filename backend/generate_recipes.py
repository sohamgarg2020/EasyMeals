from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi import HTTPException
from typing import Dict
from agents.rewording import extract_recipe

# Load model once at module import
print("Loading recipe generation model...")
MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)
print("Model loaded successfully!")


def generate_recipe(ingredients: Dict[str, int]):
    """
    Generate recipe from ingredients dictionary
    
    Args:
        ingredients: Dictionary mapping ingredient names to quantities
        
    Returns:
        Dictionary containing title, ingredients list, directions, and detected items
    """
    if not ingredients:
        raise HTTPException(status_code=400, detail="No ingredients provided")
    
    # Create input text from ingredients
    ingredient_text = ", ".join(ingredients.keys())
    input_text = f"ingredients: {ingredient_text}"
    
    print(f"Input to model: {input_text}")
    
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate recipe
    output_ids = model.generate(
        **inputs,
        max_length=256,
        min_length=64,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )

    recipe = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated recipe: {recipe}")

    return extract_recipe(recipe)

    """
    return {
        "title": title,
        "ingredients": ingredients_list,
        "directions": directions,
        "detected_items": dict(ingredients)
    }
    """