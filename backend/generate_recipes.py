from detect import ingredient_list
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re


MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)


def generate_recipe(ingredients):
    ingredient_text = ", ".join(ingredients.keys())
    input_text = f"ingredients: {ingredient_text}"
    inputs = tokenizer(input_text, return_tensors="pt")

    output_ids = model.generate(
        **inputs,
        max_length=200,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

    recipe = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    parts = recipe.split("ingredients:")
    title_part = parts[0].replace("title:", "").strip()
    ingredients_and_directions = parts[1].split("directions:")
    ingredients_part = ingredients_and_directions[0].strip()
    directions_part = ingredients_and_directions[1].strip()

    ingredients = [i.strip() for i in ingredients_part.split() if not i.isdigit()]

    ingredients = re.findall(r"\d+\s[^0-9]+", ingredients_part)

    directions = [step.strip() for step in directions_part.split('.') if step.strip()]

    print(f"Title:\n  {title_part}\n")
    print("Ingredients:")
    for item in ingredients:
        print(f"  - {item}")
    print("\nDirections:")
    for count, step in enumerate(directions):
        print(f"{count+1}. {step}")


if __name__ == "__main__":
    detected = ingredient_list("images.jpg")
    generate_recipe(detected)