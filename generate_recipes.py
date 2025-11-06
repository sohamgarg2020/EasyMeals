from detect import ingredient_list
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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

    print("\n--- Generated Recipe ---")
    print(recipe)
    return recipe


if __name__ == "__main__":
    detected = ingredient_list("fridge.jpg")
    generate_recipe(detected)