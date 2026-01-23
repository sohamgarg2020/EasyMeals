from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import re

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.6
)

PROMPT = """
You are an AI assistant that parses recipe text into structured JSON format.

Your task is to:
1. Extract the title, ingredients, and directions from the raw recipe text
2. Format ingredients as a clean list (one ingredient per list item)
3. Format directions as numbered steps (one step per list item)
4. Improve grammar and readability without changing the recipe content

INPUT FORMAT:
You will receive recipe text in this format:
"title: [recipe name] ingredients: [ingredient list] directions: [cooking steps]"

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
    "title": "Recipe Name",
    "ingredients": [
        "ingredient 1",
        "ingredient 2",
        "ingredient 3"
    ],
    "directions": [
        "Step 1 description",
        "Step 2 description",
        "Step 3 description"
    ]
}}

RULES:
- Do NOT create or add new ingredients or steps
- Do NOT change cooking methods or quantities
- DO fix grammar, capitalization, and punctuation
- DO separate combined steps into individual steps when appropriate
- DO keep ingredient measurements and amounts intact

EXAMPLE INPUT:
"title: vegetable kabobs ingredients: 1 large carrot 1 cup broccoli, steamed 1 cup peas, steamed 1 cup asparagus, steamed directions: slice carrot and broccoli into strips about 1/2 inch wide and 1/2 inch long. thread onto skewers, alternating with peas and asparagus. grill on a preheated grill, over medium heat, until vegetables are tender and lightly browned."

EXAMPLE OUTPUT:
{{
    "title": "Vegetable Kabobs",
    "ingredients": [
        "1 large carrot",
        "1 cup broccoli, steamed",
        "1 cup peas, steamed",
        "1 cup asparagus, steamed"
    ],
    "directions": [
        "Slice carrot and broccoli into strips about 1/2 inch wide and 1/2 inch long.",
        "Thread onto skewers, alternating with peas and asparagus.",
        "Grill on a preheated grill over medium heat until vegetables are tender and lightly browned."
    ]
}}

Return ONLY the JSON object, no additional text or markdown formatting.

RECIPE TEXT TO PARSE:
{recipe_text}
"""

def extract_recipe(recipe: str):
    """
    Extract and format recipe from raw text using GPT-4
    
    Args:
        recipe: Raw recipe text string
        
    Returns:
        Dictionary with structured recipe data (title, ingredients, directions)
    """
    try:
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template(PROMPT)
        
        # Create chain
        concept_chain = prompt_template | llm
        
        # Invoke the chain
        response = concept_chain.invoke({"recipe_text": recipe})
        
        # Extract the content from the response
        response_text = response.content
        
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        
        # Parse JSON
        recipe_data = json.loads(response_text)
        
        return recipe_data
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response text: {response_text}")
        return {
            "title": "Error parsing recipe",
            "ingredients": [],
            "directions": []
        }
    except Exception as e:
        print(f"Error extracting recipe: {e}")
        return {
            "title": "Error",
            "ingredients": [],
            "directions": []
        }
    
if __name__ == "__main__":
    # Test the function
    test_recipe = "title: vegetable kabobs ingredients: 1 large carrot 1 cup broccoli, steamed 1 cup peas, steamed 1 cup asparagus, steamed directions: slice carrot and broccoli into strips about 1/2 inch wide and 1/2 inch long. thread onto skewers, alternating with peas and asparagus. grill on a preheated grill, over medium heat, until vegetables are tender and lightly browned."
    
    result = extract_recipe(test_recipe)
    print(json.dumps(result, indent=2))