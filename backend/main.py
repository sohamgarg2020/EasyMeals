from fastapi import FastAPI, HTTPException, File, UploadFile
from typing import Dict
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os

from detect import ingredient_list
from generate_recipes import generate_recipe

app = FastAPI()

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class RecipeRequest(BaseModel):
    ingredients: Dict[str, int]


@app.get("/")
def read_root():
    return {"message": "Recipe Generator API is running!"}


@app.post("/recipe")
def get_recipe(request: RecipeRequest):
    """
    Endpoint to generate recipe from ingredients
    
    Example request body:
    {
        "ingredients": {
            "chicken": 1,
            "rice": 2,
            "garlic": 3
        }
    }
    """
    try:
        return generate_recipe(request.ingredients)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recipe: {str(e)}")


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image and get a recipe based on detected ingredients
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save the file temporarily
    temp_path = f"temp_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Processing image: {temp_path}")
        
        # Run detection
        detected_ingredients = ingredient_list(temp_path)
        
        # Check if any ingredients were detected
        if not detected_ingredients:
            return {
                "message": "No food items detected in the image",
                "detected_items": {},
                "title": None,
                "ingredients": [],
                "directions": []
            }
        
        print(f"Detected ingredients: {detected_ingredients}")
        
        # Generate recipe from detected ingredients
        recipe_data = generate_recipe(detected_ingredients)
        
        return recipe_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)