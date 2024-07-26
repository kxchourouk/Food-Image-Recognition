from pymongo import MongoClient
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Replace with your MongoDB connection details
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database']
collection = db['images_and_categories']

class FoodItem(BaseModel):
    category: str

class NutritionResponse(BaseModel):
    category: str
    ingredients: List[str]
    calories: float

def get_nutrition_from_db(category: str) -> Dict:
    # Query the database based on the category
    food = collection.find_one({"category": category})
    if not food:
        raise HTTPException(status_code=404, detail="Category not found in database")
    
    return {
        "category": food.get('category'),
        "ingredients": food.get('ingredients', []),
        "calories": food.get('calories', 0)
    }

@app.post("/nutrition/")
async def get_nutrition(food_items: List[FoodItem]):
    nutrition_data = []
    for item in food_items:
        try:
            data = get_nutrition_from_db(item.category)
            nutrition_data.append(data)
        except HTTPException as e:
            return {"error": str(e)}
    
    return nutrition_data
