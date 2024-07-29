import os
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get MongoDB URI from environment variable
mongodb_uri = os.getenv('MONGODB_URI')

# Connect to MongoDB
client = MongoClient(mongodb_uri)
db = client['Food_Recognition']
collection = db['images_and_categories']

# Fetch data
data = list(collection.find())
df = pd.DataFrame(data)

# Handle empty dataframe case
if df.empty:
    print("No data found in the collection.")
else:
    # Save to CSV
    df.to_csv('food_data.csv', index=False)
    print("Data successfully saved to food_data.csv")
