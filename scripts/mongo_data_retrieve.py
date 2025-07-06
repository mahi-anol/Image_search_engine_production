from PIL import Image
import io
import os
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv
from src.constants import DATABASE_NAME,COLLECTION_NAME
load_dotenv()

mongo_connection_string=os.getenv("MongoDB_connection_string")
client=MongoClient(mongo_connection_string)
db = client[DATABASE_NAME]
fs = gridfs.GridFS(db,collection=COLLECTION_NAME)

# Example: retrieve image by filename
filename = "0AJ1EAM5WZWTEQ_2.jpg"
grid_out = fs.find_one({"filename": filename})

if grid_out:
    image_data = grid_out.read()
    
    # Optional: load with PIL
    image = Image.open(io.BytesIO(image_data))
    image.show()  # or image.save("output.jpg")
else:
    print(f"No file found with filename: {filename}")
