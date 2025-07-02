from PIL import Image
import io
import os
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv
load_dotenv()

mongo_connection_string=os.getenv("MONGODB_URL")
client=MongoClient(mongo_connection_string)
db = client["image_database"]
fs = gridfs.GridFS(db,collection="images")

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
