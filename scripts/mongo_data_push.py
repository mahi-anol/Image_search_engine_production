import os
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

load_dotenv()

mongo_connection_string=os.getenv("MONGODB_connection_string")

client=MongoClient(mongo_connection_string)
db = client["image_database"]
fs = gridfs.GridFS(db,collection="images")


# Set your root directory here
root_folder = r"D:\Image_Search_Engine\image_search_data_sorted"

# Allowed image file extensions
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# Walk through the folder
for root, dirs, files in os.walk(root_folder):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, root_folder)
            subfolder = os.path.dirname(relative_path)

            with open(full_path, "rb") as f:
                # Store in GridFS with metadata
                file_id = fs.put(
                    f,
                    filename=file,
                    subfolder=subfolder,
                    full_path=relative_path,
                )
                print(f"Stored: {relative_path} as file_id {file_id}")