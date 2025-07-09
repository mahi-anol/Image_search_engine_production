import os
from datetime import date
import torch

### For MongoDB connection
DATABASE_NAME = "image_database"
COLLECTION_NAME = "images"

### MLFLOW
MLFLOW_TRACKING_SERVER="https://dagshub.com/Mahi-Anol/Image_search_engine_production.mlflow"


###Image realted constants
IMAGE_SIZE = 224
BATCH_SIZE = 32


### others
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###make dir if doesn't exists.
os.makedirs('./faiss_index',exist_ok=True)
INDEX_PATH = "./faiss_index/faiss_image_index.index"
MAP_PATH = "./faiss_index/faiss_index_metadata_mapping.pkl"
MODEL_PATH = "./checkpoints/cross_entropy_best.pt"
CLASS_FILE = "./data/processed/classes.json"


FAISS_INDEX_DIM=256