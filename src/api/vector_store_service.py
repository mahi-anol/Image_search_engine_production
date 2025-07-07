from torch.utils.data import DataLoader
from src.utils.api_utils import GridFSDataset
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import os
import json
import faiss
import pickle
import uuid
import sys
from PIL import Image
import io
from torchvision import models, transforms
from src.connections.mongodb_connection import MongoDBClient 
import gridfs
from src.model import return_model
from dotenv import load_dotenv
from src.logger import logging
from src.exception import MyException
from src.constants import DATABASE_NAME,COLLECTION_NAME,IMAGE_SIZE,BATCH_SIZE,DEVICE,INDEX_PATH,MODEL_PATH,CLASS_FILE,MAP_PATH,FAISS_INDEX_DIM

# ========== CONFIG & INIT ==========
load_dotenv()

app = FastAPI()

mongo_uri = os.getenv("MONGODB_connection_string")
client = MongoDBClient(DATABASE_NAME)
db=client.database
fs = gridfs.GridFS(db, collection=COLLECTION_NAME)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def get_embedding_model():
    with open(CLASS_FILE, "r") as f:
        classes = json.load(f)["classes"]
    num_classes = len(classes)

    weights = models.MobileNet_V2_Weights.DEFAULT
    base_model = models.mobilenet_v2(weights=weights)
    model = return_model(base_model, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model'])
    model.to(DEVICE)
    model.eval()
    return model

# Load or initialize FAISS index and metadata
if os.path.exists(INDEX_PATH):
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(MAP_PATH, "rb") as f:
        id_mapping = pickle.load(f)
else:
    faiss_index = faiss.IndexFlatL2(FAISS_INDEX_DIM)
    id_mapping = []

# ========== HELPERS ==========
def get_embedding(image_bytes,id=None):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            model=get_embedding_model()
            embedding = model(tensor)[0].cpu().numpy().astype('float32')
    except Exception as e:
        if id==None:
            logging.error("Faced error during generating embedding for image with id: ", id,"The error was the following: ",e,"\n None means it most proababilly comes from search endpoint.")
        else:
            logging.error("Faced error during generating embedding for image with id: ", id,"The error was the following: ",e)
    else:
        return embedding

def save_index():
    try:
        faiss.write_index(faiss_index, INDEX_PATH)
        with open(MAP_PATH, "wb") as f:
            pickle.dump(id_mapping, f)
    except Exception as e:
        logging.error("During saving faiss index faced the following error: ",e)

# ========== MODELS ==========
class DeleteRequest(BaseModel):
    id: str

# ========== ENDPOINTS ==========

@app.get('/')
def Image_vector_search_service():
    return {'message':"Image vector search service is available for use. "}


@app.post("/update")
async def update_index(file: UploadFile = File(...), id: str = None):
    try: 
        logging.error("Trying to update the index....")
        content = await file.read()
        embedding = get_embedding(content,id)
        faiss_index.add(embedding)
        id_mapping.append(id or str(uuid.uuid4()))
        save_index()
    except Exception as e:
        logging.error(f"During updating faiss index for id: {id}, faced the following error:\n {e}")
    else:
        logging.info(f"Image added to index. id: {id_mapping[-1]}")
        return {"message": "Image added to index", "id": id_mapping[-1]}

@app.post("/build-index")
def build_index():
    try:
        logging.info("Trying to create faiss index...")
        dataset = GridFSDataset(fs, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        all_embeddings, all_ids = [], []
        model.eval()
        with torch.inference_mode():
            model=get_embedding_model()
            for imgs, id in dataloader:
                imgs = imgs.to(DEVICE)
                emb = model(imgs)[0].cpu().numpy().astype('float32')
                all_embeddings.append(emb)
                all_ids.extend(id)

        all_embeddings = np.vstack(all_embeddings)
        index = faiss.IndexFlatL2(all_embeddings.shape[1])
        index.add(all_embeddings)
        faiss.write_index(index, INDEX_PATH)
        with open(MAP_PATH, "wb") as f:
            pickle.dump(all_ids, f)

        global faiss_index, id_mapping
        faiss_index, id_mapping = index, all_ids
    except Exception as e:
        logging.error("During building index faced the following error: ",e)
    else:
        logging.info(f"FAISS index built total: {len(all_ids)}")
        return {"message": "FAISS index built", "total": len(all_ids)}

@app.delete("/delete/{id}")
def delete_from_index(id: str):
    try:
        logging.error("Trying to delete from index...")
        global faiss_index, id_mapping
        if id not in id_mapping:
            raise HTTPException(status_code=404, detail="ID not found")
        idx = id_mapping.index(id)
        id_mapping.pop(idx)
        faiss_index.remove_ids(faiss.IDSelectorRange(idx, idx+1))
        save_index()
    except Exception as e:
        logging.error("During deletion from faiss index faced the following error: %s for data corresponding to id: %s",e,id)
    else:
        logging.info(f"ID {id} removed from index")
        return {"message": f"ID {id} removed from index"}

@app.post("/search")
async def search(file: UploadFile = File(...), top_k: int = 5):
    try:
        logging.info("Trying to perform image search")
        content = await file.read()
        query_emb = get_embedding(content,None)
        D, I = faiss_index.search(query_emb, top_k)
        results = [id_mapping[i] for i in I[0]]
    except Exception as e:
        logging.error("During performing image search, faced the following error: ",e)
    else:
        logging.info("Successfully performed vector search for image")
        return {"matches": results}

# ========== RUN INSTRUCTIONS ==========
# Save this file as main.py or run with:
# uvicorn src.api:app --reload
# You may need to set PYTHONPATH or ensure your modules are correctly structured.
# Example:
#   PYTHONPATH=. uvicorn src.api.vector_store_service:app --reload
