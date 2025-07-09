from torch.utils.data import DataLoader
from src.utils.api_utils import GridFSDataset
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException,Form
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
from fastapi.middleware.cors import CORSMiddleware
from src.constants import DATABASE_NAME,COLLECTION_NAME,IMAGE_SIZE,BATCH_SIZE,DEVICE,INDEX_PATH,MODEL_PATH,CLASS_FILE,MAP_PATH,FAISS_INDEX_DIM
from time import time
import gc

# ========== CONFIG & INIT ==========
load_dotenv()

app = FastAPI()


# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]
# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


mongo_uri = os.getenv("MONGODB_connection_string")
client = MongoDBClient(DATABASE_NAME)
db=client.database
fs = gridfs.GridFS(db, collection=COLLECTION_NAME)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

### loading model with method
# def get_embedding_model():
#     logging.info("Loading model for inference...")
#     try:
#         st=time()
#         with open(CLASS_FILE, "r") as f:
#             classes = json.load(f)["classes"]
#         num_classes = len(classes)

#         weights = models.MobileNet_V2_Weights.DEFAULT
#         base_model = models.mobilenet_v2(weights=weights)
#         model = return_model(base_model, num_classes)
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model'])
#         model.to(DEVICE)
#         model.eval()
#         end=time()

#         ### avoiding memory leaks
#         del weights
#         del base_model
#         gc.collect()
#     except Exception as e:
#         logging.error("Faced the following error during loading the model: \n%s",e)
#     else:
#         logging.info("Sucessfully loaded the model")
#         logging.info(f"Time taken to load the model: {end-st} sec")
#         return model




#### loading model
with open(CLASS_FILE, "r") as f:
    classes = json.load(f)["classes"]
num_classes = len(classes)

weights = models.MobileNet_V2_Weights.DEFAULT
base_model = models.mobilenet_v2(weights=weights)
model = return_model(base_model, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model'])
model.to(DEVICE)
model.eval()
del weights
del base_model
gc.collect()



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
        logging.info("Invoked get_embedding method to generate embedding for the search query (image).")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            # model=get_embedding_model()
            st=time()
            embedding = model(tensor)[0].cpu().numpy().astype('float32')
        end=time()


        # ### avoid memory leaks
        # del model
        # gc.collect()

    except Exception as e:
        if id==None:
            logging.error("Faced error during generating embedding for image with id: ", id,"The error was the following: ",e,"\n None means it most proababilly comes from search endpoint.")
        else:
            logging.error("Faced error during generating embedding for image with id: ", id,"The error was the following: ",e)
    else:
        logging.info("Generated embedding for the search query (image)")
        logging.info(f"Time taken to generate embedding for the search query: {end-st}")
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

        # ### avoid memory leaks
        # gc.collect()
    except Exception as e:
        logging.error(f"During updating faiss index for id: {id}, faced the following error:\n {e}")
    else:
        logging.info(f"Image added to index. id: {id_mapping[-1]}")
        return {"message": "Image added to index", "id": id_mapping[-1]}
    

# @app.post("/bulk-update")
# async def bulk_update_index(
#     files: List[UploadFile] = File(...),
#     ids: List[str] = Form(...)
# ):
#     # You can move this to constants if preferred

#     try:
#         logging.info("Trying to update index with multiple images in batch...")

#         if len(files) != len(ids):
#             raise HTTPException(status_code=400, detail="Number of files and IDs must match.")

#         # Load the model once
#         model = get_embedding_model()
#         model.eval()

#         image_tensors = []
#         new_ids = []
#         skipped_ids = []

#         for file, id in zip(files, ids):
#             if id in id_mapping:
#                 skipped_ids.append(id)
#                 logging.warning(f"Skipping image with duplicate ID: {id}")
#                 continue

#             content = await file.read()
#             try:
#                 image = Image.open(io.BytesIO(content)).convert("RGB")
#                 tensor = transform(image).unsqueeze(0)
#                 image_tensors.append(tensor)
#                 new_ids.append(id or str(uuid.uuid4()))
#             except Exception as e:
#                 logging.warning(f"Skipping invalid image with ID {id}: {e}")

#         if not image_tensors:
#             raise HTTPException(status_code=400, detail="No new valid images to process.")

#         # Process in batches
#         embeddings = []
#         for i in range(0, len(image_tensors), BATCH_SIZE):
#             batch = image_tensors[i:i + BATCH_SIZE]
#             batch_tensor = torch.cat(batch, dim=0).to(DEVICE)
#             with torch.inference_mode():
#                 batch_emb = model(batch_tensor)[0].cpu().numpy().astype('float32')
#             embeddings.append(batch_emb)

#         all_embeddings = np.vstack(embeddings)
#         faiss_index.add(all_embeddings)
#         id_mapping.extend(new_ids)
#         save_index()

#         logging.info(f"Added {len(new_ids)} new images to the index.")
#         if skipped_ids:
#             logging.info(f"Skipped {len(skipped_ids)} images with duplicate IDs.")

#     except Exception as e:
#         logging.error(f"Error during batch bulk update: {e}")
#         raise HTTPException(status_code=500, detail="Bulk update failed.")
#     else:
#         return {
#             "message": f"{len(new_ids)} new images added to index.",
#             "added_ids": new_ids,
#             "skipped_ids": skipped_ids
#         }



@app.post("/build-index-from-mongo")
def build_index_from_mongoDB_cloud():
    try:
        logging.info("Trying to create faiss index for the entire database...")

        st=time()
        dataset = GridFSDataset(fs, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        all_embeddings, all_ids = [], []
        model.eval()
        logging.info("Generating Embedding for the entire image database....")
        st=time()
        with torch.inference_mode():
            # model=get_embedding_model()
            for imgs, id in dataloader:
                imgs = imgs.to(DEVICE)
                emb = model(imgs)[0].cpu().numpy().astype('float32')
                all_embeddings.append(emb)
                all_ids.extend(id)
        
        end=time()
        logging.info("Finised generating Embedding for the entire image database")
        logging.info(f"Time taken to generate the embeddings: {end-st} sec")
        
        st=time()
        logging.info("Creating index from the enire image database embedding....")
        all_embeddings = np.vstack(all_embeddings)
        index = faiss.IndexFlatL2(all_embeddings.shape[1])
        index.add(all_embeddings)
        faiss.write_index(index, INDEX_PATH)
        with open(MAP_PATH, "wb") as f:
            pickle.dump(all_ids, f)

        global faiss_index, id_mapping
        faiss_index, id_mapping = index, all_ids
        end=time()

        logging.info("successfully created index from the entire image database embedding....")
        logging.info(f"Time taken to create the index for the entire database: {end-st} sec")
        
        # ###avoid memory leaks
        # del model
        # gc.collect()
    except Exception as e:
        logging.error("During building index faced the following error: ",e)
    else:
        logging.info(f"FAISS index built for total: {len(all_ids)} elements")
        return {"message": "FAISS index built operation succesful ", "total": len(all_ids)}

@app.delete("/delete/{id}")
def delete_from_index(id: str):
    try:
        logging.info("Trying to delete from index...")
        st=time()
        global faiss_index, id_mapping
        if id not in id_mapping:
            raise HTTPException(status_code=404, detail="ID not found")
        idx = id_mapping.index(id)
        id_mapping.pop(idx)
        faiss_index.remove_ids(faiss.IDSelectorRange(idx, idx+1))
        save_index()
        end=time()
    except Exception as e:
        logging.error("During deletion from faiss index faced the following error: %s for data corresponding to id: %s",e,id)
    else:
        logging.info(f"ID {id} removed from index")
        logging.info(f"Total time taken to remove the index: {end-st} sec" )
        return {"message": f"ID {id} removed from index"}

@app.post("/search")
async def search(file: UploadFile = File(...), top_k: int = 5):
    try:
        logging.info("Trying to perform image search")
        content = await file.read()
        query_emb = get_embedding(content,None)
        st=time()
        D, I = faiss_index.search(query_emb, top_k)
        results = [id_mapping[i] for i in I[0]]
        end=time()
        # #### avoid memory leaks
        # gc.collect()
    except Exception as e:
        logging.error("During performing image search, faced the following error: ",e)
    else:
        logging.info("Successfully performed vector search for image")
        logging.info(f"Total time taken to perform the search: {end-st} sec")
        return {"matches": results}

# ========== RUN INSTRUCTIONS ==========
# Save this file as main.py or run with:
# uvicorn src.api:app --reload
# You may need to set PYTHONPATH or ensure your modules are correctly structured.
# Example:
#   PYTHONPATH=. uvicorn src.api.search_system:app --reload
