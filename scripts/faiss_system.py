import os
from pymongo import MongoClient
import gridfs
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import faiss
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
from src.model import return_model
import json
# ========== CONFIGURATION ==========
load_dotenv()
mongo_connection_string = os.getenv("MONGODB_connection_string")
DB_NAME = "image_database"
COLLECTION_NAME = "images"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAISS_INDEX_FILE = "./faiss_index/faiss_image_index.index"
MAPPING_FILE = "./faiss_index/faiss_index_metadata_mapping.pkl"
TOP_K = 30
IMAGE_SIZE = 224
model_dir="./checkpoints/cross_entropy_best.pt"

# ========== CONNECT TO MONGODB ==========
client = MongoClient(mongo_connection_string)
db = client[DB_NAME]
fs = gridfs.GridFS(db, collection=COLLECTION_NAME)

# ========== TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ========== LOAD IMAGES FROM GRIDFS ==========
def get_images_from_gridfs(fs, transform=None):
    image_data = []
    cursor = fs.find()
    for file in cursor:
        try:
            img_bytes = file.read()
            image = Image.open(BytesIO(img_bytes)).convert('RGB')
            if transform:
                image = transform(image)
            image_data.append((image, file.filename, file.full_path))
        except:
            continue
    return image_data

# ========== CUSTOM DATASET ==========
class GridFSDataset(Dataset):
    def __init__(self, fs, transform=None):
        self.data = get_images_from_gridfs(fs, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, filename, path = self.data[idx]
        return img, filename, path

# ========== EMBEDDING FUNCTION ==========
def get_dataset_embeddings(model, dataloader, device):
    all_embeddings = []
    all_paths = []
    model.eval()
    with torch.inference_mode():
        for imgs, _, paths in tqdm(dataloader, desc='Embedding dataset'):
            imgs = imgs.to(device)
            embeddings = model(imgs)[0].cpu()
            all_embeddings.append(embeddings)
            all_paths.extend(paths)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings, all_paths

# ========== BUILD FAISS INDEX ==========
def build_faiss_index(embeddings, paths, index_file, mapping_file):
    embeddings_np = embeddings.numpy().astype('float32')
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)
    faiss.write_index(index, index_file)
    with open(mapping_file, "wb") as f:
        pickle.dump(paths, f)
    print(f"FAISS index and metadata saved.")

# ========== SHOW SIMILAR IMAGES ==========
def show_images_from_gridfs(fs, matched_paths):
    images = []
    for path in matched_paths:
        file = fs.find_one({'full_path': path})
        if file:
            img = Image.open(BytesIO(file.read())).convert("RGB")
            images.append(img)

    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis("off")
    plt.show()

# ========== QUERY FUNCTION ==========
def query_image_and_show_results(query_img_path, model, fs, index_file, mapping_file, top_k=5):
    # Load index + metadata
    index = faiss.read_index(index_file)
    with open(mapping_file, "rb") as f:
        metadata_mapping = pickle.load(f)

    # Transform query image
    img = Image.open(query_img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Get embedding
    with torch.inference_mode():
        query_emb = model(img_tensor)[0].cpu().numpy().astype('float32')

    # Search
    D, I = index.search(query_emb, top_k)
    matched_paths = [metadata_mapping[i] for i in I[0]]
    print("Matched paths:", matched_paths)
    show_images_from_gridfs(fs, matched_paths)

# ========== MAIN ==========
def main():

    print("Loading model...")
    # -------------------------------
    # Load Model
    # -------------------------------
    # Load the JSON file
    with open("./data/processed/classes.json", "r") as f:
        data = json.load(f)
    
    # Extract the classes
    classes = data["classes"]

    # Get number of classes
    num_of_class = len(classes)

    model_weights = models.MobileNet_V2_Weights.DEFAULT
    auto_transforms = model_weights.transforms()
    b_model = models.mobilenet_v2(weights=model_weights)
    final_model = return_model(b_model, num_of_class)
    final_model.to(DEVICE)

    checkpoint = torch.load(model_dir, map_location=DEVICE)
    final_model.load_state_dict(checkpoint['model'])
    final_model.eval()

    if not os.path.exists(FAISS_INDEX_FILE) and not os.path.exists(MAPPING_FILE):
        print("Loading dataset from MongoDB...")
        dataset = GridFSDataset(fs, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        print("Generating embeddings...")
        embeddings, paths = get_dataset_embeddings(final_model, dataloader, DEVICE)

        print("Building FAISS index...")
        build_faiss_index(embeddings, paths, FAISS_INDEX_FILE, MAPPING_FILE)

    # Optional: Query and show top-k
    query_image_path = r"E:\Image_search_engine_production\data\raw\Beds\0AJ1EAM5WZWTEQ_2.jpg"  # ← UPDATE THIS
    print("Querying image...")
    query_image_and_show_results(query_image_path, final_model, fs, FAISS_INDEX_FILE, MAPPING_FILE, TOP_K)

if __name__ == "__main__":
    main()
