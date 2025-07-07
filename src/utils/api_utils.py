from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
from src.logger import logging
# ========== LOAD IMAGES FROM GRIDFS ==========
def get_images_from_gridfs(fs, transform=None):
    image_data = []
    cursor = fs.find()
    for file in cursor:
        try:
            logging.info("Adding image with image id: %s which will be used for creating gridFS dataset",file._id)
            img_bytes = file.read()
            image = Image.open(BytesIO(img_bytes)).convert('RGB')
            if transform:
                image = transform(image)
            image_data.append((image,str(file._id)))
        except:
            logging.error(f"Faced error during adding image data for {file._id}. so skipping....")
            continue
    return image_data


class GridFSDataset(Dataset):
    def __init__(self, fs, transform=None):
        self.data = get_images_from_gridfs(fs, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, file_id = self.data[idx]
        return img, file_id