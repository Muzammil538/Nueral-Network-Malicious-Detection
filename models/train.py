import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_model import TextCNN
from image_model import ImageCNN
from fusion_model import FusionModel
from preprocess.char_tokenizer import char_tokenizer
from preprocess.tfidf_to_image import generate_image_tensor

# ========== CONFIG ==========
DATA_PATH = 'dataset/full_urls.csv'
BATCH_SIZE = 64
EPOCHS = 5
MAX_LEN = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = 'saved_models/'

os.makedirs(SAVE_PATH, exist_ok=True)

# ========== DATASET ==========
class URLDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.data = self.data.dropna()
        self.data['label'] = self.data['label'].astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        url = self.data.iloc[idx]['url']
        label = self.data.iloc[idx]['label']
        text_tensor = torch.tensor(char_tokenizer(url, MAX_LEN), dtype=torch.long)
        image_tensor = generate_image_tensor(url)
        return text_tensor, image_tensor, torch.tensor(label, dtype=torch.float32)

# ========== MODELS ==========
text_model = TextCNN().to(DEVICE)
image_model = ImageCNN().to(DEVICE)
fusion_model = FusionModel(text_model, image_model).to(DEVICE)

# ========== TRAINING ==========
def train():
    dataset = URLDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(fusion_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):
        fusion_model.train()
        total_loss = 0
        for text_input, image_input, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            text_input, image_input, labels = text_input.to(DEVICE), image_input.to(DEVICE), labels.to(DEVICE)

            outputs = fusion_model(text_input, image_input).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    # Save models
    torch.save(text_model.state_dict(), os.path.join(SAVE_PATH, 'text_model.pth'))
    torch.save(image_model.state_dict(), os.path.join(SAVE_PATH, 'image_model.pth'))
    torch.save(fusion_model.state_dict(), os.path.join(SAVE_PATH, 'fusion_model.pth'))
    print("✅ Models saved in 'saved_models/'")

if __name__ == '__main__':
    train()
