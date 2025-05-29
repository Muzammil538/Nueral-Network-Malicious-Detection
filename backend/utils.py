import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from preprocess.char_tokenizer import char_tokenizer
from preprocess.tfidf_to_image import generate_image_tensor
from models.text_model import TextCNN
from models.image_model import ImageCNN
from models.fusion_model import FusionModel



# Load models
text_model = TextCNN()
image_model = ImageCNN()
fusion_model = FusionModel(text_model, image_model)

fusion_model.load_state_dict(torch.load("../saved_models/fusion_model.pth", map_location='cpu'))
fusion_model.eval()

def predict_url(url):
    text_tensor = torch.tensor([char_tokenizer(url)], dtype=torch.long) 
    image_tensor, image_filename = generate_image_tensor(url)

    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        output = fusion_model(text_tensor, image_tensor)
        prediction = "malicious" if output.item() > 0.5 else "benign"
        return prediction, output.item(), image_filename
