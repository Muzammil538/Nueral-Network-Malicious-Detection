import torch.nn as nn
import torch

class FusionModel(nn.Module):
    def __init__(self, text_model, image_model):
        super(FusionModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, text_input, image_input):
        text_feat = self.text_model(text_input)
        image_feat = self.image_model(image_input)
        combined = torch.cat((text_feat, image_feat), dim=1)
        return self.classifier(combined)
