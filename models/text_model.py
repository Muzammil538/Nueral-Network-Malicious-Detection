import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size=50, embed_dim=128):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 256)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x))).squeeze(2)
        return self.fc(x)
