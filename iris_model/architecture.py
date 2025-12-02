# components/shared/model_architecture.py
import torch.nn as nn

class IrisArchitecture(nn.Module):
    """Simple PyTorch neural network for Iris classification"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(5, 5)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(5, 3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x