import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # Input: (Batch_Size, 3, 64, 64) due to transforms.Resize((64, 64))
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # After Conv2d+ReLU: (Batch_Size, 32, 64, 64)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # After MaxPool2d: (Batch_Size, 32, 32, 32) (64 / 2 = 32)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # After Conv2d+ReLU: (Batch_Size, 64, 32, 32)
            nn.MaxPool2d(kernel_size=2, stride=2)
            # After MaxPool2d: (Batch_Size, 64, 16, 16) (32 / 2 = 16)
        )
        
        # Calculate the input features for the first Linear layer:
        # Number of channels * final height * final width
        # 64 channels * 16 height * 16 width = 16384
        
        self.classifier = nn.Sequential(
            nn.Flatten(), # Flattens (N, 64, 16, 16) to (N, 64 * 16 * 16)
            nn.Linear(64 * 16 * 16, 128), # <--- UPDATED THIS LINE
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x