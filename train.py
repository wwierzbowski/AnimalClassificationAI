import kagglehub
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random

from constants import BATCH_SIZE, NUM_EPOCHS
from model import CNN

def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels) # This line is now correct
            
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    test_loss = running_loss / total_samples if total_samples > 0 else 0
    test_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    
    return test_loss, test_accuracy

if __name__ == '__main__':
    NUM_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset_root_path = kagglehub.dataset_download("alessiocorrado99/animals10")

    # Automatically find images and assign labels based on folder names
    actual_dataset_path = os.path.join(dataset_root_path, "raw-img")
    full_image_dataset = datasets.ImageFolder(root=actual_dataset_path, transform=transform)

    # Define the proportion for training and testing
    train_ratio = 0.8
    test_ratio = 1 - train_ratio
    
    # Calculate sizes
    total_size = len(full_image_dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size # Ensure all samples are used

    # Set a manual seed for reproducibility of the split
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(full_image_dataset, [train_size, test_size])

    # --- DataLoaders ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Using device: {device}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print(f"Dataset classes: {full_image_dataset.classes}")
    print(f"Number of classes: {len(full_image_dataset.classes)}")
    print(f"Total samples: {len(full_image_dataset)}")

    # Check a few samples
    for i, (image, label) in enumerate(train_dataloader):
        print(f"Batch {i}: Image shape: {image.shape}, Labels: {label}")
        if i >= 2:  # Just check first few batches
            break

    num_classes = len(full_image_dataset.classes) # Use full dataset for class count
    model = CNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nBegin training loop!")
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss_train = 0.0
        correct_predictions_train = 0
        total_samples_train = 0

        for batch_idx, (images, labels) in enumerate(train_dataloader): # Use train_dataloader
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # Zero the gradients
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            running_loss_train += loss.item() * images.size(0) # Accumulate batch loss
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples_train += labels.size(0)
            correct_predictions_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss_train / total_samples_train if total_samples_train > 0 else 0
        epoch_train_accuracy = (correct_predictions_train / total_samples_train) * 100 if total_samples_train > 0 else 0

        epoch_test_loss, epoch_test_accuracy = evaluate_model(model, test_dataloader, device, criterion)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_accuracy:.2f}%")

    print("\nTraining complete!")