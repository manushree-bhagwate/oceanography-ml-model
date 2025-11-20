import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import copy
import os

# --- Configuration ---
data_dir = "marine_species_dataset"
num_epochs = 10
batch_size = 16
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Preprocessing ---
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

image_datasets = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=data_transforms["train"])
dataloader = DataLoader(image_datasets, batch_size=batch_size, shuffle=True)
class_names = image_datasets.classes  # ['malabar_group', 'napoleon_wrass', 'whale_shark']
num_classes = len(class_names)

print(f"Detected Classes: {class_names}")

# --- Model Setup ---
model = models.mobilenet_v2(pretrained=True)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 30)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets)
        epoch_acc = running_corrects.double() / len(image_datasets)

        print(f"Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")

    return model

# --- Train and Save Model ---
model = train_model(model, criterion, optimizer, num_epochs=num_epochs)
torch.save(model.state_dict(), "marine_species_model.pth")
print("Model saved as marine_species_model.pth")
