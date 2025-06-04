import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Set up data augmentation and normalization
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class LFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.images = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.images.append((os.path.join(class_dir, image_name), class_name))
        random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, class_name = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.classes.index(class_name)
        return image, label

# Prepare dataset and dataloader
lfw_root_dir = "D:\PROJECTS\Face recognition Dataset\lfw-deepfunneled2"
lfw_dataset = LFWDataset(root_dir=lfw_root_dir, transform=transform_train)
dataloader = DataLoader(lfw_dataset, batch_size=32, shuffle=True)

# Define model - switch to ResNet50
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
num_classes = len(lfw_dataset.classes)
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(15):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

# Save model and class names
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': lfw_dataset.classes
}, "resnet50_face_recognition.pth")

print("Model training completed and saved.")
