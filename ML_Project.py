import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for filename in os.listdir(root_dir):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".webp") :
                label_str = os.path.splitext(filename)[0].split('-')[-1]
                label = int(label_str)
                self.labels.append(label)
                self.images.append(os.path.join(root_dir, filename))

    def __len__(self):
        return len(self.images)
    
    def filename(self):
        for file in self.images:
            print(file)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CustomDataset(root_dir='images/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = CustomDataset(root_dir='images/test', transform=transform)
print(test_dataset.filename())
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = models.resnet18(pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 4)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

model.eval()
all_preds = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        # print(preds, labels, inputs)
        all_preds.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    print(len(all_preds), len(true_labels))
    print(all_preds, true_labels)

accuracy = accuracy_score(true_labels, all_preds)
conf_matrix = confusion_matrix(true_labels, all_preds)

print(f'Test Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
