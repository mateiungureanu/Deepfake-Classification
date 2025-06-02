import os
import time
import pandas as pd
import PIL.Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

BATCH_SIZE = 128
NUM_EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.1)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout4 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.dropout_fc = nn.Dropout(0.5)
        self.output = nn.Linear(512, 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        return self.output(x)


class ImageDataset(Dataset):
    def __init__(self, csv_file, images_folder, transform=None, has_labels=True):
        self.data = pd.read_csv(csv_file)
        self.images_folder = images_folder
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_folder, self.data.iloc[idx, 0] + ".png")
        image = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.has_labels:
            label = self.data.iloc[idx, 1]
            return image, label
        return image


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(100, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


test_val_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

train_data = ImageDataset("train.csv", "train", transform=train_transform, has_labels=True)
val_data = ImageDataset("validation.csv", "validation", transform=test_val_transform, has_labels=True)
test_data = ImageDataset("test.csv", "test", transform=test_val_transform, has_labels=False)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

model = NeuralNetwork().to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

start = time.time()

best_val_loss = float('inf')
epochs_no_improve = 0
early_stop_patience = 7
for epoch in range(NUM_EPOCHS):
    print(f"\n=== Epoch {epoch + 1} ===")
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).long()
        preds = model(images)
        loss = loss_function(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            preds = model(images)
            val_loss += loss_function(preds, labels).item()
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_val_loss:.6f}, Accuracy: {accuracy:.2f}%")
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_cnn_model.pth")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs")
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break


cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Validation Set")
plt.savefig("confusion_matrix_validation.png")

model.load_state_dict(torch.load("best_cnn_model.pth"))
model.eval()
predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(DEVICE)
        preds = model(images)
        predictions.extend(preds.argmax(1).cpu().numpy())

test_ids = pd.read_csv("test.csv")["image_id"]
submission_df = pd.DataFrame({
    "image_id": test_ids,
    "label": predictions
})
submission_df.to_csv("submission.csv", index=False)

end = time.time()
elapsed = end - start
print(f"\nTimp total: {int(elapsed // 60)}m {elapsed % 60:.2f}s")
