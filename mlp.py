# import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
import PIL
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
# from torchvision import datasets
from torchvision.transforms import ToTensor

start_time = time.time()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.first_layer = nn.Linear(100*100*3, 512)
        self.second_layer = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, 5)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        x = self.output_layer(x)
        return x

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, images_folder, transform=None, has_labels=True):
        self.data = pd.read_csv(csv_file)
        self.images_folder = images_folder
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(
            self.images_folder,
            self.data.iloc[idx, 0] + ".png"
        )

        image = PIL.Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            label = self.data.iloc[idx, 1]
            return image, label
        else:
            return image


train_data = ImageDataset(
    'train.csv',
    'train',
    transform=ToTensor(),
    has_labels=True
)

validation_data = ImageDataset(
    'validation.csv',
    'validation',
    transform=ToTensor(),
    has_labels=True
)

test_data = ImageDataset(
    'test.csv',
    'test',
    transform=ToTensor(),
    has_labels=False
)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

model = NeuralNetwork()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

NUM_EPOCHS=10
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
loss_function = nn.CrossEntropyLoss()

model.train(True)
for i in range(NUM_EPOCHS):
    print(f"=== Epoch {i+1} ===")
    for batch, (image_batch, labels_batch) in enumerate(train_dataloader):
        image_batch = image_batch.to(device)
        labels_batch = labels_batch.long().to(device) #(batch_size, )
        # print(image_batch.shape)
        # print(labels_batch.shape)

        pred = model(image_batch)
        print(pred.shape)
        loss = loss_function(pred, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss = loss.item()
            print(f"Batch index {batch }, loss: {loss:>7f}")
# torch.save(model.state_dict(), 'model.pth')

correct = 0.
validation_loss = 0.
size = len(validation_dataloader.dataset)
model.to(device)
# model.load_state_dict(torch.load('model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for image_batch, labels_batch in validation_dataloader:
        image_batch = image_batch.to(device)
        labels_batch = labels_batch.to(device)
        pred = model(image_batch)
        validation_loss += loss_function(pred, labels_batch).item()
        correct += (pred.argmax(1) == labels_batch).type(torch.float).sum().item()

        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

correct /= size
validation_loss /= size
print(f"Accuracy: {(100*correct):>0.1f}%, Loss: {validation_loss:>8f} \n")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Validation Set")
# plt.show()
plt.savefig("confusion_matrix_validation.png")

# model.load_state_dict(torch.load('model.pth'))
model.eval()
predictions = []

with torch.no_grad():
    for batch in test_dataloader:
        image_batch = batch.to(device)
        pred = model(image_batch)
        predicted_labels = pred.argmax(1)
        predictions.extend(predicted_labels.cpu().numpy())

test_ids = pd.read_csv("test.csv")["image_id"]

submission_df = pd.DataFrame({
    "image_id": test_ids,
    "label": predictions
})

submission_df.to_csv("submission.csv", index=False)

end_time = time.time()
elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = elapsed % 60

print(f"Timp de rulare: {minutes} minute și {seconds:.2f} secunde")