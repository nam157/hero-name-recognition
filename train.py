import os
import warnings
from typing import Any
import torch
import pandas as pd
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms as trns
from sklearn.metrics import precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")  # warning weights=DenseNet121_Weights.IMAGENET1K_V1


def label_encode(x, classes):
    for i, cls in enumerate(classes):
        if x == cls:
            return i


def create_dataset(path: str = "./test_data/test.txt"):
    data = []
    with open(path, "r") as files:
        for file in files:
            split_file = file.split()
            data.append((split_file[0], split_file[1]))
    df = pd.DataFrame(data, columns=["name_image", "label"])
    classes = sorted(list(set(df["label"].unique())))
    df["label"] = df["label"].apply(lambda x: label_encode(x, classes))
    return df, classes


class GameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Any = None) -> None:
        self.df = df
        self.transform = transform
        self.label = self.df.loc[:, "label"].values

    def __len__(self):
        return self.df.shape[0]

    def _get_image(self, path_img: str):
        img = Image.open(path_img).convert("RGB")
        return img

    def __getitem__(self, index):
        path_img = os.path.join(
            "./test_data/test_images/", self.df.loc[:, "name_image"][index]
        )
        img = self._get_image(path_img=path_img)
        if self.transform:
            img = self.transform(img)

        target = self.label[index]
        return img, target


train_transforms = trns.Compose(
    [
        trns.Resize((128, 128)),
        trns.RandomHorizontalFlip(),
        trns.RandomVerticalFlip(),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transforms = trns.Compose(
    [
        trns.Resize((128, 128)),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class DenseNet121(nn.Module):
    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(
            nn.Linear(kernelCount, classCount), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x



def train(model, optimizer, criterion, epoch, train_loader):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 5 == 0:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0


def val(model, optimizer, criterion, epoch, test_loader):
    model.eval()
    running_loss = 0.0
    precision_score_sum,recall_socre_sum = [],[]
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            if i % 5 == 0:
                print(
                    f"Validation:[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}"
                )
                running_loss = 0.0
            _, predicted = torch.max(outputs, 1)
            precision = precision_score(labels,predicted,average='micro')
            recall    = recall_score(labels,predicted,average='micro')

            precision_score_sum.append(precision)
            recall_socre_sum.append(recall)

    return sum(precision_score_sum)/len(precision_score_sum),sum(recall_socre_sum)/len(recall_socre_sum)


