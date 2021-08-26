#  -*- coding: utf-8 -*-

import torch 
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm import tqdm
from model import resnet18, resnext50, cnn 


train_tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.3),
    transforms.RandomPerspective(0.7),     
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
])

test_tfm = transforms.Compose([
    
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])




batch_size = 64

train_set = DatasetFolder("data_3classes/train", loader=lambda x:Image.open(x), extensions="jpeg", transform=train_tfm)
validate_set = DatasetFolder("data_3classes/validation", loader=lambda x:Image.open(x), extensions="jpeg", transform=test_tfm)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
validate_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)




device = "cuda:0"
model = resnet18().to(device)
model.device = device
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
n_epochs = 50
from torchsummary import summary


summary(model, (3, 224, 224), device="cuda")

for epoch in range(n_epochs):
    
    #set model to training mode
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        
        # A batch consists of image data and corresponding labels.
        imgs, targets = batch
        # print(imgs.shape())

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits,targets.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == targets.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

     # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")



    model.eval()


    val_loss = []
    val_acc = []

    for batch in tqdm(validate_loader):

        imags, targets = batch

        with torch.no_grad():
          logits = model(imags.to(device))

        loss = criterion(logits, targets.to(device))

        acc = (logits.argmax(dim=-1) == targets.to(device)).float().mean()

        val_loss.append(loss.item())
        val_acc.append(acc)

    val_loss = sum(val_loss)/len(val_loss)
    val_acc = sum(val_acc)/len(val_acc)    

    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")

torch.save(model, 'resnet18.pt')