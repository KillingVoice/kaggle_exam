"""/**
 * @author ljm00
 * @email ljm000701@naver.com
 * @create date 2024-05-15 03:13:48
 * @modify date 2024-05-15 03:13:48
 * @desc [description]
 */"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import glob
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#cuda
device = torch.device("cuda")

def datapath_Label():
    dl = pd.read_csv("./DATA/data.csv")
    labels = dl['is_True']
    path = dl['data_path']
    return path, labels

#data preprocessing
def dataLoader():
    paths, labels = datapath_Label()
    Y = torch.tensor(labels)
    X = []
    for path in paths:
        a = pd.read_csv(path,index_col=None).transpose()
        a = np.array(a[1:])
        data_tensor = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
        if data_tensor.shape[-1] > 17826:
            data_tensor = torch.cat((data_tensor[:,:,:17824],data_tensor[:,:,-2:]),dim=2)
        X.append(data_tensor)

    X = torch.stack(X)
    return X,Y

x,y = dataLoader()

data = TensorDataset(x,y)
train_data, test_data = random_split(data,[int(len(data)*0.7)+1,int(len(data)*0.3)])

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
test_loader = DataLoader(test_data,batch_size=1,shuffle=True)


# CNN Architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=3,out_channels=9,kernel_size = (3,3))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2,2))
        self.conv3 = nn.Conv2d(in_channels=9,out_channels=16,kernel_size=(3,3))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*1*2226,32*9*33) # CNN Data shape->16*1*2226
        self.fc2 = nn.Linear(32 * 9 * 33, 128)
        self.fc3 = nn.Linear(128, 2)  # Fake or Real, dinary classfier

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Train
total_losses=[]
num_epochs = 14
for epoch in range(num_epochs):    
    running_loss=0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    total_loss = running_loss / len(train_loader)
    total_losses.append(total_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

torch.save(model.state_dict(),"./model1.pth")

plt.figure(figsize=(10,12))
plt.plot(range(1, num_epochs+1), total_losses)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training Loss')
plt.show()
