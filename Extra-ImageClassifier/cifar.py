import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score

from keras.datasets import cifar100
from keras.utils import to_categorical
import os, ssl
# Pytorch 
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)): ssl._create_default_https_context = ssl._create_unverified_context                             shuffle=False, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x2 = self.relu4(x)
        x = self.fc3(x2) 
        return (x,x2)

net = Net()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
            
# Extraction for Train Set
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    _, features = net(inputs)
    
    feature = features.data.numpy()
    label = labels.data.numpy()
    label = np.reshape(label,(labels.size(0),1))

    if i==0:
        featureMatrix = np.copy(feature)
        labelVector = np.copy(label)
    else:
        featureMatrix = np.vstack([featureMatrix,feature])
        labelVector = np.vstack([labelVector,label])

# Extraction for Test Set
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    _, features = net(inputs)
    feature = features.data.numpy()
    label = labels.data.numpy()
    label = np.reshape(label,(labels.size(0),1))

    if i==0:
        featureMatrixTest = np.copy(feature)
        labelVectorTest = np.copy(label)
    else:
        featureMatrixTest = np.vstack([featureMatrixTest,feature])
        labelVectorTest = np.vstack([labelVectorTest,label])


clfier = RandomForestClassifier(n_estimators = 100, max_depth = 10, min_samples_leaf = 10)

# Train 
clfier.fit(featureMatrix, np.ravel(labelVector))

# Test
predicted = clfier.predict(featureMatrixTest)

correct = (predicted == labelVectorTest).sum()
print('Accuracy on the  test images: %d %%' % (
    100 * correct / labelVectorTest.shape[0]))