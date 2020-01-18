import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from models import CNN, deepCNN
from dataset import Datasets


# function to count number of parameters

def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np



    
num_workers = 0
# how many samples per batch to load
batch_size = 20
input_size  = 224*224*3   # images are 224x224 pixels
output_size = 2      # there are 2 classes

# define training, valid and test data directories
#data_dir = './Cat_Dog_data/'
train_dir = './train/pets'
test_dir = './test/pets'

train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                    transforms.RandomRotation(degrees=15),
                                    transforms.CenterCrop(size=224),
                                    transforms.ToTensor()])
#                                     transforms.Normalize([0.485, 0.456, 0.406],
#                                                          [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(size=(224,224)),
                                    transforms.ToTensor()])
#                                     transforms.Normalize([0.485, 0.456, 0.406],
#                                                          [0.229, 0.224, 0.225])])


# train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
# test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
train_data=Datasets(root_dir=train_dir, transform=train_transforms)
test_data=Datasets(root_dir=test_dir, transform=test_transforms)

train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_data, batch_size=batch_size, shuffle=False)



def funct(x):
    if x == 'cat':
        return 1
    else:
        return 0

accuracy_list = []

    
def train(epoch, model, perm=torch.arange(0, 50176).long()):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        target=pd.Series(target)
        target=torch.from_numpy(target.apply(funct).values)
        # permute pixels
        data = data.view(-1, 224*224)
        data = data[:, perm]
        data = data.view(-1, 3, 224, 224)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, perm=torch.arange(0, 50176).long()):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target=pd.Series(target)
        target=torch.from_numpy(target.apply(funct).values)
        # permute pixels
        data = data.view(-1, 224*224)
        data = data[:, perm]
        data = data.view(-1, 3, 224, 224)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))


n_features = 8 # number of feature maps

#model_cnn = CNN(input_size, n_features, output_size)
#optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
#print('Number of parameters: {}'.format(get_n_params(model_cnn)))

#for epoch in range(0, 1):
#    train(epoch, model_cnn)
#    test(model_cnn)
    

print("Multiple hidden layers CNN model:")
print()


model_cnn = deepCNN(input_size, n_features, output_size)
optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model_cnn)))

for epoch in range(0, 1):
    train(epoch, model_cnn)
    test(model_cnn)
