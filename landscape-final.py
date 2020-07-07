#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install jovian --upgrade --quiet')
import jovian
project_name='landscape-final' # will be used by jovian.commit


# In[2]:


import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Look into the data directory
data_dir = '../input/intel-image-classification/'
print(os.listdir(data_dir))


# In[4]:


train_tfms = tt.Compose([
    tt.Resize((150, 150)),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
#     tt.RandomErasing(inplace=True),
#     tt.Normalize((0.4332263 , 0.4585635 , 0.45523438), (0.237158  , 0.23522326, 0.24391426))
])

valid_tfms = tt.Compose([
    tt.Resize((150, 150)),
    tt.ToTensor(),
#     tt.Normalize((0.4332263 , 0.4585635 , 0.45523438), (0.237158  , 0.23522326, 0.24391426))
])


# In[5]:


# PyTorch datasets
train_ds = ImageFolder(data_dir+'/seg_train/seg_train', train_tfms)
valid_ds = ImageFolder(data_dir+'/seg_test/seg_test', valid_tfms)
len(train_ds),len(valid_ds)


# In[6]:


batch_size = 64


# In[7]:


# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)


# In[8]:


from tqdm import tqdm
no_of_images = 0
mean = 0.
std = 0.
for batch, _ in tqdm(train_dl):
    # Rearrange batch to be the shape of [batch_size, no_of_channels, width_of_image * height_of_image]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    
    # Update total number of images
    no_of_images += batch.size(0)
    
    # Compute mean and std here
    mean += batch.mean(2).sum(0) 
    std += batch.std(2).sum(0)


mean /= no_of_images
std /= no_of_images

print("Total number of images :", no_of_images," where Mean of images across channels: ", mean , "and std of images across channels: " , std)


# In[9]:


# Get the classes
import pathlib
root = pathlib.Path('../input/intel-image-classification/seg_train/seg_train')
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)


# In[10]:


buildings_files = os.listdir(data_dir + "/seg_train/seg_train/buildings")
print('No. of training examples for buildings:', len(buildings_files))

forest_files = os.listdir(data_dir + "/seg_train/seg_train/forest")
print('No. of training examples for forests:', len(forest_files))

glacier_files = os.listdir(data_dir + "/seg_train/seg_train/glacier")
print('No. of training examples for glaciers:', len(glacier_files))

mountain_files = os.listdir(data_dir + "/seg_train/seg_train/mountain")
print('No. of training examples for mountains:', len(mountain_files))

sea_files = os.listdir(data_dir + "/seg_train/seg_train/sea")
print('No. of training examples for seas:', len(sea_files))

street_files = os.listdir(data_dir + "/seg_train/seg_train/street")
print('No. of training examples for streets:', len(street_files))


# In[11]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
landscapes = [len(buildings_files),len(forest_files),len(glacier_files),len(mountain_files),len(sea_files),len(street_files)]
ax.bar(classes,landscapes)
plt.show()


# As we can see, the data is pretty balanced

# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def imshow(img):
  '''
  Function to un-normalize and display an image
  '''
#   img = img/2 + 0.5 # un-normalize
  plt.imshow(np.transpose(img, (1, 2, 0))) # convert from tensor image
  
# Get a batch of training images
dataiter = iter(train_dl)
images, labels = next(dataiter)
images = images.cpu().data.numpy() # convert images to numpy for display

# Plot the images from the batch, along with corresponding labels
fig = plt.figure(figsize = (25, 4))

# Display 20 images
for idx in np.arange(20):
  ax = fig.add_subplot(2, 20/2, idx+1, xticks = [], yticks = [])
  imshow(images[idx])
  ax.set_title(classes[labels[idx]])


# The images look at bit odd as the pixel values have been altered in the process of normalization.

# In[13]:


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break


# In[14]:


show_batch(train_dl)


# In[15]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[16]:


device = get_default_device()
device


# In[17]:


train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)


# In[18]:


class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the input


# In[19]:


simple_resnet = to_device(SimpleResidualBlock(), device)

for images, labels in train_dl:
    out = simple_resnet(images)
    print(out.shape)
    break
    
del simple_resnet, images, labels
torch.cuda.empty_cache()


# In[20]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


# In[21]:


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(8192, 6))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

class convNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(82944, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6))
        
    def forward(self, xb):
        return self.network(xb)


# In[22]:


model1 = to_device(ResNet9(3, 6), device)
# model1 = to_device(convNet(), device)
model1


# In[23]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[24]:


history = [evaluate(model1, valid_dl)]
history


# In[25]:


epochs = 10
max_lr = 0.0005
grad_clip = 0.2
weight_decay = 1e-5
opt_func = torch.optim.Adam


# In[26]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, max_lr, model1, train_dl, valid_dl, \n                             grad_clip=grad_clip, \n                             weight_decay=weight_decay, \n                             opt_func=opt_func)')


# In[ ]:


train_time='13:56'


# In[27]:


def plot_scores(history):
    scores = [x['val_acc'] for x in history]
    plt.plot(scores, '-x')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('Accuracy vs. No. of epochs'); 


# In[28]:


plot_scores(history)


# In[29]:


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');


# In[30]:


plot_losses(history)


# In[31]:


def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');


# In[32]:


plot_lrs(history)


# In[33]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# Normal ConvNet

# In[42]:


model1 = to_device(convNet(), device)
model1


# In[43]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[44]:


history = [evaluate(model1, valid_dl)]
history


# In[45]:


epochs = 20
max_lr = 0.0005
grad_clip = 0.2
weight_decay = 1e-5
opt_func = torch.optim.Adam


# In[46]:


get_ipython().run_cell_magic('time', '', 'history += fit_one_cycle(epochs, max_lr, model1, train_dl, valid_dl, \n                             grad_clip=grad_clip, \n                             weight_decay=weight_decay, \n                             opt_func=opt_func)')


# In[47]:


def plot_scores(history):
    scores = [x['val_acc'] for x in history]
    plt.plot(scores, '-x')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('Accuracy vs. No. of epochs'); 


# In[48]:


plot_scores(history)


# In[49]:


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');


# In[50]:


plot_losses(history)


# In[51]:


def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');


# In[52]:


plot_lrs(history)


# **Transfer Learning**

# VGG19 

# In[53]:


# import torchvision
import torchvision.models as models
model2 = models.vgg19(pretrained=True).to(device)
for param in model2.features.parameters():
    param.requires_grad = False


# In[55]:


model2.classifier[6] = nn.Linear(model2.classifier[6].in_features, len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.classifier.parameters(), lr=0.0005)


# In[56]:


model2


# In[57]:


trainlosses = []
testlosses = []
for e in range(20):
    trainloss = 0
    traintotal = 0
    trainsuccessful = 0
    for traininput, trainlabel in tqdm(train_dl):
        traininputs, trainlabels = traininput.to(device), trainlabel.to(device)
        optimizer.zero_grad()
        trainpredictions = model2(traininputs)
        _, trainpredict = torch.max(trainpredictions.data, 1)
        loss = criterion(trainpredictions, trainlabels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
        traintotal += trainlabels.size(0)
        trainsuccessful += (trainpredict == trainlabels).sum().item()
    else:
        testloss = 0
        testtotal = 0
        testsuccessful = 0
        with torch.no_grad():
            for testinput, testlabel in tqdm(valid_dl):
                testinputs, testlabels = testinput.to(device), testlabel.to(device)
                testpredictions = model2(testinputs)
                _, testpredict = torch.max(testpredictions.data, 1)
                tloss = criterion(testpredictions, testlabels)
                testloss += tloss.item()
                testtotal += testlabels.size(0)
                testsuccessful += (testpredict == testlabels).sum().item()
        trainlosses.append(trainloss/len(train_dl))
        testlosses.append(testloss/len(valid_dl))
        print('Epoch :',e)
        print('Train Accuracy %{:.2f}'.format(100*trainsuccessful/traintotal))
        print('Test Accuracy %{:.2f}'.format(100*testsuccessful/testtotal))


# In[ ]:





# Resnet18

# In[64]:


# import torchvision
import torchvision.models as models
model3 = models.resnet18(pretrained=True).to(device)
# for param in model3.features.parameters():
#     param.requires_grad = False


# In[65]:


model3.fc = nn.Linear(512, len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model3.fc.parameters(), lr=0.005)
model3


# In[66]:


trainlosses = []
testlosses = []
for e in range(10):
    trainloss = 0
    traintotal = 0
    trainsuccessful = 0
    for traininput, trainlabel in tqdm(train_dl):
        traininputs, trainlabels = traininput.to(device), trainlabel.to(device)
        optimizer.zero_grad()
        trainpredictions = model3(traininputs)
        _, trainpredict = torch.max(trainpredictions.data, 1)
        loss = criterion(trainpredictions, trainlabels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
        traintotal += trainlabels.size(0)
        trainsuccessful += (trainpredict == trainlabels).sum().item()
    else:
        testloss = 0
        testtotal = 0
        testsuccessful = 0
        with torch.no_grad():
            for testinput, testlabel in tqdm(valid_dl):
                testinputs, testlabels = testinput.to(device), testlabel.to(device)
                testpredictions = model3(testinputs)
                _, testpredict = torch.max(testpredictions.data, 1)
                tloss = criterion(testpredictions, testlabels)
                testloss += tloss.item()
                testtotal += testlabels.size(0)
                testsuccessful += (testpredict == testlabels).sum().item()
        trainlosses.append(trainloss/len(train_dl))
        testlosses.append(testloss/len(valid_dl))
        print('Epoch: ', e)
        print('Train Accuracy %{:.2f}'.format(100*trainsuccessful/traintotal))
        print('Test Accuracy %{:.2f}'.format(100*testsuccessful/testtotal))


# Googlenet

# In[67]:


# import torchvision
import torchvision.models as models
model4 = models.googlenet(pretrained=True).to(device)
for param in model4.parameters():
    param.requires_grad = False
for param in model4.fc.parameters():
    param.requires_grad = True
model4


# In[71]:


model4.fc.out_features = 6
model4.to(device)
# model2.classifier[6] = nn.Linear(model2.classifier[6].in_features, len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model4.fc.parameters(), lr=0.005)


# In[72]:


trainlosses = []
testlosses = []
for e in range(20):
    trainloss = 0
    traintotal = 0
    trainsuccessful = 0
    for traininput, trainlabel in tqdm(train_dl):
        traininputs, trainlabels = traininput.to(device), trainlabel.to(device)
        optimizer.zero_grad()
        trainpredictions = model4(traininputs)
        _, trainpredict = torch.max(trainpredictions.data, 1)
        loss = criterion(trainpredictions, trainlabels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
        traintotal += trainlabels.size(0)
        trainsuccessful += (trainpredict == trainlabels).sum().item()
    else:
        testloss = 0
        testtotal = 0
        testsuccessful = 0
        with torch.no_grad():
            for testinput, testlabel in tqdm(valid_dl):
                testinputs, testlabels = testinput.to(device), testlabel.to(device)
                testpredictions = model4(testinputs)
                _, testpredict = torch.max(testpredictions.data, 1)
                tloss = criterion(testpredictions, testlabels)
                testloss += tloss.item()
                testtotal += testlabels.size(0)
                testsuccessful += (testpredict == testlabels).sum().item()
        trainlosses.append(trainloss/len(train_dl))
        testlosses.append(testloss/len(valid_dl))
        print('Epoch :',e)
        print('Train Accuracy %{:.2f}'.format(100*trainsuccessful/traintotal))
        print('Test Accuracy %{:.2f}'.format(100*testsuccessful/testtotal))


# In[74]:


# import torchvision
import torchvision.models as models
model5 = models.mobilenet_v2(pretrained=True).to(device)
for param in model2.features.parameters():
    param.requires_grad = False


# In[75]:


model5.classifier[1].out_features = 6
model5.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model5.classifier.parameters(), lr=0.0005)
model5


# In[76]:


trainlosses = []
testlosses = []
for e in range(10):
    trainloss = 0
    traintotal = 0
    trainsuccessful = 0
    for traininput, trainlabel in tqdm(train_dl):
        traininputs, trainlabels = traininput.to(device), trainlabel.to(device)
        optimizer.zero_grad()
        trainpredictions = model5(traininputs)
        _, trainpredict = torch.max(trainpredictions.data, 1)
        loss = criterion(trainpredictions, trainlabels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
        traintotal += trainlabels.size(0)
        trainsuccessful += (trainpredict == trainlabels).sum().item()
    else:
        testloss = 0
        testtotal = 0
        testsuccessful = 0
        with torch.no_grad():
            for testinput, testlabel in tqdm(valid_dl):
                testinputs, testlabels = testinput.to(device), testlabel.to(device)
                testpredictions = model5(testinputs)
                _, testpredict = torch.max(testpredictions.data, 1)
                tloss = criterion(testpredictions, testlabels)
                testloss += tloss.item()
                testtotal += testlabels.size(0)
                testsuccessful += (testpredict == testlabels).sum().item()
        trainlosses.append(trainloss/len(train_dl))
        testlosses.append(testloss/len(valid_dl))
        print('Epoch: ', e)
        print('Train Accuracy %{:.2f}'.format(100*trainsuccessful/traintotal))
        print('Test Accuracy %{:.2f}'.format(100*testsuccessful/testtotal))

Alexnet
# In[77]:


# import torchvision
import torchvision.models as models
model6 = models.alexnet(pretrained=True).to(device)
for param in model6.features.parameters():
    param.requires_grad = False


# In[78]:


model6.classifier[6].out_features = 6
model6.to(device)
# # model2.classifier[6] = nn.Linear(model2.classifier[6].in_features, len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model6.classifier.parameters(), lr=0.00001)
model6


# In[79]:


trainlosses = []
testlosses = []
for e in range(15):
    trainloss = 0
    traintotal = 0
    trainsuccessful = 0
    for traininput, trainlabel in tqdm(train_dl):
        traininputs, trainlabels = traininput.to(device), trainlabel.to(device)
        optimizer.zero_grad()
        trainpredictions = model6(traininputs)
        _, trainpredict = torch.max(trainpredictions.data, 1)
        loss = criterion(trainpredictions, trainlabels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
        traintotal += trainlabels.size(0)
        trainsuccessful += (trainpredict == trainlabels).sum().item()
    else:
        testloss = 0
        testtotal = 0
        testsuccessful = 0
        with torch.no_grad():
            for testinput, testlabel in tqdm(valid_dl):
                testinputs, testlabels = testinput.to(device), testlabel.to(device)
                testpredictions = model6(testinputs)
                _, testpredict = torch.max(testpredictions.data, 1)
                tloss = criterion(testpredictions, testlabels)
                testloss += tloss.item()
                testtotal += testlabels.size(0)
                testsuccessful += (testpredict == testlabels).sum().item()
        trainlosses.append(trainloss/len(train_dl))
        testlosses.append(testloss/len(valid_dl))
        print('Epoch: ',e)
        print('Train Accuracy %{:.2f}'.format(100*trainsuccessful/traintotal))
        print('Test Accuracy %{:.2f}'.format(100*testsuccessful/testtotal))


# In[ ]:


# import torchvision
import torchvision.models as models
model8 = models.vgg11_bn(pretrained=True).to(device)
for param in model8.features.parameters():
    param.requires_grad = False


# In[ ]:


model8.classifier[6] = nn.Linear(model2.classifier[6].in_features, len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model8.classifier.parameters(), lr=0.00001)
model8


# In[ ]:


trainlosses = []
testlosses = []
for e in range(20):
    trainloss = 0
    traintotal = 0
    trainsuccessful = 0
    for traininput, trainlabel in tqdm(train_dl):
        traininputs, trainlabels = traininput.to(device), trainlabel.to(device)
        optimizer.zero_grad()
        trainpredictions = model8(traininputs)
        _, trainpredict = torch.max(trainpredictions.data, 1)
        loss = criterion(trainpredictions, trainlabels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
        traintotal += trainlabels.size(0)
        trainsuccessful += (trainpredict == trainlabels).sum().item()
    else:
        testloss = 0
        testtotal = 0
        testsuccessful = 0
        with torch.no_grad():
            for testinput, testlabel in tqdm(valid_dl):
                testinputs, testlabels = testinput.to(device), testlabel.to(device)
                testpredictions = model8(testinputs)
                _, testpredict = torch.max(testpredictions.data, 1)
                tloss = criterion(testpredictions, testlabels)
                testloss += tloss.item()
                testtotal += testlabels.size(0)
                testsuccessful += (testpredict == testlabels).sum().item()
        trainlosses.append(trainloss/len(train_dl))
        testlosses.append(testloss/len(valid_dl))
        print('Epoch: ',e)
        print('Train Accuracy %{:.2f}'.format(100*trainsuccessful/traintotal))
        print('Test Accuracy %{:.2f}'.format(100*testsuccessful/testtotal))


# In[20]:


# import torchvision
import torchvision.models as models
model9 = models.resnet34(pretrained=True).to(device)
for param in model9.parameters():
    param.requires_grad = False
model9


# In[23]:


model9.fc = nn.Linear(512,6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model9.fc.parameters(), lr=0.00001)
model9


# In[25]:


from tqdm import tqdm
trainlosses = []
testlosses = []
for e in range(10):
    trainloss = 0
    traintotal = 0
    trainsuccessful = 0
    for traininput, trainlabel in tqdm(train_dl):
        traininputs, trainlabels = traininput.to(device), trainlabel.to(device)
        optimizer.zero_grad()
        trainpredictions = model9(traininputs)
        _, trainpredict = torch.max(trainpredictions.data, 1)
        loss = criterion(trainpredictions, trainlabels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
        traintotal += trainlabels.size(0)
        trainsuccessful += (trainpredict == trainlabels).sum().item()
    else:
        testloss = 0
        testtotal = 0
        testsuccessful = 0
        with torch.no_grad():
            for testinput, testlabel in tqdm(valid_dl):
                testinputs, testlabels = testinput.to(device), testlabel.to(device)
                testpredictions = model9(testinputs)
                _, testpredict = torch.max(testpredictions.data, 1)
                tloss = criterion(testpredictions, testlabels)
                testloss += tloss.item()
                testtotal += testlabels.size(0)
                testsuccessful += (testpredict == testlabels).sum().item()
        trainlosses.append(trainloss/len(train_dl))
        testlosses.append(testloss/len(valid_dl))
        print('Epoch: ',e)
        print('Train Accuracy %{:.2f}'.format(100*trainsuccessful/traintotal))
        print('Test Accuracy %{:.2f}'.format(100*testsuccessful/testtotal))


# In[ ]:





# In[ ]:





# In[ ]:


jovian.log_metrics(train_loss=history[-1]['train_loss'], 
                   val_loss=history[-1]['val_loss'], 
                   val_acc=history[-1]['val_acc'])


# In[80]:


jovian.commit(project=project_name, environment=None)


# In[ ]:




