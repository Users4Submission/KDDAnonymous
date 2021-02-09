'''Extract Feature of CIFAR10 with PyTorch VGG19.'''
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import torch.nn as nn

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Feature Extraction')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
net = models.vgg19(pretrained=True)
new_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
new_classifier = new_classifier.to(device)
def extract():
    global best_acc
    net.eval()
    with torch.no_grad():
        embedding_list = []
        label_list = []
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            embedding = new_classifier(inputs)
            embedding_list.append(embedding.data.cpu().numpy())
            label_list.append(targets.data.cpu().numpy())
        np.save("./data/vgg19cifar10.npy", {'x': embedding_list, 'y': label_list})
        print("Finished!")
extract()


