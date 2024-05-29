import torch.nn as nn
import torch
import os
import json
import torchvision.models as models

import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

class SimpleCNN(nn.Module):
    def __init__(self, channel, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.base = FE(channel, input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def forward(self, x):
        return self.classifier((self.base(x)))
    
class BNCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(BNCNN, self).__init__()
        self.base = BNFE(input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def forward(self, x):
        return self.classifier((self.base(x)))
    
class FE(nn.Module):
    def __init__(self, channel, input_dim, hidden_dims):
        super(FE, self).__init__()
        self.conv1 = nn.Conv2d(channel, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
        
class BNFE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(BNFE, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
 
class Classifier(nn.Module):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dims, output_dim)
    
    def forward(self, x):
        # print(x.shape)
        # print(self.fc3.weight.shape)
        x = self.fc(x)
        return x

class CIFAR100_FE(nn.Module):
    def __init__(self):
        super(CIFAR100_FE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(256, 128)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # x = x.view(-1, 16 * 5 * 5)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return x

class CIFAR100_Classifier(nn.Module):
    def __init__(self):
        super(CIFAR100_Classifier, self).__init__()
        self.fc = nn.Linear(128, 100)
    
    def forward(self, x):
        x = self.fc(x)
        return x
    
class CIFAR100_CNN(nn.Module):
    def __init__(self):
        super(CIFAR100_CNN, self).__init__()
        self.base = CIFAR100_FE()
        self.classifier = CIFAR100_Classifier()

    def forward(self, x):
        return self.classifier((self.base(x)))
    
def simplecnn(n_classes):
    return SimpleCNN(channel=3, input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)

def simplecnn_mnist(n_classes):
    return SimpleCNN(channel=1, input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)

def bncnn(n_classes):
    return BNCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)

class TextCNN_FE(nn.Module):
    def __init__(self, vocab_size, embeddings, emb_size):
        super(TextCNN_FE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # self.embedding.weight = nn.Parameter(embeddings, requires_grad = True)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels = 1,
                out_channels = 100,
                kernel_size = (size, emb_size)
            )
            for size in [3, 4, 5]
        ])
        self.relu = nn.ReLU()
        
    def forward(self, text):
        embeddings = self.embedding(text).unsqueeze(1)  # (batch_size, 1, word_pad_len, emb_size)
        conved = [self.relu(conv(embeddings)).squeeze(3) for conv in self.convs]  # [(batch size, n_kernels, word_pad_len - kernel_sizes[n] + 1)]
        pooled = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]  # [(batch size, n_kernels)]
        flattened = torch.cat(pooled, dim = 1)  # (batch size, n_kernels * len(kernel_sizes))
        return flattened
    
class TextCNN(nn.Module):
    def __init__(self, n_classes, vocab_size, embeddings, emb_size):
        super(TextCNN, self).__init__()
        self.base = TextCNN_FE(vocab_size, embeddings, emb_size)
        self.classifier = Classifier(300, n_classes)
        
    def forward(self, x):
        return self.classifier((self.base(x)))

def textcnn(n_classes):
    with open(os.path.join("/GPFS/data/zhenyangni/moonfm/data/yahoo_answers_csv/sents", 'word_map.json'), 'r') as j:
        word_map = json.load(j)
        vocab_size = len(word_map)
        # embeddings, emb_size = load_embeddings(
        #             emb_file = os.path.join("/DB/data/zhenyangni/moonfm/glove", "glove.6B.300d.txt"),
        #             word_map = word_map,
        #             output_folder = "/DB/data/zhenyangni/moonfm/glove"
        #         )
    return TextCNN(n_classes, vocab_size, None, 256)

class ResNet18_FE(nn.Module):
    def __init__(self, conv3=False, gn=False):
        super(ResNet18_FE, self).__init__()
        basemodel = models.resnet18(pretrained=False)
        if conv3:
            basemodel.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            basemodel.maxpool = torch.nn.Identity()
        if gn:
            basemodel.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        self.fe = nn.Sequential(*list(basemodel.children())[:-1])

    def forward(self, x):
        out = self.fe(x)
        return out.view(out.size(0), -1)

class ResNet18(nn.Module):
    def __init__(self, n_classes, conv3=False, gn=False):
        super(ResNet18, self).__init__()
        self.base = ResNet18_FE(conv3, gn)
        self.classifier = nn.Linear(512, n_classes)
        
    def forward(self, x):
        x = self.base(x)
        return self.classifier(x)

def resnet18_3(n_classes):
    return ResNet18(n_classes, conv3=True)

def resnet18(n_classes):
    return ResNet18(n_classes, conv3=False)

def resnet18_gn(n_classes):
    return ResNet18(n_classes, conv3=False, gn=True)

def resnet18_3_gn(n_classes):
    return ResNet18(n_classes, conv3=True, gn=True)

def cifar100_cnn(n_classes):
    return CIFAR100_CNN()

def get_model(args):
    if args.model == 'simplecnn':
        model = simplecnn
    elif args.model == 'simplecnn-mnist':
        model = simplecnn_mnist
    elif args.model == 'bncnn':
        model = bncnn
    elif args.model == 'textcnn':
        model = textcnn
    elif args.model == 'resnet18-7':
        model = resnet18
    elif args.model == 'resnet18-3':
        model = resnet18_3
    elif args.model == 'resnet18':
        model = resnet18
    elif args.model == 'resnet18-gn':
        model = resnet18_gn
    elif args.model == 'resnet18-3-gn':
        model = resnet18_3_gn
    elif args.model == 'cifar100-cnn':
        model = cifar100_cnn
    return model
