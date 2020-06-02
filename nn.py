import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, img_size, num_classes):
        super(NeuralNet, self).__init__()
        # input shape [batch size, 1, img_size, img_size]

        # multi-label
        # INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
        self.N = 1
        self.M = 4
        self.K = 2

        self.input_size = img_size
        self.num_classes = num_classes
        self.loss_fn = loss_fn

        self.conv_pool1 = nn.Sequential(
                            nn.Conv2d(1, 64, (3, 3), padding=1),
                            nn.MaxPool2d((2,2)),
                            nn.ReLU()
                        )
        self.conv_pool2 = nn.Sequential(
                            nn.Conv2d(64, 128, (3, 3), padding=1),
                            nn.MaxPool2d((2,2)),
                            nn.ReLU()
                        )
        self.conv_pool3 = nn.Sequential(
                            nn.Conv2d(128, 256, (3, 3), padding=1),
                            nn.MaxPool2d((2,2)),
                            nn.ReLU()
                        )
        self.conv_pool4 = nn.Sequential(
                            nn.Conv2d(256, 512, (3, 3), padding=1),
                            nn.MaxPool2d((2,2)),
                            nn.ReLU(),
                            nn.Dropout(p=0.2)
                        )

        self.fc = nn.Sequential(
                    nn.Linear(512 * 4 * 4, 1024),
                    nn.Dropout(p=0.7),
                    nn.Linear(1024,256),
                    nn.Dropout(p=0.5),
                    nn.Linear(256, self.num_classes)
                )

        self.softmax = torch.nn.Softmax(dim=1)
        self.optimizer = optim.Adam(self.get_parameters(), lr=lrate, weight_decay = 1e-5)

    def get_parameters(self):
        return self.parameters()

    def forward(self, x):
        x = self.conv_pool1(x)
        x = self.conv_pool2(x)
        x = self.conv_pool3(x)
        x = self.conv_pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def step(self, x, y):
        self.optimizer.zero_grad()
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item(), yhat
