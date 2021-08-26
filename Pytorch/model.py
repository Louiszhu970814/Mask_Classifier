import torch
import torch.nn as nn



class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)

        self.ending_layer = nn.Sequential(
            nn.Linear(1000,128),
            nn.Linear(128,3),
            nn.Softmax()
        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.ending_layer(x) 
        return x 



class resnext50(nn.Module):
    def __init__(self):
        super(resnext50, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'resnext50_32x4d', pretrained=True)

        self.ending_layer = nn.Sequential(
            nn.Linear(1000,3),
            nn.Softmax()
        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.ending_layer(x)
        return x


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),      #[32,128,128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          #[32,128,128]  

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          #[64,64,64]

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),          #[128,32,32]


            nn.AdaptiveAvgPool2d((1, 1)),
      
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256,3)

        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x
     
    