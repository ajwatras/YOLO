import torch
import torchvision
import torch.nn as nn

architecture_config = [
    #Tuple: (kernel_size, number of filters, strides, padding)
    (7, 64, 2, 3),
    #"M" = Max Pool Layer
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    #List: [(tuple), (tuple), how many times to repeat]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
    #Doesnt include fc layers
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, X):
        return self.leakyrelu(self.batchnorm(self.conv(X)))

class YoloV1Model(nn.Module):
    def __init__(self, in_channels = 3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.conv_layers = self._create_conv_layers() 
        self.fc_layers = self._create_fc_layers(7, 2, 80)
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1 )
        x = self.fc_layers(x)
        return x  
    def _create_conv_layers(self):
        layers = []
        in_channels = self.in_channels

        for x in self.architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, x[1], kernel_size = x[0], stride = x[2], padding = x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                repeats = x[-1]

                for _ in range(repeats):
                    for block in x[0:-1]:
                        if type(block) == tuple:
                            layers += [CNNBlock(in_channels, block[1], kernel_size = block[0], stride = block[2], padding = block[3])]
                            in_channels = block[1]
                        elif type(block) == str:
                            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)
    
    def _create_fc_layers(self, grid_size, num_boxes, num_classes):
        fc_layers = nn.Sequential(nn.Flatten(), nn.Linear(1024 * grid_size * grid_size, 496), 
                                  nn.Dropout(0.0), nn.LeakyReLU(0.1), nn.Linear(496, grid_size * grid_size * (num_classes + num_boxes * 5)))
        return fc_layers
    def __str__(self):
        return super().__str__()
    