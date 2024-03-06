"""
Here you'll find a simple CNN network that you'll can use during the assessment;
"""
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, hooked = False,num_classes=10, num_depth=1):
        super(SimpleModel, self).__init__()
        self.num_classes=num_classes
        self.hooked = hooked
        self.features_conv =None
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=num_depth, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

           
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=4608, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=64, out_features=self.num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
     
        "Here is the hook!!"
        self.features_conv=x
        if self.hooked == True:
            x.register_hook(self.activations_hook)
  
        x = self.classifier(x)
        return x
    

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
       
      
        return self.features_conv


