# Importation des bibliothèques nécessaires
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights, DenseNet121_Weights

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Fonction pour créer les modèles
def get_models(num_classes):
    models_dict = {}

    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    models_dict['ResNet18'] = resnet

    efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
    models_dict['EfficientNet_b0'] = efficientnet

    densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
    models_dict['DenseNet121'] = densenet

    models_dict['CustomCNN'] = CustomCNN(num_classes)

    return models_dict