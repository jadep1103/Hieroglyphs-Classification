# Importation des bibliothèques nécessaires
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
from torchvision import models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights, DenseNet121_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.heads.parameters():  # On garde la tête entraînable
        param.requires_grad = True

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class ViTClassifier(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=170,
                 dim=256, depth=4, heads=4, mlp_dim=512, dropout=0.1):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),              # (B, dim, num_patches)
            Transpose(1, 2),            # (B, num_patches, dim)
            nn.LayerNorm(dim)          # ✅ ajout LayerNorm juste après embeddings
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)  # ✅ initialisation propre

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                             # (B, num_patches, dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)       # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)               # (B, num_patches+1, dim)
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.transformer(x)                             # (B, num_patches+1, dim)
        return self.mlp_head(x[:, 0])                       # CLS token


    

# Fonction pour créer les modèles
def get_models(num_classes):
    models_dict = {}

    # resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    # models_dict['ResNet18'] = resnet

    # efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
    # models_dict['EfficientNet_b0'] = efficientnet

    # densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    # densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
    # models_dict['DenseNet121'] = densenet

    #models_dict['CustomCNN'] = CustomCNN(num_classes)

    #From scratch
    models_dict['ViT'] = ViTClassifier(image_size=224, num_classes=num_classes)

 # ViT préentraîné - fine-tuning complet
    vit_finetune = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    in_features = vit_finetune.heads[0].in_features  # ✅ CORRECTION ICI
    vit_finetune.heads = nn.Linear(in_features, num_classes)
    models_dict['ViT_pretrained_finetune'] = vit_finetune

    # ViT préentraîné - feature extractor (backbone figé)
    vit_frozen = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    vit_frozen.heads = nn.Linear(in_features, num_classes)
    freeze_backbone(vit_frozen)
    models_dict['ViT_pretrained_frozen'] = vit_frozen

    return models_dict
