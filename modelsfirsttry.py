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

class ViTClassifier(nn.Module):
    def __init__(self, image_size=128, patch_size=16, num_classes=170, dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.1):
        super(ViTClassifier, self).__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisibles par patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size  # 3 canaux (RGB)

        self.patch_size = patch_size

        # Embedding des patches
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            Transpose(1, 2)  # (batch, num_patches, dim)
        )

        # Tokens + positionnels
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Encodeur Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.to_patch_embedding(x)  # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

    

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

    # models_dict['CustomCNN'] = CustomCNN(num_classes)

    #From scratch
    # models_dict['ViT'] = ViTClassifier(image_size=224, num_classes=num_classes)

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
