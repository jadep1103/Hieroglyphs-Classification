import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# # Charger l'historique
# history_vit = pd.read_csv('logs/history_ViT.csv')

# # Générer un timestamp formaté
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# # Créer la figure
# plt.figure(figsize=(10, 4))

# # Courbe de loss
# plt.subplot(1, 2, 1)
# plt.plot(history_vit['epoch'], history_vit['train_loss'], label='Train Loss')
# plt.plot(history_vit['epoch'], history_vit['val_loss'], label='Val Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Curve Updated - ViT')
# plt.legend()
# plt.grid()

# # Courbe d'accuracy
# plt.subplot(1, 2, 2)
# plt.plot(history_vit['epoch'], history_vit['train_acc'], label='Train Accuracy')
# plt.plot(history_vit['epoch'], history_vit['val_acc'], label='Val Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy Curve - ViT')
# plt.legend()
# plt.grid()

# # Enregistrer la figure
# save_path = f'logs/upgraded/vit_curves_{timestamp}.png'
# plt.tight_layout()
# plt.savefig(save_path, dpi=300)
# print(f"Courbes sauvegardées sous : {save_path}")

# # Charger l'historique
# history_CustomCNN = pd.read_csv('logs/history_CustomCNN.csv')

# # Générer un timestamp formaté
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# # Créer la figure
# plt.figure(figsize=(10, 4))

# # Courbe de loss
# plt.subplot(1, 2, 1)
# plt.plot(history_CustomCNN['epoch'], history_CustomCNN['train_loss'], label='Train Loss')
# plt.plot(history_CustomCNN['epoch'], history_CustomCNN['val_loss'], label='Val Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Curve Updated - CustomCNN')
# plt.legend()
# plt.grid()

# # Courbe d'accuracy
# plt.subplot(1, 2, 2)
# plt.plot(history_CustomCNN['epoch'], history_CustomCNN['train_acc'], label='Train Accuracy')
# plt.plot(history_CustomCNN['epoch'], history_CustomCNN['val_acc'], label='Val Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy Curve - CustomCNN')
# plt.legend()
# plt.grid()

# # Enregistrer la figure
# save_path = f'logs/upgraded/CustomCNN_curves_{timestamp}.png'
# plt.tight_layout()
# plt.savefig(save_path, dpi=300)
# print(f"Courbes sauvegardées sous : {save_path}")

# history_ResNet18 = pd.read_csv('logs/history_ResNet18.csv')

# # Générer un timestamp formaté
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# # Créer la figure
# plt.figure(figsize=(10, 4))

# # Courbe de loss
# plt.subplot(1, 2, 1)
# plt.plot(history_ResNet18['epoch'], history_ResNet18['train_loss'], label='Train Loss')
# plt.plot(history_ResNet18['epoch'], history_ResNet18['val_loss'], label='Val Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Curve Updated - ResNet18')
# plt.legend()
# plt.grid()

# # Courbe d'accuracy
# plt.subplot(1, 2, 2)
# plt.plot(history_ResNet18['epoch'], history_ResNet18['train_acc'], label='Train Accuracy')
# plt.plot(history_ResNet18['epoch'], history_ResNet18['val_acc'], label='Val Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy Curve - ResNet18')
# plt.legend()
# plt.grid()

# # Enregistrer la figure
# save_path = f'logs/upgraded/ResNet18_curves_{timestamp}.png'
# plt.tight_layout()
# plt.savefig(save_path, dpi=300)
# print(f"Courbes sauvegardées sous : {save_path}")


# Charger l'historique
history_ViT_pretrained_frozen = pd.read_csv('logs/history_ViT_pretrained_frozen.csv')

# Générer un timestamp formaté
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Créer la figure
plt.figure(figsize=(10, 4))

# Courbe de loss
plt.subplot(1, 2, 1)
plt.plot(history_ViT_pretrained_frozen['epoch'], history_ViT_pretrained_frozen['train_loss'], label='Train Loss')
plt.plot(history_ViT_pretrained_frozen['epoch'], history_ViT_pretrained_frozen['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve Updated - ViT_pretrained_frozen')
plt.legend()
plt.grid()

# Courbe d'accuracy
plt.subplot(1, 2, 2)
plt.plot(history_ViT_pretrained_frozen['epoch'], history_ViT_pretrained_frozen['train_acc'], label='Train Accuracy')
plt.plot(history_ViT_pretrained_frozen['epoch'], history_ViT_pretrained_frozen['val_acc'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve - ViT_pretrained_frozen')
plt.legend()
plt.grid()

# Enregistrer la figure
save_path = f'logs/finalvitpretrained/ViT_pretrained_frozen_curves_{timestamp}.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Courbes sauvegardées sous : {save_path}")

# Charger l'historique
history_vit = pd.read_csv('logs/history_ViT.csv')

# Générer un timestamp formaté
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Créer la figure
plt.figure(figsize=(10, 4))

# Courbe de loss
plt.subplot(1, 2, 1)
plt.plot(history_vit['epoch'], history_vit['train_loss'], label='Train Loss')
plt.plot(history_vit['epoch'], history_vit['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve Updated - ViT')
plt.legend()
plt.grid()

# Courbe d'accuracy
plt.subplot(1, 2, 2)
plt.plot(history_vit['epoch'], history_vit['train_acc'], label='Train Accuracy')
plt.plot(history_vit['epoch'], history_vit['val_acc'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve - ViT')
plt.legend()
plt.grid()


# Charger l'historique
history_ViT_pretrained_finetune = pd.read_csv('logs/history_ViT_pretrained_finetune.csv')

# Générer un timestamp formaté
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Créer la figure
plt.figure(figsize=(10, 4))

# Courbe de loss
plt.subplot(1, 2, 1)
plt.plot(history_ViT_pretrained_finetune['epoch'], history_ViT_pretrained_finetune['train_loss'], label='Train Loss')
plt.plot(history_ViT_pretrained_finetune['epoch'], history_ViT_pretrained_finetune['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve Updated - ViT_pretrained_finetune')
plt.legend()
plt.grid()

# Courbe d'accuracy
plt.subplot(1, 2, 2)
plt.plot(history_ViT_pretrained_finetune['epoch'], history_ViT_pretrained_finetune['train_acc'], label='Train Accuracy')
plt.plot(history_ViT_pretrained_finetune['epoch'], history_ViT_pretrained_finetune['val_acc'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve - ViT_pretrained_finetune')
plt.legend()
plt.grid()

# Enregistrer la figure
save_path = f'logs/finalvitpretrained/ViT_pretrained_finetune_curves_{timestamp}.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Courbes sauvegardées sous : {save_path}")