import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
plt.title('Loss Curve - ViT')
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

# Enregistrer la figure
save_path = f'logs/vit_curves_{timestamp}.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Courbes sauvegardées sous : {save_path}")
