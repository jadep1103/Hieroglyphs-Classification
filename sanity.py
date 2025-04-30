import torch
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torch.utils.data import DataLoader
from dataset import load_datasets

# === CONFIGURATION ===
data_dir = "./EgyptianHieroglyphDataset-1"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/valid"
test_dir = f"{data_dir}/test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ViT_B_16_Weights.DEFAULT
transform = weights.transforms()

# === LOAD DATASETS ===
train_loader, val_loader, test_loader, num_classes = load_datasets(train_dir, val_dir, test_dir, transform)

# === Fonction de diagnostic ===
def check_loader(loader, name="loader"):
    print(f"\n🔍 Vérification des labels du {name}...")
    all_targets = []
    for _, y in loader:
        all_targets.extend(y.tolist())

    counter = Counter(all_targets)
    print(f"\n📊 Répartition des classes ({name}) :")
    for cls_id, count in sorted(counter.items()):
        print(f"Classe {cls_id:3d} → {count} images")

    # Mini batch check
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.heads = torch.nn.Linear(model.heads[0].in_features, num_classes)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)

            print(f"\n🔎 Mini batch check ({name}):")
            print("GT   :", y.tolist())
            print("Pred :", preds.tolist())

            # Visualisation (8 images max)
            plt.figure(figsize=(12, 6))
            for i in range(min(8, len(x))):
                img = x[i].cpu().permute(1, 2, 0)
                img = (img * 0.5 + 0.5).clip(0, 1)  # denormalize
                plt.subplot(2, 4, i + 1)
                plt.imshow(img)
                plt.title(f"GT: {y[i].item()} / Pred: {preds[i].item()}")
                plt.axis("off")
            plt.tight_layout()
            plt.show()
            break

# === LANCEMENT DES TESTS ===
check_loader(train_loader, "train_loader")
check_loader(val_loader, "val_loader")
