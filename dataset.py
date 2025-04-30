from tqdm import tqdm
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
import os

def compute_mean_std(dataset_path):
    """
    Calcule la moyenne et l'écart-type des canaux R, G, B pour le dataset CUB-200-2011
    
    Args:
        dataset_path (str): Chemin vers le dossier contenant les images
        
    Returns:
        mean (tuple): Moyenne pour chaque canal (R, G, B)
        std (tuple): Écart-type pour chaque canal (R, G, B)
    """
    # Transformation sans normalisation (juste conversion en tensor et resize)
    transform = T.Compose([
        T.Resize((224, 224)),  # Redimensionne toutes les images directement à 224x224
        T.ToTensor(),
    ])
    
    # Chargement du dataset
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)
    
    # Variables pour le calcul
    mean = 0.
    std = 0.
    nb_samples = 0.
    
    print("Calcul des statistiques...")
    for data, _ in tqdm(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    
    return mean.tolist(), std.tolist()

# Chargement des datasets
def load_datasets(train_dir, val_dir, test_dir,transform):
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    # print(train_dataset.classes)

    return train_loader, val_loader, test_loader, len(train_dataset.classes)



# ============ SECTION DEBUG =============
if __name__ == "__main__":
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    data_dir = "./EgyptianHieroglyphDataset-1"
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "valid")
    test_dir  = os.path.join(data_dir, "test")

    train_loader, val_loader, test_loader, num_classes = load_datasets(train_dir, val_dir, test_dir, transform)

    print("\n✅ Vérification des classes alignées :")
    train_classes = train_loader.dataset.classes
    val_classes   = val_loader.dataset.classes
    test_classes  = test_loader.dataset.classes

    print(f"Nombre de classes : {num_classes}")
    print(f"Train classes (n={len(train_classes)}): {train_classes[:5]} ...")
    print(f"Val   classes (n={len(val_classes)}): {val_classes[:5]} ...")
    print(f"Test  classes (n={len(test_classes)}): {test_classes[:5]} ...")

    assert train_classes == val_classes == test_classes, "💥 Mismatch de mapping entre les splits !"
    print("✅ Mapping de classes cohérent !")