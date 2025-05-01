def get_transforms(pretrained=False, mean_data=None, std_data=None):
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
        return weights.transforms()
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_data, std=std_data)
        ])

if __name__ == "__main__":
    import os
    import sys
    import torch
    import torch.multiprocessing as mp
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    from train import train_process
    from dataset import load_datasets, compute_mean_std
    from modelsfirsttry import get_models
    from torchvision.models import ViT_B_16_Weights
    import pandas as pd

    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if os.path.exists('./EgyptianHieroglyphDataset-1'):
        path = './EgyptianHieroglyphDataset-1'
    else:
        raise FileNotFoundError("Dataset not found. Please check the dataset folder name.")


    # Calcul des statistiques
    path_train=os.path.join(path, "train")
    mean_data, std_data = compute_mean_std(path_train)
    print(f"\nRésultats pour Hyeroglyphs:")
    print(f"Moyenne: {mean_data}")
    print(f"Écart-type: {std_data}")

    # Vérification GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Définition des chemins
    data_dir = "./EgyptianHieroglyphDataset-1"
    train_dir = os.path.join(data_dir, "new_train_aug")
    val_dir = os.path.join(data_dir, "new_valid")
    test_dir = os.path.join(data_dir, "new_test")
    # Chargement des datasets
    # Transforms
    transform_scratch = get_transforms(pretrained=False, mean_data=mean_data, std_data=std_data)
    transform_pretrained = get_transforms(pretrained=True)

    # Datasets (deux versions)
    train_loader_scratch, val_loader_scratch, test_loader_scratch, _ = load_datasets(train_dir, val_dir, test_dir, transform_scratch)
    train_loader_pretrained, val_loader_pretrained, test_loader_pretrained, num_classes = load_datasets(train_dir, val_dir, test_dir, transform_pretrained)

    # print(f"Nombre d'images d'entraînement : {len(train_loader.dataset)}")
    # print(f"Nombre d'images de validation : {len(val_loader.dataset)}")
    # print(f"Nombre d'images de test : {len(test_loader.dataset)}")
    # print(f"Nombre total de classes : {num_classes}")

    # Récupération des modèles
    models_dict = get_models(num_classes)

    # Liste des processus
    result_queue = mp.Queue()
    processes = []
    epochs=15

    for name, model in models_dict.items():
        if "pretrained" in name.lower():
            train_loader = train_loader_pretrained
            val_loader = val_loader_pretrained
            test_loader = test_loader_pretrained
        else:
            train_loader = train_loader_scratch
            val_loader = val_loader_scratch
            test_loader = test_loader_scratch

        p = mp.Process(
            target=train_process,
            args=(name, model, train_loader, val_loader, test_loader, epochs, result_queue)
        )
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    # Récupération des résultats
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    for res in results:
        model_name = res["model_name"]
        test_acc = res["test_accuracy"]
        history = res["history"]

        # Convertir l'historique en DataFrame pour logger proprement
        df_history = pd.DataFrame(history)
        print(df_history)

        print(f"{model_name} - Test Accuracy: {test_acc:.2f}%")

    print("Tous les entraînements sont terminés !")

