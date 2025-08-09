if __name__ == "__main__":
    import os
    import sys
    import torch
    import torch.multiprocessing as mp
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    from train import train_process
    from dataset import load_datasets, compute_mean_std
    from models import get_models
    import pandas as pd

    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    if os.path.exists('./EgyptianHieroglyphDataset-1') :
        path = './EgyptianHieroglyphDataset-1/train'
    elif os.path.exists('./egyptianhieroglyphadataset') :
        path = './egyptianhieroglyphadataset/train'
    else:
        import download_dataset
        download_dataset()
    
      # Chemin vers le dataset CUB-200-2011

    # Calcul des statistiques
    path_train=os.path.join(path, "train")
    mean_data, std_data = compute_mean_std(path_train)
    print(f"\nRésultats pour CUB-200-2011:")
    print(f"Moyenne: {mean_data}")
    print(f"Écart-type: {std_data}")

    # Vérification GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations des images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_data, std=std_data)
    ])

    # Définition des chemins
    data_dir = "./EgyptianHieroglyphDataset-1"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")
    # Chargement des datasets
    train_loader, val_loader, test_loader, num_classes = load_datasets(train_dir, val_dir, test_dir,transform)

    # Récupération des modèles
    models_dict = get_models(num_classes)

    # Liste des processus
    result_queue = mp.Queue()
    processes = []
    epochs=10

    for name, model in models_dict.items():
        p = mp.Process(target=train_process, args=(name, model, train_loader, val_loader, test_loader, epochs, result_queue))
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

