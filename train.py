# Importation des bibliothèques nécessaires
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fonction d'entraînement
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []  # Stockage des métriques

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Précision d'entraînement
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation
        val_loss, val_acc = validate_model(model, val_loader)

        # Log dans l'historique
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")
        torch.cuda.empty_cache()

    return history


# Fonction de validation
def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    acc = 100 * correct / total
    return avg_loss, acc


# Fonction d'évaluation finale
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc
    
def train_process(model_name, model, train_loader, val_loader, test_loader, epochs, result_queue):
    print(f"Starting training for {model_name}")

    history = train_model(model, train_loader, val_loader, epochs=epochs)
    acc = evaluate_model(model, test_loader)
    torch.cuda.empty_cache()

    print(f"Finished training {model_name} with Test Accuracy: {acc:.2f}%")

    # === Nouveau : Sauvegarder l'historique dans un CSV ===
    df_history = pd.DataFrame(history)
    os.makedirs("logs", exist_ok=True)
    df_history.to_csv(f"logs/history_{model_name}.csv", index=False)

    # === Résultat pour la queue multiprocess ===
    result_queue.put({
        "model_name": model_name,
        "test_accuracy": acc,
        "history": history
    })
