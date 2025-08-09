# Importation des bibliothèques nécessaires
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fonction d'entraînement
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []  # 📊 Stockage des métriques

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_acc = validate_model(model, val_loader)

        history.append({"epoch": epoch+1, "loss": avg_loss, "val_acc": val_acc})

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}%")
        torch.cuda.empty_cache()

    return history

# Fonction de validation
def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

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
    
def train_process(model_name, model, train_loader, val_loader, test_loader, epochs,result_queue):
    print(f"Starting training for {model_name}")

    history = train_model(model, train_loader, val_loader, epochs=epochs)
    acc = evaluate_model(model, test_loader)
    torch.cuda.empty_cache()

    print(f"Finished training {model_name} with Test Accuracy: {acc:.2f}%")

    # Envoi du résultat complet
    result_queue.put({
        "model_name": model_name,
        "test_accuracy": acc,
        "history": history
    })