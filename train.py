# Importation des bibliothèques nécessaires
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fonction d'entraînement
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-4):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    #PRINCIPAL
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #POUR VIT FINETUNED
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    #POUR VIT FROZEN
    #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    #LR Scheduler éventuel
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            # Si sortie auxiliaire (Inception-v3)
            if isinstance(outputs, tuple):
                main_output, aux_output = outputs
                loss_main = criterion(main_output, labels)
                loss_aux = criterion(aux_output, labels)
                loss = loss_main + 0.4 * loss_aux
                preds = main_output
            else:
                loss = criterion(outputs, labels)
                preds = outputs

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Précision d'entraînement
            _, predicted = torch.max(preds, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        val_loss, val_acc = validate_model(model, val_loader)

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

            # Gestion des sorties multiples
            if isinstance(outputs, tuple):
                main_output, _ = outputs
                loss = criterion(main_output, labels)
                preds = main_output
            else:
                loss = criterion(outputs, labels)
                preds = outputs

            running_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    acc = 100 * correct / total
    return avg_loss, acc



# Fonction d'évaluation finale
# Fonction d'évaluation finale
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                preds = outputs[0]  # main_output
            else:
                preds = outputs

            _, predicted = torch.max(preds, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {acc:.2f}%")

    report = classification_report(all_labels, all_preds, output_dict=True)
    f1_macro = report["macro avg"]["f1-score"]
    precision_macro = report["macro avg"]["precision"]
    recall_macro = report["macro avg"]["recall"]

    print(f"F1 (macro): {f1_macro:.4f}, Precision (macro): {precision_macro:.4f}, Recall (macro): {recall_macro:.4f}")

    return acc, report, all_labels, all_preds


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()

    os.makedirs("logs", exist_ok=True)
    path = f"logs/confusion_matrix_{model_name}.png"
    plt.savefig(path)
    plt.close()

    return path  # Pour wandb

def train_process(model_name, model, train_loader, val_loader, test_loader, epochs, result_queue):
    print(f"Starting training for {model_name}")

    history = train_model(model, train_loader, val_loader, epochs=epochs)
    acc, report, all_labels, all_preds = evaluate_model(model, test_loader)

    # === Résumé global (macro) ===
    macro_metrics = {
        "test_accuracy": acc,
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"]
    }

    # === Par classe ===
    class_metrics = {f"class_{label}_{metric}": value
                    for label, scores in report.items() if label.isdigit()
                    for metric, value in scores.items()}

    # Matrice de confusion
    class_names = [str(i) for i in sorted(set(all_labels))]
    cm_path = plot_confusion_matrix(all_labels, all_preds, class_names, model_name)

    # Log dans wandb
    if wandb.run:
        wandb.log({
            "confusion_matrix": wandb.Image(cm_path),
            **macro_metrics,
            **class_metrics
        })

    # Sauvegarde CSV du rapport complet
    df_report = pd.DataFrame(report).transpose()
    os.makedirs("logs", exist_ok=True)
    df_report.to_csv(f"logs/report_{model_name}.csv")

    # Historique
    df_history = pd.DataFrame(history)
    df_history.to_csv(f"logs/history_{model_name}.csv", index=False)

    results = {
        "model_name": model_name,
        **macro_metrics,
        **class_metrics,
        "history": history
    }

    result_queue.put(results)
