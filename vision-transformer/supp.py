import os
import shutil

# Liste des classes à supprimer
classes_to_delete = [
    "Aa15", "Aa27", "Aa28", "D1", "D10", "D156", "D19", "D34", "D39", "D52", "D53", "D56",
    "E1", "E17", "F12", "F16", "F21", "F22", "F23", "F26", "F29", "F31", "F35", "F40", "F9",
    "G10", "G14", "G21", "G26", "G29", "G37", "G40", "G50", "L1", "M1", "M12", "M16", "M195",
    "M26", "M29", "M3", "M4", "M40", "M41", "M42", "N16", "N17", "N19", "N2", "N25", "N26",
    "N41", "O11", "O28", "O29", "O31", "O51", "P1", "P13", "P6", "P98", "Q7", "R4", "S24",
    "S42", "T14", "T20", "T21", "T22", "T28", "T30", "U28", "U7", "V22", "V24", "V25", "V6",
    "W11", "W14", "W15", "W18", "W19", "W22", "X6", "Y1", "Y2", "Y3", "Y5", "Z7"
]

# Répertoires à nettoyer
root_dirs = [
    "./EgyptianHieroglyphDataset-1/train",
    "./EgyptianHieroglyphDataset-1/valid",
    "./EgyptianHieroglyphDataset-1/test"
]

for root in root_dirs:
    print(f"\n🔍 Vérification dans : {root}")
    for cls in classes_to_delete:
        cls_path = os.path.join(root, cls)
        if os.path.isdir(cls_path):
            shutil.rmtree(cls_path)
            print(f"🗑️ Supprimé : {cls}")
        else:
            print(f"⚠️ Classe absente : {cls}")

print("\n✅ Suppression terminée.")
