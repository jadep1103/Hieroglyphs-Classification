import os
import shutil
from collections import defaultdict

# === PARAMÈTRES ===
base_dir = "./EgyptianHieroglyphDataset-1"
splits = ["train", "valid", "test"]
output_prefix = "new_"

# === ÉTAPE 0 : Trouver l'intersection des classes ===
split_classes = {}
for split in splits:
    split_dir = os.path.join(base_dir, split)
    classes = set([
        cls for cls in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, cls))
    ])
    split_classes[split] = classes
    print(f"{split} contient {len(classes)} classes.")

common_classes = set.intersection(*split_classes.values())
print(f"\n Classes communes aux 3 splits : {len(common_classes)}\n")

# === ÉTAPE 1 : Compter le total d’images global par classe (seulement les communes) ===
global_class_counts = defaultdict(int)
for split in splits:
    split_dir = os.path.join(base_dir, split)
    for cls in common_classes:
        cls_path = os.path.join(split_dir, cls)
        if os.path.isdir(cls_path):
            n_images = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            global_class_counts[cls] += n_images

# === ÉTAPE 2 : Trier les classes par fréquence décroissante ===
sorted_classes = sorted(global_class_counts.items(), key=lambda x: -x[1])
top_classes = [cls for cls, _ in sorted_classes]

print(" Classes communes triées par nombre total d'images :")
for cls, count in sorted_classes:
    print(f" - {cls}: {count} images")

# === PARAMÈTRE : combien de classes garder ?
N = 40
selected_classes = top_classes[:N]

print(f"\n✅ On garde les {N} classes les plus représentées.\n")

# === ÉTAPE 3 : Copier les images dans de nouveaux dossiers ===
for split in splits:
    src_split_dir = os.path.join(base_dir, split)
    dst_split_dir = os.path.join(base_dir, f"{output_prefix}{split}")
    os.makedirs(dst_split_dir, exist_ok=True)

    for cls in selected_classes:
        src_cls_dir = os.path.join(src_split_dir, cls)
        dst_cls_dir = os.path.join(dst_split_dir, cls)
        if os.path.isdir(src_cls_dir):
            os.makedirs(dst_cls_dir, exist_ok=True)
            for fname in os.listdir(src_cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    src_file = os.path.join(src_cls_dir, fname)
                    dst_file = os.path.join(dst_cls_dir, fname)
                    shutil.copy2(src_file, dst_file)

print(" Copie des classes sélectionnées terminée.")
print(f"Résultat : {output_prefix}train/, {output_prefix}valid/, {output_prefix}test/")

# === ÉTAPE 4 : Vérification finale ===
print("\n Vérification des nouvelles classes...")
final_class_sets = []
for split in splits:
    path = os.path.join(base_dir, f"{output_prefix}{split}")
    classes = sorted([
        cls for cls in os.listdir(path)
        if os.path.isdir(os.path.join(path, cls))
    ])
    final_class_sets.append(classes)
    print(f"{output_prefix}{split} contient {len(classes)} classes.")

assert all(c == final_class_sets[0] for c in final_class_sets), "💥 Mismatch entre les splits !"
print(" Tous les splits contiennent exactement les mêmes 40 classes.")
