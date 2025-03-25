# Hieroglyphs Classification
_Projet résenté dans le cadre du cours GLO-7030 : Apprentissage par réseaux de neurones profonds._

## Résumé
Ce projet vise à développer un modèle de Deep Learning pour identifier et classifier les hiéroglyphes égyptiens. Si le temps nous le permet, nous pourrions éventuellement nous intéresser à des tâches comme extraire le thème ou traduire un texte en fonction des hiéroglyphes détectés. 

En utilisant un jeu de données d’images de hiéroglyphes, nous commencerons par comparer deux architectures afin de classifier les symboles. Puis, nous expérimenterons sur des techniques et des variantes que nous jugerons intéressantes, telles que l’augmentation des données et l’apprentissage semi-supervisé... Enfin, nous aimerions effectuer une analyse d’ablation dans l’optique d’évaluer l’impact de chaque approche.

## 1 Introduction
Les hiéroglyphes égyptiens représentent une forme ancienne d’écriture picturale complexe. Leur identification et leur interprétation peuvent fournir des informations précieuses sur les textes anciens, mais le processus est long et nécessite une expertise humaine approfondie. Ce projet se propose de développer un modèle capable d’identifier ces symboles à l’aide de techniques de Deep Learning, posant les bases pour des analyses plus complexes comme la reconstruction de textes effacés, l’étude des structures sémantiques, traduction assistée... Cette approche s’inscrit donc dans une perspective où la reconnaissance des hiéroglyphes devient un outil facilitant des explorations plus approfondies des inscriptions égyptiennes.

## 2 Description de l’approche proposée
### 2.1 Choix du jeu de données :
Nous choisirons un jeu de données d’images de hiéroglyphes égyptiens annotées, tel que le jeu de données "EgyptianHieroglyphDataset Computer Vision Project"[1]. Nous pourrions également récupérer d’autres jeux d’images non annotées disponibles en ligne pour de l’apprentissage semi-supervisé.

### 2.2 Identification des hiéroglyphes :
Nous voudrions implémenter deux modèles : un réseau de neurones convolutionnels (CNN) et un VisionTransformer pour identifier et classifier avec les annotations de Gardiner [2] les différents hiéroglyphes des images. Nous pourrions également explorer l’utilisation de modèles pré-entraînés.

### 2.3 Entraînement du modèle et régularisation :
Nous appliquerons des techniques d’augmentation des données pour améliorer la robustesse du modèle et testerons les méthodes de régularisation qui se révéleront nécessaires (dropout, batch normalization ou autres) afin d’éviter le sur-apprentissage. Nous pourrions également étudier l’impact de l’apprentissage semi-supervisé en utilisant des données non-étiquetées pour compléter le jeu de données d’entraînement.

### 2.4 Évaluation des résultats :
Les performances du modèle seront évaluées en termes de précision, rappel et F1-score et éventuellement d’autres métriques que nous jugerons utiles pour l’identification des hiéroglyphes.

## Références
[1] Jeu de données contenant 3 584 images de hiéroglyphes égyptiens, organisées en 170 classes, https://universe.roboflow.com/sameh-zaghloul/egyptianhieroglyphdataset  
[2] Liste des signes et annotations utilisées pour la classification, https://www.egyptianhieroglyphs.net/gardiners-sign-list/
