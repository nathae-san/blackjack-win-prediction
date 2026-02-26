#  Projet Machine Learning : Prédiction de Victoire au Blackjack

Ce dépôt contient l'intégralité du pipeline de Machine Learning développé dans le cadre de notre projet académique. L'objectif est de prédire l'issue d'une main de Blackjack (Victoire vs Défaite/Égalité) à partir d'un dataset massif de 50 millions de mains simulées.

##  Structure du Projet

Le projet respecte une architecture modulaire stricte pour garantir une qualité professionnelle :

* `data/` : Contient les données brutes (`raw/`) téléchargées depuis Kaggle et les données nettoyées (`processed/`). *(Dossier ignoré par Git pour des raisons de poids).*
* `models/` : Contient les modèles entraînés et sauvegardés (`.pkl`, `.pth`) ainsi que les scalers.
* `notebooks/` : Contient le notebook `EDA.ipynb` dédié exclusivement à l'exploration, au nettoyage initial et à la visualisation des données.
* `src/` : Contient les scripts Python `.py` modulaires constituant le pipeline d'apprentissage.
* `requirements.txt` : Liste figée des dépendances et de leurs versions exactes pour exécuter le projet de manière reproductible.

##  Comment exécuter le pipeline ?

Pour reproduire nos résultats, veuillez exécuter les commandes suivantes dans cet ordre précis depuis la racine du projet :

**1. Installation des dépendances**
```bash
pip install -r requirements.txt
```

**2. Téléchargement des données**
Télécharge le dataset brut depuis Kaggle et le place dans le dossier `data/raw/`.
```bash
python src/download_data.py
```

**3. Prétraitement des données (Feature Engineering & Nettoyage)**
Traite les données brutes par morceaux (chunks) pour optimiser la mémoire et génère le dataset prêt pour l'entraînement dans `data/processed/`.
```bash
python src/preprocess.py
```

**4. Entraînement du Modèle Classique (LightGBM)**
Entraîne un modèle LightGBM avec validation croisée stratifiée (Stratified K-Fold) et optimisation des hyperparamètres via Optuna.
```bash
python src/train_classic.py
```

**5. Entraînement du Modèle Deep Learning (PyTorch)**
Entraîne un réseau de neurones multicouche (MLP) avec une architecture (couches, neurones, learning rate) optimisée par Optuna.
*  **Implémentation Personnalisée (Custom) :** Ce modèle utilise une fonction de perte asymétrique (`AsymmetricLoss`) créée sur mesure. Elle pénalise volontairement plus lourdement les faux positifs (prédictions de victoire entraînant une perte financière réelle pour le joueur) afin de simuler une aversion au risque.
```bash
python src/train_dl.py
```

##  Auteurs
* **[SANTERRE Nathaé]**
* **[HAVERLAND Clément]**