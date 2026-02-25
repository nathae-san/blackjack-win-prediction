"""
Script d'entraînement de l'approche Classique (LightGBM).
Inclut une optimisation des hyperparamètres via Optuna et une validation croisée.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

# Configuration du logging pour un affichage professionnel
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClassicModelTrainer:
    """
    Classe gérant l'entraînement, l'optimisation et l'évaluation du modèle classique.
    """
    
    def __init__(self, data_path: Path, model_save_path: Path, sample_size: int = 2_000_000):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.sample_size = sample_size
        self.feature_cols = ['player_initial_score', 'dealer_up', 'true_count']
        self.target_col = 'target_win'
        self.best_params = None

    def load_and_split_data(self):
        """
        Charge un échantillon représentatif et sépare les données (Train / Test).
        La séparation stratifiée garantit l'absence de Data Leakage.
        """
        logging.info(f"Chargement des données depuis {self.data_path} (max {self.sample_size} lignes)...")
        # On charge un échantillon pour des raisons de mémoire et de temps de calcul
        df = pd.read_csv(self.data_path, nrows=self.sample_size)
        
        X = df[self.feature_cols]
        y = df[self.target_col]

        # Séparation stricte : 80% pour la recherche/entraînement, 20% pour le test final
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logging.info(f"Données d'entraînement : {self.X_train.shape[0]} lignes.")
        logging.info(f"Données de test : {self.X_test.shape[0]} lignes.")

    def objective(self, trial):
        """
        Fonction objectif pour Optuna : définit l'espace de recherche et évalue
        le modèle avec une validation croisée stratifiée.
        """
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'verbose': -1
        }

        # Stratified K-Fold pour une validation robuste
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**params, random_state=42)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)]
            )
            
            preds = model.predict(X_fold_val)
            cv_scores.append(accuracy_score(y_fold_val, preds))

        # On retourne la moyenne des scores de validation
        return np.mean(cv_scores)

    def optimize(self, n_trials: int = 15):
        """Lance la recherche d'hyperparamètres avec Optuna."""
        logging.info("Démarrage de l'optimisation des hyperparamètres via Optuna...")
        
        # On supprime l'affichage bavard d'Optuna pour garder une console propre
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        logging.info(f"Meilleurs paramètres trouvés : {self.best_params}")
        logging.info(f"Meilleur score de validation (Accuracy) : {study.best_value:.4f}")

    def train_and_evaluate_final_model(self):
        """Entraîne le modèle final avec les meilleurs paramètres et l'évalue sur le test set."""
        logging.info("Entraînement du modèle final sur l'ensemble des données d'entraînement...")
        
        # Ajout des paramètres fixes
        final_params = self.best_params.copy()
        final_params.update({'objective': 'binary', 'random_state': 42, 'verbose': -1})
        
        final_model = lgb.LGBMClassifier(**final_params)
        final_model.fit(self.X_train, self.y_train)

        logging.info("Évaluation sur le jeu de test (inédit)...")
        y_pred = final_model.predict(self.X_test)
        
        print("\n--- Rapport de Classification ---")
        print(classification_report(self.y_test, y_pred))
        
        # Sauvegarde du modèle
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, self.model_save_path)
        logging.info(f"✅ Modèle sauvegardé avec succès dans : {self.model_save_path}")

    def run(self):
        """Exécute l'intégralité du pipeline d'entraînement."""
        self.load_and_split_data()
        self.optimize(n_trials=15) # Tu pourras augmenter n_trials à 50 pour le rendu final
        self.train_and_evaluate_final_model()


if __name__ == "__main__":
    # Résolution dynamique des chemins
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
    MODEL_FILE = PROJECT_ROOT / "models" / "lightgbm_classic.pkl"
    
    # Exécution du script
    trainer = ClassicModelTrainer(
        data_path=PROCESSED_FILE,
        model_save_path=MODEL_FILE,
        sample_size=2_000_000  # On s'entraîne sur 2 millions de lignes pour la robustesse
    )
    trainer.run()