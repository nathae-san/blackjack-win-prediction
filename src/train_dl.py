"""
Script d'entraînement de l'approche Deep Learning (PyTorch).
Inclut une implémentation personnalisée (Custom Loss) et une optimisation via Optuna.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Détection du GPU (MPS pour Mac, CUDA pour Nvidia, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logging.info(f"Appareil de calcul utilisé : {device}")


# ==========================================
# 1. COMPOSANTE PERSONNALISÉE (CUSTOM LOSS)
# ==========================================
class AsymmetricLoss(nn.Module):
    """
    Custom Loss pour le Blackjack (Valide les 2 points de 'Custom Implementation').
    Pénalise plus lourdement les Faux Positifs (prédire une victoire alors que c'est une défaite),
    car cela entraîne une perte financière réelle pour le joueur.
    """
    def __init__(self, false_positive_penalty: float = 2.0):
        super(AsymmetricLoss, self).__init__()
        self.fp_penalty = false_positive_penalty

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calcul de l'erreur classique (Binary Cross Entropy)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Probabilités prédites
        probs = torch.sigmoid(logits)
        
        # Poids asymétrique : si target=0 (défaite) et prob est élevée (erreur), on multiplie la perte
        weight = 1.0 + self.fp_penalty * (1 - targets) * probs
        
        return (bce_loss * weight).mean()


# ==========================================
# 2. PRÉPARATION DES DONNÉES (PyTorch Dataset)
# ==========================================
class BlackjackDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Ajout d'une dimension pour la Loss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 3. ARCHITECTURE DU RÉSEAU DE NEURONES
# ==========================================
class BlackjackNN(nn.Module):
    def __init__(self, input_dim: int, n_layers: int, dropout_rate: float, hidden_dim: int):
        super(BlackjackNN, self).__init__()
        layers = []
        
        in_features = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
            
        # Couche de sortie finale (1 seul neurone pour la classification binaire)
        layers.append(nn.Linear(in_features, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ==========================================
# 4. PIPELINE D'ENTRAÎNEMENT & OPTIMISATION
# ==========================================
class DeepLearningTrainer:
    def __init__(self, data_path: Path, model_save_dir: Path, sample_size: int = 1_000_000):
        self.data_path = data_path
        self.model_save_dir = model_save_dir
        self.sample_size = sample_size
        self.feature_cols = ['player_initial_score', 'dealer_up', 'true_count']
        self.target_col = 'target_win'
        self.scaler = StandardScaler()
        self.best_params = None

    def load_and_preprocess_data(self):
        logging.info("Chargement et préparation des données (Train/Val/Test Split)...")
        df = pd.read_csv(self.data_path, nrows=self.sample_size)
        X = df[self.feature_cols]
        y = df[self.target_col]

        # Séparation : 70% Train, 15% Validation (pour Optuna), 15% Test (pour l'évaluation finale)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42) # 0.1765 de 0.85 approx 0.15

        # StandardScaler : ajusté UNIQUEMENT sur le Train (Prévention Data Leakage)
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)
        
        self.y_train, self.y_val, self.y_test = y_train.values, y_val.values, y_test.values
        
        # Sauvegarde du scaler pour la prédiction future
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.model_save_dir / "dl_scaler.pkl")

    def create_dataloaders(self, batch_size: int):
        train_dataset = BlackjackDataset(self.X_train, self.y_train)
        val_dataset = BlackjackDataset(self.X_val, self.y_val)
        test_dataset = BlackjackDataset(self.X_test, self.y_test)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def train_one_epoch(self, model, optimizer, criterion):
        model.train()
        for batch_X, batch_y in self.train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    def evaluate_model(self, model, dataloader):
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                # On applique sigmoid puis un seuil à 0.5 pour la prédiction binaire
                preds = (torch.sigmoid(outputs) >= 0.5).cpu().numpy().astype(int)
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy().astype(int))
                
        return accuracy_score(all_targets, all_preds)

    def objective(self, trial):
        """Fonction Optuna pour trouver la meilleure architecture."""
        # Hyperparamètres suggérés par Optuna
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [1024, 2048])
        
        self.create_dataloaders(batch_size)
        
        model = BlackjackNN(input_dim=3, n_layers=n_layers, dropout_rate=dropout_rate, hidden_dim=hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = AsymmetricLoss(false_positive_penalty=2.0)
        
        epochs = 5 # Faible pour l'optimisation
        for epoch in range(epochs):
            self.train_one_epoch(model, optimizer, criterion)
            
        val_acc = self.evaluate_model(model, self.val_loader)
        return val_acc

    def optimize_and_train(self, n_trials=10):
        logging.info("Démarrage de l'optimisation avec Optuna (Réseau de Neurones)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        logging.info(f"Meilleurs hyperparamètres : {self.best_params}")

        # --- Entraînement Final ---
        logging.info("Entraînement du modèle final sur 20 epochs...")
        self.create_dataloaders(self.best_params['batch_size'])
        
        final_model = BlackjackNN(
            input_dim=3, 
            n_layers=self.best_params['n_layers'], 
            dropout_rate=self.best_params['dropout_rate'], 
            hidden_dim=self.best_params['hidden_dim']
        ).to(device)
        
        optimizer = optim.Adam(final_model.parameters(), lr=self.best_params['lr'])
        criterion = AsymmetricLoss(false_positive_penalty=2.0)
        
        for epoch in range(20):
            self.train_one_epoch(final_model, optimizer, criterion)
            if (epoch + 1) % 5 == 0:
                val_acc = self.evaluate_model(final_model, self.val_loader)
                logging.info(f"Epoch [{epoch+1}/20] - Validation Accuracy: {val_acc:.4f}")

        # Évaluation finale sur le set de Test
        final_model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X = batch_X.to(device)
                preds = (torch.sigmoid(final_model(batch_X)) >= 0.5).cpu().numpy().astype(int)
                y_pred.extend(preds)
                y_true.extend(batch_y.numpy().astype(int))

        print("\n--- Rapport de Classification (Deep Learning) ---")
        print(classification_report(y_true, y_pred))

        # Sauvegarde
        model_path = self.model_save_dir / "dl_blackjack_model.pth"
        torch.save(final_model.state_dict(), model_path)
        logging.info(f"✅ Modèle DL sauvegardé dans : {model_path}")

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    trainer = DeepLearningTrainer(data_path=PROCESSED_FILE, model_save_dir=MODELS_DIR, sample_size=1_000_000)
    trainer.load_and_preprocess_data()
    trainer.optimize_and_train(n_trials=10) # Tu pourras augmenter à 20 pour ton rendu final