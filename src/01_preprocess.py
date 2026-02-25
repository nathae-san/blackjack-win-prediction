"""
Script de prétraitement pour le projet Machine Learning - Blackjack.
Transforme les données brutes en un format exploitable pour l'entraînement,
en traitant le fichier par morceaux (chunks) pour optimiser la mémoire.
"""

import pandas as pd
import ast
from pathlib import Path

class BlackjackPreprocessor:
    """
    Classe chargée du nettoyage et du Feature Engineering des données de Blackjack.
    """
    
    def __init__(self, raw_data_path: Path, processed_data_path: Path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        
        # Sélection stricte des colonnes pour éviter le Data Leakage
        # On ne garde que ce qui est connu au début de la main
        self.feature_cols = ['player_initial_score', 'dealer_up', 'true_count']
        self.target_col = 'target_win'

    @staticmethod
    def calculate_initial_score(hand_str: str) -> int:
        """
        Convertit la chaîne de caractères de la main en liste et retourne la somme.
        Exemple : "[10, 11]" -> 21
        """
        try:
            # Sécurité : on vérifie que la valeur est bien une chaîne non nulle
            if pd.isna(hand_str):
                return 0
            hand_list = ast.literal_eval(hand_str)
            return sum(hand_list)
        except (ValueError, SyntaxError):
            return 0

    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Applique les transformations nécessaires à un morceau (chunk) du dataset.
        """
        # 1. Feature Engineering : Création du score joueur
        chunk['player_initial_score'] = chunk['initial_hand'].apply(self.calculate_initial_score)
        
        # 2. Simplification de la cible : Classification binaire (1 = Victoire, 0 = Perte/Égalité)
        chunk['target_win'] = (chunk['win'] > 0).astype(int)
        
        # 3. Filtrage des colonnes (pour éviter d'embarquer dealer_final_value etc.)
        columns_to_keep = self.feature_cols + [self.target_col]
        return chunk[columns_to_keep]

    def run(self, chunksize: int = 1_000_000):
        """
        Exécute le pipeline de prétraitement complet en lisant le fichier par morceaux.
        """
        print(f"🚀 Début du prétraitement depuis : {self.raw_data_path}")
        
        # Création du dossier cible s'il n'existe pas
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        first_chunk = True
        total_rows = 0
        
        # Lecture itérative pour ne pas saturer la RAM
        for chunk in pd.read_csv(self.raw_data_path, chunksize=chunksize):
            processed_chunk = self.process_chunk(chunk)
            
            # Sauvegarde dans le fichier CSV final
            # 'w' (write) pour le premier morceau, puis 'a' (append) pour ajouter à la suite
            processed_chunk.to_csv(
                self.processed_data_path, 
                mode='w' if first_chunk else 'a', 
                index=False, 
                header=first_chunk
            )
            
            total_rows += len(chunk)
            first_chunk = False
            print(f"✅ {total_rows:,} lignes traitées et sauvegardées...")
            
        print(f"Prétraitement terminé ! Fichier généré : {self.processed_data_path}")


if __name__ == "__main__":
    # Résolution dynamique des chemins depuis l'emplacement de preprocess.py
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    RAW_FILE = PROJECT_ROOT / "data" / "raw" / "blackjack_simulator.csv"
    PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
    
    # Instanciation et exécution
    preprocessor = BlackjackPreprocessor(RAW_FILE, PROCESSED_FILE)
    
    # On traite par blocs de 1 million de lignes (tu peux réduire à 500_000 si ton PC rame)
    preprocessor.run(chunksize=1_000_000)