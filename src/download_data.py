"""
Script de téléchargement du jeu de données Blackjack depuis Kaggle.
Ce script rapatrie les données brutes dans le dossier data/raw/.
"""
import shutil
from pathlib import Path
import kagglehub

def download_blackjack_data() -> None:

    
    # Résolution dynamique des chemins basée sur l'emplacement de ce script (src/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    raw_data_dir = project_root / "data" / "raw"
    
    # Création du dossier cible s'il n'existe pas
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print("⏳ Téléchargement du dataset depuis Kaggle...")
    # Le téléchargement via kagglehub se met d'abord en cache
    cache_path = Path(kagglehub.dataset_download("dennisho/blackjack-hands"))

    print(f" Copie des fichiers vers {raw_data_dir}...")
    # Déplacement des fichiers du cache vers notre dossier local
    for item in cache_path.iterdir():
        dest = raw_data_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print("✅ Jeu de données importé avec succès. Fichiers disponibles :")
    for file_path in raw_data_dir.rglob("*"):
        if file_path.is_file():
            print(f"  - {file_path.name}")

if __name__ == "__main__":
    download_blackjack_data()