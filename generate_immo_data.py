import pandas as pd
import numpy as np

# 1. Génération de données aléatoires mais cohérentes
np.random.seed(80)
n_échantillons = 100

donnees = {
    'Superficie_m2': np.random.randint(30, 150, size=n_échantillons),
    'Nb_Chambres': np.random.randint(1, 5, size=n_échantillons),
    'Age_Ans': np.random.randint(0, 60, size=n_échantillons),
    'Distance_Centre_Km': np.round(np.random.uniform(0.5, 15.0, size=n_échantillons), 1),
    'Propriétaire': np.random.choice(['Mour', 'Sokhna'], p=[0.5, 0.5], size=n_échantillons)
}

df = pd.DataFrame(donnees)

# 2. Création d'une colonne cible "Prix" liée logiquement aux caractéristiques + un bruit aléatoire
df['Prix_Euros'] = (
    df['Superficie_m2'] * 3500 +
    df['Nb_Chambres'] * 15000 -
    df['Age_Ans'] * 800 +
    (15 - df['Distance_Centre_Km']) * 5000 +
    np.random.normal(0, 15000, size=n_échantillons)
).astype(int)


# 3. Sauvegarde dans un fichier CSV
df.to_csv('donnees_immobilier.csv', index=False)
print("Fichier 'donnees_immobilier.csv' créé avec succès !")
print(df.head(100)) # Affiche les 5 premières lignes
