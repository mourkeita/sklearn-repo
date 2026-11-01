import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Charger le fichier CSV généré précédemment
df = pd.read_csv('donnees_immobilier.csv')

# 2. Séparer les fonctionnalités (X) et la cible à prédire (y)
X = df[['Superficie_m2', 'Nb_Chambres', 'Age_Ans', 'Distance_Centre_Km']]
y = df['Prix_Euros']

# 3. Diviser en lots d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialiser et entraîner le modèle de Régression Linéaire
modele = LinearRegression()
modele.fit(X_train, y_train)

# 5. Évaluer le modèle sur les données de test
y_pred = modele.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erreur moyenne absolue (MAE) : {mae:.2f} €")
print(f"Score R² (Précision globale) : {r2 * 100:.2f}%\n")

# 6. Bonus : Prédire le prix d'un tout nouvel appartement inconnu
# Un appartement de 85m², 3 chambres, 5 ans d'ancienneté, à 3.5 km du centre
nouvel_appart = pd.DataFrame([[85, 3, 5, 3.5]], columns=X.columns)
prix_predit = modele.predict(nouvel_appart)

print(f"Prix estimé pour cet appartement de 85m² : {prix_predit[0]:,.2f} €")
