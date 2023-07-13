import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack

# Lire le fichier CSV
df = pd.read_csv('../data/pdf/data-186.csv')

df['file_size'] = df['file_size'].fillna(df['file_size'].mean())

counts = df['category_name'].value_counts()
df = df[df['category_name'].isin(counts[counts > 1].index)]

# Conversion des dates en format numérique (nombre de secondes depuis une date donnée)
df['CREATION_DATETIME'] = pd.to_datetime(df['CREATION_DATETIME'])
df['CREATION_DATETIME'] = (df['CREATION_DATETIME'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Conversion des extensions de fichier en représentation numérique
extension_encoder = LabelEncoder()
df['file_extension'] = extension_encoder.fit_transform(df['file_extension'])

# Extraction de caractéristiques du nom du fichier
vectorizer = CountVectorizer()
X_name = vectorizer.fit_transform(df['name'])

# Combinaison de toutes les caractéristiques en une seule matrice
X = hstack((X_name, df[['file_extension', 'CREATION_DATETIME']]))

# Conversion des étiquettes de catégorie en représentation numérique
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category_name'])

# Séparation des données en ensembles d'entraînement et de test de manière stratifiée
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# test avec un MLP
# mlp_model = MLPClassifier(max_iter=1000, random_state=42)
# mlp_model.fit(X_train, y_train)
# accuracy_mlp = mlp_model.score(X_test, y_test)
# print("Accuracy on test set :", accuracy_mlp)

# Grille de paramètres pour le MLP
param_grid_mlp = {
    'hidden_layer_sizes': [(10,), (50,),(200,)],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.1],
}

# Entraînement du modèle MLP
mlp_model = MLPClassifier(max_iter=1000, random_state=42)

mlp_grid_search = GridSearchCV(mlp_model, param_grid_mlp, cv=3, verbose=2, n_jobs=-1)

mlp_grid_search.fit(X_train, y_train)


# Afficher les meilleurs paramètres
print("Best params : ", mlp_grid_search.best_params_)
# Afficher la meilleure précision
print("Best accuracy : ", mlp_grid_search.best_score_)
# Afficher le meilleur modèle
print(mlp_grid_search.best_estimator_)

# Prédiction sur l'ensemble de test
predictions_mlp = mlp_grid_search.predict(X_test)

# Tester l'exactitude sur les données de test
mlp_model.set_params(**mlp_grid_search.best_params_)
mlp_model.fit(X_train, y_train)
accuracy_mlp = mlp_model.score(X_test, y_test)
print("Accuracy on test set :", accuracy_mlp)


