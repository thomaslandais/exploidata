from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import autosklearn.classification
import sklearn.metrics

# Lire le fichier CSV
df = pd.read_csv('../data/img/data-204.csv')

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
X_name = vectorizer.fit_transform(df['name']).toarray()

# Combinaison de toutes les caractéristiques en une seule matrice
df_numeric_features = df[['file_extension', 'CREATION_DATETIME']]
X = np.hstack((X_name, df_numeric_features))

# Conversion des étiquettes de catégorie en représentation numérique
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category_name'])

# Séparation des données en ensembles d'entraînement et de test de manière stratifiée
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Entraînement du modèle
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60*8, per_run_time_limit=30)
automl.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
predictions = automl.predict(X_test)

# Evaluation du modèle
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))

# Affichage du modèle
print(automl.show_models())
# Affichage des résultats
print(automl.sprint_statistics())


import matplotlib.pyplot as plt

# Récupérer les résultats de la validation croisée
cv_results = automl.cv_results_

# Créer un dataframe à partir des résultats de la validation croisée
df_cv_results = pd.DataFrame.from_dict(cv_results)

# Extraire le score moyen de test et les noms des modèles
mean_test_scores = df_cv_results['mean_test_score']
model_names = df_cv_results['param_classifier:__choice__']

# Créer un dataframe avec les scores et les noms des modèles
df_scores = pd.DataFrame({
    'Model': model_names,
    'Score': mean_test_scores
})

# Trier les modèles par score
df_scores_sorted = df_scores.sort_values(by='Score', ascending=False)

# Créer un graphique à barres
plt.figure(figsize=(10, 6))
plt.barh(df_scores_sorted['Model'], df_scores_sorted['Score'], color='skyblue')
plt.xlabel('Mean Test Score')
plt.title('Model Performance')
plt.gca().invert_yaxis()
plt.show()
