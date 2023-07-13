from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

# Lire le fichier CSV
df = pd.read_csv('../data/img/data-4.csv')

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
from scipy.sparse import hstack
X = hstack((X_name, df[['file_extension', 'CREATION_DATETIME']]))

# Conversion des étiquettes de catégorie en représentation numérique
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category_name'])

# Séparation des données en ensembles d'entraînement et de test de manière stratifiée
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

param_grid = {
    'n_estimators': [10,50,100,200,500],
    'max_depth': [None, 5, 10, 100],
    'min_samples_split': [2, 20, 50, 100],
    'min_samples_leaf': [1, 2, 50, 100],
    'bootstrap': [True, False]
}

# Entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)

grid_search = GridSearchCV(model, param_grid, cv=5, verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

#print best parameters 
print("Best params : ",grid_search.best_params_)
#print accuracy
print("Best accuracy : ",grid_search.best_score_)
#print best model
print(grid_search.best_estimator_)
# Predict on the test set
predictions = grid_search.predict(X_test)

model.set_params(**grid_search.best_params_)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Accuracy on test set :", accuracy)


