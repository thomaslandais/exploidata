from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from scipy.sparse import hstack
import matplotlib.pyplot as plt

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
X = hstack((X_name, df[['file_extension', 'CREATION_DATETIME']].values))

# Conversion des étiquettes de catégorie en représentation numérique
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category_name'])

# Séparation des données en ensembles d'entraînement et de test de manière stratifiée
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# List to store the accuracy for each value of n_neighbors
accuracy_list = []

# Range of n_neighbors values to test
neighbors_range = range(1, 20)

for n in neighbors_range:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    print("n_neighbors = %d, accuracy=%.2f%%" % (n, accuracy * 100))

# Find the value of n_neighbors that gave the highest accuracy
best_n = neighbors_range[accuracy_list.index(max(accuracy_list))]

# Plot accuracy vs n_neighbors
plt.figure(figsize=(12, 6))
plt.plot(neighbors_range, accuracy_list, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=5)
# Highlight the best value
plt.plot(best_n, max(accuracy_list), color='red', marker='o', markersize=10)
plt.title('Accuracy vs. Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

print("Best n_neighbors: ", best_n)
print("Best accuracy: ", max(accuracy_list))
