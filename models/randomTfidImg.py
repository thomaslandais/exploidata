import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Lire le fichier CSV
df = pd.read_csv('../data/img/data-3.csv')

# Remplacer les valeurs np.nan par des chaînes vides
df['content'] = df['name'] + ' ' + df['category_name']
df['content'] = df['content'].fillna('')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['category_name'], test_size=0.3, random_state=42)

# Create a pipeline: TF-IDF Vectorizer and RandomForest Classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier()),
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# print accuracy on test set
print("Accuracy on test set : ", pipeline.score(X_test, y_test))
