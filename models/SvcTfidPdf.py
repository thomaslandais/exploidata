import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Read the CSV file
df = pd.read_csv('../data/pdf/data-5.csv')

# Replace np.nan values with empty strings
df['content'] = df['content'].fillna('')
df['title'] = df['title'].fillna('')

# Convert the 'CREATION_DATETIME' to datetime
df['CREATION_DATETIME'] = pd.to_datetime(df['CREATION_DATETIME'])

# Extract features from the 'CREATION_DATETIME'
df['year'] = df['CREATION_DATETIME'].dt.year
df['month'] = df['CREATION_DATETIME'].dt.month
df['day'] = df['CREATION_DATETIME'].dt.day
df['hour'] = df['CREATION_DATETIME'].dt.hour
df['minute'] = df['CREATION_DATETIME'].dt.minute
df['second'] = df['CREATION_DATETIME'].dt.second
df['day_of_week'] = df['CREATION_DATETIME'].dt.dayofweek

# Features and target
X = df[['title', 'content', 'file_extension', 'year', 'month', 'day', 'hour', 'minute', 'second', 'day_of_week', 'Number_of_pages', 'file_size']]
y = df['category_name']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline: TF-IDF Vectorizer and RandomForest Classifier
preprocessor = ColumnTransformer(
    transformers=[
        ('title_tfidf', TfidfVectorizer(), 'title'),
        ('content_tfidf', TfidfVectorizer(), 'content'),
        ('file_extension_ohe', OneHotEncoder(), ['file_extension']),
        ('date_std', StandardScaler(), ['year', 'month', 'day', 'hour', 'minute', 'second', 'day_of_week']),
        ('number_of_page_std', StandardScaler(), ['Number_of_pages']),
        ('file_size_std', StandardScaler(), ['file_size']),
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', SVC())])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Print accuracy on the test set
print("Accuracy on test set : ", pipeline.score(X_test, y_test))
