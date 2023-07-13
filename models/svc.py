import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Read the CSV file
df = pd.read_csv('../data/pdf/data-186.csv')

# Convert the 'CREATION_DATETIME' to datetime
df['CREATION_DATETIME'] = pd.to_datetime(df['CREATION_DATETIME'])
df['CREATION_DATETIME'] = (df['CREATION_DATETIME'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Features and target
X = df[['name','file_extension','title','CREATION_DATETIME','Number_of_pages','file_size','content']]
y = df['category_name']

# Replace NaNs in text features with an empty string
X['name'].fillna('', inplace=True)
X['title'].fillna('', inplace=True)
X['content'].fillna('', inplace=True)
X = X.fillna('')


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing for numerical data
numeric_features = ['CREATION_DATETIME', 'Number_of_pages', 'file_size']
numeric_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_features = ['file_extension']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocessing for text data
text_features = ['name', 'title', 'content']
text_transformer = TfidfVectorizer()

# Bundle preprocessing for numerical, categorical and text data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('txt1', text_transformer, text_features[0]),
        ('txt2', text_transformer, text_features[1]),
        ('txt3', text_transformer, text_features[2])
    ])

from sklearn.model_selection import GridSearchCV

# Define the model
model = SVC()

# Bundle preprocessing and modeling code in a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Define parameter grid
param_grid = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf'],
    'model__gamma': [0.1, 1, 10, 'scale']
}

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best params: ", grid_search.best_params_)

# Print accuracy
print("Best accuracy: ", grid_search.best_score_)

# Make predictions with the best model
y_pred = grid_search.predict(X_test)

# Print accuracy on the test set
print("Accuracy on test set : ", grid_search.score(X_test, y_test))

