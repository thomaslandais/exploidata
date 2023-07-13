from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
import pandas as pd

# Lire le fichier CSV
df = pd.read_csv('../data/pdf/data-3.csv')
# MiniCatalogueLunettes,pdf,2020-12-02 15:06:28,1,854159,Mini Catalogue Lunettes Junior   Lunettes Decathlon 10€ Lunettes Unihoc Victory 30~35€ plusieurs modèles disponibles Lunettes Salming V1 ~32€ plusieurs modèles disponibles Lunettes Fat Pipe Protective ~28€ plusieurs modèles disponibles Prix au 25 septembre pris sur decathlon.fr et efloorball.net Il est possible de faire une commande groupée par l’intermédiaire du club pour profiter  d’une réduction et partager les frais de port.  Lunettes correctrices de sport  : Alexis Thébaud (Opticien et licencié AAEEC Floorball) peut vous conseiller sur les possibilités d'équipement à la vue. Contact par mail : alexis.thebaud@mfam.fr (Équipement complet à partir de 69€.) ,All - Thumb
# wench-double,pdf,2020-12-02 15:11:17,1,38681,,All - Thumb
# stamps-famous,pdf,2020-12-02 15:11:16,1,38606,,All - Thumb

# Conversion des dates en format numérique (nombre de secondes depuis une date donnée)
df['CREATION_DATETIME'] = pd.to_datetime(df['CREATION_DATETIME'])
df['CREATION_DATETIME'] = (df['CREATION_DATETIME'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Preprocess the data (assuming the necessary preprocessing steps)
df = df[['CREATION_DATETIME', 'content', 'category_name']].dropna()
X = df[['CREATION_DATETIME', 'content']]
y = df['category_name']

# Convert categorical variables to numerical representations
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the feature vectors
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train['content'])
X_test_vectors = vectorizer.transform(X_test['content'])

# Train the Naive Bayes classifier
clf = GaussianNB()
clf.fit(X_train_vectors.toarray(), y_train)

# Predict on the test set
predictions = clf.predict(X_test_vectors.toarray())

# Evaluate the model
accuracy = (predictions == y_test).mean()
print(f"Accuracy: {accuracy*100:.2f}%")

