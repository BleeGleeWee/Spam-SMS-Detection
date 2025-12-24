import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# 1. Loading dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
df.columns = ['label', 'text']

# 2. Preprocess
df['transformed_text'] = df['text'].apply(transform_text)

# 3. Vectorize
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['transformed_text'])

# 4. Labels
y = df['label'].map({'ham': 0, 'spam': 1})

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluate (optional)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. Save model and vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

