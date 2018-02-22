import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from nltk.stem.snowball import SnowballStemmer
import random

train = pd.read_csv('train.csv')
train = train.fillna(value="NA")
merged = []
for row in train.itertuples():
    merged.append(" ".join(row[2:5]))
train['merged'] = merged
stems = []
stemmer = SnowballStemmer("english", ignore_stopwords=True)
for row in train.itertuples():
    body = row[-1].split()
    words = [stemmer.stem(word) for word in body]
    stems.append(" ".join(words))
train["Stems"] = stems
print(train.head())

vectorizer = TfidfVectorizer(min_df=6, stop_words='english', ngram_range=(1, 5))
X = vectorizer.fit_transform(train['Stems'].values.astype('U'))
y = np.array(train['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=random.randint(1000, 9999))
lr = LogisticRegression(penalty='l2', solver='newton-cg', C=1000000)
model = lr.fit(X_train,y_train)
print(model.score(X_test,y_test))

test = pd.read_csv('test.csv')
test = test.fillna(value = "NA")
merged_test = []
for row in test.itertuples():
    merged_test.append(" ".join(row[2:5]))
test['merged'] = merged_test
stems = []
for row in test.itertuples():
    body = row[-1].split()
    words = [stemmer.stem(word) for word in body]
    stems.append(" ".join(words))
test["Stems"] = stems

X_validation = vectorizer.transform(test['Stems'].values.astype('U'))
preds = model.predict(X_validation)
df = test['id'].to_frame()
df['label'] = preds
print(df.head())
df.to_csv("preds3.csv", index=False)
print("Written")
