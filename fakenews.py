import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
vectorizer = TfidfVectorizer(min_df=3, stop_words='english',ngram_range=(1, 2))
X = vectorizer.fit_transform(train['text'].values.astype('U'))
y = np.array(train['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1234)

#nb = MultinomialNB()
#nb.fit(X_train,y_train)
#print("Multinomial NB: ", end="")
#print(nb.score(X_test,y_test))

lr = LogisticRegression(penalty='l2',C=1, solver='lbfgs', multi_class='multinomial')
lr.fit(X_train,y_train)
print("Logistic Regression: ", end="")
print(lr.score(X_test,y_test))

#vectorizer = TfidfVectorizer(min_df=3, stop_words='english',ngram_range=(1, 2))
X_test = vectorizer.transform(test['text'].values.astype('U'))
lr_preds = lr.predict(X_test)
ids = test.as_matrix(columns=test.columns[:1])
ids = pd.DataFrame(ids)
labels = pd.DataFrame(lr_preds)
merged = pd.concat([ids,labels], axis=1, keys=['id','label'])
merged.to_csv("predictions.csv")
print("Done")

#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(X_train,y_train)
#print("Random Forest: ", end="")
#print(rf.score(X_test,y_test))

#nb_preds = nb.predict(X_test)
#print(nb_preds)
#lr_preds = lr.predict(X_test)
#print(lr_preds)
#rf_preds = rf.predict(X_test)
#print(rf_preds)
