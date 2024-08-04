import pickle

import pandas as pd

data_reddit = pd.read_csv('Reddit_Data.csv')
data_twitter = pd.read_csv('Twitter_Data.csv')
df_red = pd.DataFrame(data_reddit)
df_red = df_red.rename({'clean_comment': 'clean_text'}, axis='columns')
df_twt = pd.DataFrame(data_twitter)
df = pd.concat([df_red,df_twt])
df.dropna(inplace=True)
print(df.info())

def labelling (x):
  if x==1.0:
    return 'pos'
  elif x==-1.0:
    return 'neg'
  else:
    return 'neu'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

df['label'] = df['category'].apply(labelling)
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC())])
text_clf.fit(X_train, y_train)

pickle.dump(text_clf, open('ml_model.pkl', 'wb'))
