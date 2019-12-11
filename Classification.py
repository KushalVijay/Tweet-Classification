import pandas as pd
import re
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

pipeline_sgd = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('nb',SGDClassifier()),
])

train = pd.read_csv('Tweet Classification/train.csv')
print("Training Set:" % train.columns,train.shape,len(train))

test = pd.read_csv('Tweet Classification/test.csv')
print("Test Set:"% test.columns,test.shape,len(test))

def clean_text(df,text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+) | ([^0-9A-Za-z \t]) | (\w+:\/\/\S+) | ^rt | http.+?", "",elem))
    return df

test_clean = clean_text(test,"tweet")
train_clean = clean_text(train,"tweet")

# Upsampling : We repeatedly takes samples with replacement from minority class until
# the class is the same size as the majority
train_majority = train_clean[train_clean.label==0]
train_minority = train_clean[train_clean.label==1]

train_minority_upsampled = resample(train_minority,replace=True,n_samples=len(train_majority),random_state=123)

train_upsampled = pd.concat([train_minority_upsampled,train_majority])
train_upsampled['label'].value_counts()


train_majority = train_clean[train_clean.label==0]
train_minority = train_clean[train_clean.label==1]

train_majority_downsampled = resample(train_majority,replace=True,n_samples=len(train_minority),random_state=123)

train_downsampled = pd.concat([train_majority_downsampled,train_minority])
train_downsampled['label'].value_counts()


X_train,X_test,y_train,y_test =train_test_split(train_upsampled['tweet'],train_upsampled['label'],random_state = 0)

model = pipeline_sgd.fit(X_train,y_train)

y_predict = model.predict(X_test)

print(f1_score(y_test,y_predict))
