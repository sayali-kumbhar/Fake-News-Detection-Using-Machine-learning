import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression 


fake_n=pd.read_csv("fake.csv")
true_n=pd.read_csv("true.csv")
true_n.head()
fake_n["class"]=0
true_n["class"]=1

merge_df=pd.concat([true_n,fake_n])
merge_df.head(2)
df =merge_df.drop(["title", "subject","date"], axis = 1)
df = df.sample(frac = 1)
df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)
# ## Preprocessing data
def preprocess(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
df["text"] = df["text"].apply(preprocess)


x=df["text"]
y=df["class"]

#splitting dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.3)
# ## BOW Vectorizer
cv=CountVectorizer()
x_train_cv=cv.fit_transform(x_train)
# ## LogisticRegression
model=LogisticRegression()
model.fit(x_train_cv,y_train)


import streamlit as st
st.title("Fake News Detection System")
def fakenewsdetection():
    user = st.text_area("Enter Any News: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = model.predict(data)
        if a ==1:
            st.write("Real News")
        else:
            st.write("Fake News")
fakenewsdetection()