import streamlit as st
import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

st.title('Myth Detector')

user_input=st.text_input('Entre Text For Checking')

with open('Lemetizer.pkl','rb') as f:
    lematizer=pickle.load(f)
    
wv_model=Word2Vec.load('Word2Vec.model')

stopword=set(stopwords.words('english'))

def preprocessor(df):
    words=[]
    for i in df['message']:
        text=simple_preprocess(str(i))
        filtered=[lematizer.lemmatize(word) for word in text if word not in stopword]
        words.append(filtered)

    return words

def avg_word2vec(sentences,wv_model):
    vectors=[]
    for token in sentences:
        
        valid=[word for word in token if word in wv_model.wv.index_to_key]

        if not valid:
            vectors.append(np.zeros(wv_model.vector_size))
        
        else:
            vectors.append(np.mean([wv_model.wv[word] for word in valid],axis=0)) 

    return np.array(vectors)

if st.button('Check Karo'):
    df=pd.DataFrame({'message':[user_input]})

    sentence=preprocessor(df)

    vector=avg_word2vec(sentence,wv_model)

    dl_model=load_model('ANN_Model.h5')

    pred=dl_model.predict(vector)[0][0]
    pred_class=int(pred>=0.5)

    if pred_class==0:
        pred_class='Fully Jhut'
    else:
        pred_class='Reality Hai Bhai Maan Le'


    st.success(f'Prediction is -->>{pred}')
    st.success(f'Output is -->> {pred_class}')
