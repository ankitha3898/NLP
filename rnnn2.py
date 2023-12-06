import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Embedding,Dense,SimpleRNN
from keras.models import Sequential
data=pd.read_csv(r'E:\nlp\NLP\SMSSpamCollection.txt',sep='\t',names=['label','message'])
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,len(data)):
    regex=re.sub("[^A-Za-z]"," ",data['message'][i])
    regex=regex.lower()
    regex=regex.split()
    regex=[ps.stem(word) for word in regex if word not in stopwords.words('english')]
    regex=' '.join(regex)
    corpus.append(regex)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
replace={'ham':0,'spam':1}
y=data['label'].map(replace)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,random_state=0)
model=Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
histroy_rnn=model.fit(X_train,Y_train,epochs=10,validation_data=(X_test,Y_test))
acc = histroy_rnn.history['acc']
val_acc = histroy_rnn.history['val_acc']
loss = histroy_rnn.history['loss']
val_loss = histroy_rnn.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, '-', color='orange', label='training acc')
plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='orange', label='training acc')
plt.plot(epochs, val_loss,  '-', color='blue', label='validation acc')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model.save('hamspammodelRNN.h5')

