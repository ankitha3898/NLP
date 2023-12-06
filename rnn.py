from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding,SimpleRNN
from keras.utils import pad_sequences
(X_train,Y_train),(X_test,Y_Test)=imdb.load_data()
X_train=pad_sequences(X_train,padding='post',maxlen=50)
X_test=pad_sequences(X_test,padding='post',maxlen=50)
model=Sequential()
model.add(SimpleRNN(32,input_shape=(50,1),return_sequences=False))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=10,validation_data=(X_test,Y_Test))
model.save('modelRNN.h5')