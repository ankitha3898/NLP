import numpy as np
list1=['go india',
       'hu how you',
       'india is my country']
from keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer(oov_token='<nothing>')
tokenizer.fit_on_texts(list1)
print(tokenizer.word_counts)
print(tokenizer.word_index)
sequences=tokenizer.texts_to_sequences(list1)
from keras.utils import pad_sequences
sequences=pad_sequences(sequences,padding='post')
print(sequences)