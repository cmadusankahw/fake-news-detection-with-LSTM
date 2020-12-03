import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')
# import keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model

# getting the accuracy
from sklearn.metrics import accuracy_score

# get the confusion matrix
from sklearn.metrics import confusion_matrix

# importing tokanized test/ train splits
from tokanizer import *

# TASK #9: BUILD AND TRAIN THE MODEL

# Sequential Model
model = Sequential()

# embedding layer
model.add(Embedding(total_words, output_dim=128))
# model.add(Embedding(total_words, output_dim = 240))

# Bi-Directional RNN and LSTM
model.add(Bidirectional(LSTM(128)))

# Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

print()

y_train = np.asarray(y_train)

# train the model
model.fit(padded_train, y_train, batch_size=64, validation_split=0.1, epochs=2)


# TASK #9: ASSESS TRAINED MODEL PERFORMANCE


# make prediction
pred = model.predict(padded_test)

# if the predicted value is >0.5 it is real else it is fake
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

accuracy = accuracy_score(list(y_test), prediction)

print("Model Accuracy : ", accuracy)

cm = confusion_matrix(list(y_test), prediction)
plt.figure(figsize=(25, 25))
sns.heatmap(cm, annot=True)

# category dict
category = {0: 'Fake News', 1: "Real News"}