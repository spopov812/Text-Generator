from keras.layers import Dense, LSTM, TimeDistributed, Dropout
from keras.layers.core import Activation
from keras.models import Sequential
import numpy as np


def build_model(vocab_size):

    model = Sequential()

    model.add(LSTM(100, return_sequences=True, input_shape=(None, vocab_size)))
    model.add(Dropout(.3))

    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(.3))

    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(.3))

    model.add(TimeDistributed(Dense(vocab_size)))

    model.add(Activation("softmax"))

    model.compile(

        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=["categorical_crossentropy"]
    )

    return model

text = open("warpeace_input.txt", 'r').read()

chars = list(set(text))

vocab_size = len(chars)

# print("\n\n There are %d unique chars in the dataset.\n\n" % vocab_size)
# print(chars)

index_to_char = {index:char for index, char in enumerate(chars)}
char_to_index = {char:index for index, char in enumerate(chars)}



x = []
y = []

for i in range(len(text) - 1):

    one_hot = np.zeros(vocab_size)
    one_hot[char_to_index[text[i]]] = 1

    x.append(one_hot)

    one_hot = np.zeros(vocab_size)
    one_hot[char_to_index[text[i + 1]]] = 1

    y.append(one_hot)

    if i % 1000000 == 0:
        print(i)

x = np.array(x)
y = np.array(y)

x = x.reshape(len(text) - 1, 1, vocab_size)
y = y.reshape(len(text) - 1, 1, vocab_size)

# print(x.shape)

model = build_model(vocab_size)

model.fit(x, y, epochs=1, batch_size=32)
