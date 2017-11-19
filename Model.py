from keras.layers import Dense, LSTM, TimeDistributed, Dropout
from keras.layers.core import Activation
from keras.models import Sequential
import numpy as np


def build_model(vocab_size):

    model = Sequential()

    model.add(LSTM(150, return_sequences=True, input_shape=(None, vocab_size)))
    model.add(Dropout(.3))

    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(.3))

    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(.3))

    model.add(TimeDistributed(Dense(vocab_size)))

    model.add(Activation("softmax"))

    model.compile(

        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=["categorical_crossentropy"]
    )

    return model


def text_generator(text_size=1000):

    rand_index = np.random.randint(vocab_size)

    text_as_num = []
    text_as_char = []

    text_as_char.append(index_to_char[rand_index])
    text_as_num.append(rand_index)

    for i in range(text_size):

        last_char = np.zeros(vocab_size)

        last_char[text_as_num[len(text_as_num) - 1]] = 1

        last_char.reshape(1, 1, vocab_size)

        prediction = np.argmax(model.predict(last_char))

        prediction = prediction[0]

        text_as_num.append(prediction)
        text_as_char.append(index_to_char[prediction])


    final_output = ''

    for i in range(len(text_as_char)):
        final_output += text_as_char[i]

    print("\n\n%s\n" % final_output)


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

epoch = 0

while True:

    epoch += 1

    print("\n\nStarting Epoch %d\n\n" % epoch)

    model.fit(x, y, epochs=1, batch_size=32)

    if epoch % 1 == 0:
        model.save_weights('Epoch{}.hdf5'.format(epoch))
        text_generator()
