from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import itertools
import datetime
import os
from zipfile import ZipFile
from os.path import exists

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, Merge, Bidirectional, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import dot, subtract, multiply, concatenate, add
import keras.backend as K
from keras.optimizers import Adadelta

NUM = 3
GoogleNews = False
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 30
MA_DISTANCE = False
n_MLP = 100
PATH = '/u/xh3426/cs388/nlp-project/'
TRAIN_CSV = PATH + 'data/train.csv'
TEST_CSV = PATH + 'data/test.csv'
if GoogleNews:
    EMBEDDING_FILE_PATH = PATH + 'data/GoogleNews-vectors-negative300.bin.gz'
else:
    EMBEDDING_FILE_PATH = '/scratch/cluster/xh3426/nlp/glove.840B.300d.txt'
    if not exists(EMBEDDING_FILE_PATH):
        zipfile = ZipFile('/scratch/cluster/xh3426/nlp/glove.840B.300d.zip')
        zipfile.extract("glove.840B.300d.txt", path='/scratch/cluster/xh3426/nlp/')


SAVEPATH = '/scratch/cluster/xh3426/nlp/MaLSTM' + str(NUM)
if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)
PNGSAVEPATH = PATH + 'MaLSTM' + str(NUM)
if not os.path.exists(PNGSAVEPATH):
    os.makedirs(PNGSAVEPATH)
MODEL_FILE = SAVEPATH + '/model.h5'
HISTORY_FILE = PNGSAVEPATH + '/trainHistory.p'
PRIDICT_FILE = SAVEPATH + '/predict.p'
LOG_FILE = PNGSAVEPATH + '/log.p'
ACC_PNG = PNGSAVEPATH + '/accuracy.png'
LOSS_PNG = PNGSAVEPATH + '/loss.png'


pickle.dump({
    'n_hidden': n_hidden,
    'gradient_clipping_norm': gradient_clipping_norm,
    'batch_size': batch_size,
    'n_epoch': n_epoch,
    'EMBEDDING': GoogleNews,
    'MA_DISTANCE': MA_DISTANCE,
    'n_MLP': n_MLP
}, open(LOG_FILE, "wb"))


# Load training and test set
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

stops = set(stopwords.words('english'))


def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


# Prepare embedding
questions_cols = ['question1', 'question2']
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding


# Iterate over the questions only of both training and test datasets
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():
        # Iterate through the text of both questions of the row
        for question in questions_cols:
            q2n = []  # q2n -> question numbers representation
            for word in text_to_word_list(row[question]):
                # Check for unwanted words
                if word in stops:
                    continue
                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    q2n.append(vocabulary[word])
            # Replace questions as word to question as number representation
            dataset.set_value(index, question, q2n)


embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored


if GoogleNews:
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE_PATH, binary=True)
    for word, index in vocabulary.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    del word2vec
else:
    embeddings_index = {}
    with open(EMBEDDING_FILE_PATH, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    for word, index in vocabulary.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings[index] = embedding_vector
    del embeddings_index


max_seq_length = max(train_df.question1.map(lambda x: len(x)).max(),
                     train_df.question2.map(lambda x: len(x)).max(),
                     test_df.question1.map(lambda x: len(x)).max(),
                     test_df.question2.map(lambda x: len(x)).max())

# Split to train validation
validation_size = 40000
training_size = len(train_df) - validation_size

X = train_df[questions_cols]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
X_test = {'left': test_df.question1, 'right': test_df.question2}


# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation, X_test], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)


# Model variables

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_sequences = shared_lstm(encoded_left)
right_sequences = shared_lstm(encoded_right)


# attention layer
left_output = left_sequences
right_output = right_sequences

# Calculates the distance as defined by the MaLSTM model
if MA_DISTANCE:
    output = Merge(mode=lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
else:
    addition = add([left_output, right_output])
    differences = subtract([left_output, right_output])
    square_diff = multiply([differences, differences])
    merged = concatenate([addition, square_diff])
    merged = Dense(n_MLP, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)
    output = Dense(1, activation='sigmoid')(merged)
    
# Pack it all up into a model
malstm = Model([left_input, right_input], [output])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])


# Start training
training_start_time = time()

malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation))

# malstm_predict = malstm.predict([X_test['left'], X_test['right']])
# pickle.dump(malstm_predict, open(PRIDICT_FILE, "wb"))
# del malstm_predict

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

malstm.save(MODEL_FILE)
del malstm

pickle.dump(malstm_trained.history, open(HISTORY_FILE, "wb"))
del malstm_trained

# Plot accuracy
history = pickle.load(open(HISTORY_FILE, "rb"))
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(ACC_PNG)
plt.clf()

# Plot loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(LOSS_PNG)
plt.clf()

# malstm = load_model(MODEL_FILE)
# malstm_predict = pickle.load(open(PRIDICT_FILE, "rb"))
# print(malstm_predict)
