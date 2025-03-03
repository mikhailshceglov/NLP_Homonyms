import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import fasttext
import pickle
import xml.etree.ElementTree as ET
import re

# Загрузка и обработка данных из файла processed_corpus.txt
file_path = "../processed_corpus.txt"
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        tokens = line.strip().split()
        sentence = []
        for token in tokens:
            if '{' in token and '=' in token:
                try:
                    word, lemma_tag = token.split('{', 1)
                    lemma, tag = lemma_tag.rsplit('=', 1)
                    tag = tag.strip('}')
                    sentence.append((word, lemma, tag))
                except ValueError:
                    continue
        if sentence:
            data.append(sentence)

# Формирование списков слов и тегов
words = [word for sentence in data for word, _, _ in sentence]
tags = [tag for sentence in data for _, _, tag in sentence]

# Кодирование тегов и слов
tag_encoder = LabelEncoder()
word_encoder = LabelEncoder()
tag_encoder.fit(tags)
word_encoder.fit(words)

# Преобразуем списки в массивы и кодируем
all_words = np.array(words)
all_tags = np.array(tags)
encoded_words = word_encoder.transform(all_words)
encoded_tags = tag_encoder.transform(all_tags)

# Сегментация обратно на предложения
X, y = [], []
word_idx = 0
tag_idx = 0
for sentence in data:
    length = len(sentence)
    X.append(encoded_words[word_idx:word_idx + length])
    y.append(encoded_tags[tag_idx:tag_idx + length])
    word_idx += length
    tag_idx += length

# Определение максимальной длины последовательности и паддинг
max_seq_len = max(len(sentence) for sentence in data)
X = pad_sequences(X, maxlen=max_seq_len, padding='post')
y = pad_sequences(y, maxlen=max_seq_len, padding='post')
num_tags = len(tag_encoder.classes_)
y = to_categorical(y, num_classes=num_tags)

# Разбиение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение fastText-модели на текстовом корпусе для эмбеддингов
ft_model = fasttext.train_unsupervised(file_path, model='skipgram', dim=128)
vocab_size = len(word_encoder.classes_)
embedding_dim = 128

# Создание эмбеддинг-матрицы с использованием fastText
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in zip(word_encoder.classes_, range(vocab_size)):
    embedding_matrix[idx] = ft_model.get_word_vector(word)

# Построение модели
model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_seq_len,
        trainable=True
    ),
    Bidirectional(LSTM(units=32, return_sequences=True)),
    TimeDistributed(Dense(num_tags, activation='softmax'))
])

model.build(input_shape=(None, max_seq_len))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=3,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

model.save('pos_model.h5')
with open('word_encoder.pkl', 'wb') as f:
    pickle.dump(word_encoder, f)
with open('tag_encoder.pkl', 'wb') as f:
    pickle.dump(tag_encoder, f)
with open('max_seq_len.pkl', 'wb') as f:
    pickle.dump(max_seq_len, f)

ft_model.save_model('ft_model.bin')
