import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import fasttext  # fastText для получения эмбеддингов
import pickle
from keras.models import load_model, save_model
import xml.etree.ElementTree as ET
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import fasttext
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Загрузка и обработка данных из processed_corpus.txt
file_path = 'processed_corpus.txt'
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        tokens = line.strip().split()
        sentence = []
        for token in tokens:
            try:
                if '{' in token and '=' in token:
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
tag_encoder = LabelEncoder()
word_encoder = LabelEncoder()
tag_encoder.fit(tags)
word_encoder.fit(words)

# Преобразуем списки в массивы
all_words = np.array(words)
all_tags = np.array(tags)

encoded_words = word_encoder.transform(all_words)
encoded_tags = tag_encoder.transform(all_tags)

# Сегментация обратно на предложения
X = []
y = []
word_idx = 0
tag_idx = 0
for sentence in data:
    length = len(sentence)
    X.append(encoded_words[word_idx:word_idx + length])
    y.append(encoded_tags[tag_idx:tag_idx + length])
    word_idx += length
    tag_idx += length

# Выравнивание длины последовательностей
max_seq_len = max(len(sentence) for sentence in data)
X = pad_sequences(X, maxlen=max_seq_len, padding='post')
y = pad_sequences(y, maxlen=max_seq_len, padding='post')
num_tags = len(tag_encoder.classes_)
y = to_categorical(y, num_classes=num_tags)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Использование fastText для получения эмбеддингов, обучаем fastText-модель на файле processed_corpus.txt
# Параметр dim задаёт размерность эмбеддингов (можно сделать равной 128, чтобы соответствовать модели)
ft_model = fasttext.train_unsupervised(file_path, model='skipgram', dim=128)

vocab_size = len(word_encoder.classes_)
embedding_dim = 128

# Создаем эмбеддинг-матрицу для всех уникальных слов, используя fastText
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in zip(word_encoder.classes_, range(vocab_size)):
    embedding_matrix[idx] = ft_model.get_word_vector(word)

# Построение модели Keras с использованием fastText эмбеддингов
model = Sequential([
    # Здесь вместо случайной инициализации передаём подготовленную матрицу
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
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=3,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Сохранил модель для удобства
model.save('pos_model_big_new.h5')

y_pred = model.predict(X_test)  
y_pred_indices = np.argmax(y_pred, axis=-1)  
y_true_indices = np.argmax(y_test, axis=-1)  

def decode_tags(indices_2d):
    all_tags = []
    for seq in indices_2d:
        tags_in_seq = tag_encoder.inverse_transform(seq)
        all_tags.append(list(tags_in_seq))
    return all_tags

pred_tags_2d = decode_tags(y_pred_indices)   
true_tags_2d = decode_tags(y_true_indices)

pred_tags_flat = []
true_tags_flat = []

for p_seq, t_seq in zip(pred_tags_2d, true_tags_2d):
    pred_tags_flat.extend(p_seq)
    true_tags_flat.extend(t_seq)

print()
print(classification_report(true_tags_flat, pred_tags_flat, digits=4))

# ---------------------
# Лемматизация с использованием XML-словаря
# ---------------------
class Lemmatizer:
    def __init__(self, dictionary_path):
        self.lemmas = {}
        self.load_dictionary(dictionary_path)

    def load_dictionary(self, dictionary_path):
        tree = ET.parse(dictionary_path)
        root = tree.getroot()
        for lemma in root.findall("lemmata/lemma"):
            lemma_text = lemma.find("l").get("t")
            for form in lemma.findall("f"):
                form_text = form.get("t").lower()
                self.lemmas[form_text] = lemma_text

    def lemmatize(self, text):
        # Токенизация с сохранением знаков препинания
        tokens = re.findall(r"\w+|[.,!?;]", text)
        lemmatized_tokens = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token in self.lemmas:
                lemma = self.lemmas[lower_token]
                lemmatized_tokens.append(f"{token}{{{lemma}=}}")
            else:
                lemmatized_tokens.append(f"{token}{{{token}=}}")
        return " ".join(lemmatized_tokens)

opencorpora_dict_path = "dict.opcorpora.xml"
lemmatizer = Lemmatizer(opencorpora_dict_path)

# ---------------------
# Заданное предложение и подготовка данных для модели
# ---------------------

input_text = "Тут можно ввести любой свой текст! Сеть угадывает даже сленг. Кринж, скуф"

lemmatized_output = lemmatizer.lemmatize(input_text)
print("Лемматизированное предложение:")
print(lemmatized_output)

# Разбираем лемматизированный текст и получаем пары (word, lemma)
data = []
tokens = lemmatized_output.strip().split()
sentence = []
for token in tokens:
    if '{' in token and '=' in token:
        word, lemma_tag = token.split('{', 1)
        lemma, _ = lemma_tag.rsplit('=', 1)
        lemma = lemma.strip('}')
        sentence.append((word, lemma))
data.append(sentence)
print("Распарсили (word, lemma):", data[0])

# ---------------------
# Преобразование слов в индексы с использованием word_encoder
# ---------------------
words_only = [w for (w, _) in data[0]]
fallback_index = word_encoder.transform([word_encoder.classes_[0]])[0]
encoded_words = []
for word in words_only:
    try:
        idx = word_encoder.transform([word.lower()])[0]
    except Exception:
        idx = fallback_index
    encoded_words.append(idx)

# Паддинг до фиксированной длины (например, 427)
X = pad_sequences([encoded_words], maxlen=427, padding='post')

# ---------------------
# Предсказание частей речи моделью
# ---------------------

loaded_model = tf.keras.models.load_model('pos_model_big_new.h5')

predictions = loaded_model.predict(X)    # shape: (1, seq_len, num_tags)
predictions_for_seq = predictions[0]     # shape: (seq_len, num_tags)
pred_indices = np.argmax(predictions_for_seq, axis=-1)
pred_tags = tag_encoder.inverse_transform(pred_indices)

# ---------------------
# Формирование итоговой строки
# ---------------------
result_tokens = []
for (word, lemma), tag in zip(data[0], pred_tags):
    result_tokens.append(f"{word}{{{lemma}={tag}}}")

result_line = " ".join(result_tokens)
print("Результат:")
print(result_line)
