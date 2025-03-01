import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
import xml.etree.ElementTree as ET
import re

# Загрузка обученной модели и сохранённых объектов
model = tf.keras.models.load_model('pos_model.h5')
with open('word_encoder.pkl', 'rb') as f:
    word_encoder = pickle.load(f)
with open('tag_encoder.pkl', 'rb') as f:
    tag_encoder = pickle.load(f)
with open('max_seq_len.pkl', 'rb') as f:
    max_seq_len = pickle.load(f)

# Класс лемматизатора (использует словарь из dict.opcorpora.xml)
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

# Инициализация лемматизатора (файл dict.opcorpora.xml должен быть доступен)
lemmatizer = Lemmatizer("dict.opcorpora.xml")

# Обработка входного текста
input_text = "Тут можно ввести любой свой текст! Сеть угадывает даже сленг. Кринж, скуф"
input_text = "Царица. Лечь на печь. я хочу печь пироги"
input_text = "Отче наш, в 20:00 12.12.2012 Иже еси на небесех, да святится Имя твоё"
input_text = "я люблю русскую землю"

lemmatized_output = lemmatizer.lemmatize(input_text)
print("Лемматизированное предложение:")
print(lemmatized_output)

# Разбор лемматизированного текста для получения пар (слово, лемма)
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

# Преобразование слов в индексы с использованием сохранённого word_encoder
words_only = [w for (w, _) in data[0]]
# Если слово не найдено в словаре, используем индекс первого слова в словаре как запасной вариант
fallback_index = word_encoder.transform([word_encoder.classes_[0]])[0]
encoded_words = []
for word in words_only:
    try:
        idx = word_encoder.transform([word.lower()])[0]
    except Exception:
        idx = fallback_index
    encoded_words.append(idx)

# Паддинг последовательности до длины max_seq_len
X = pad_sequences([encoded_words], maxlen=max_seq_len, padding='post')

# Получение предсказаний модели
predictions = model.predict(X)  # shape: (1, seq_len, num_tags)
pred_indices = np.argmax(predictions[0], axis=-1)
pred_tags = tag_encoder.inverse_transform(pred_indices)

# Формирование итоговой строки вида "слово{лемма=тег}"
result_tokens = []
for (word, lemma), tag in zip(data[0], pred_tags):
    result_tokens.append(f"{word}{{{lemma}={tag}}}")
result_line = " ".join(result_tokens)
print("Результат:")
print(result_line)
