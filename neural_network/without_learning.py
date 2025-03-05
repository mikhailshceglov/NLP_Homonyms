import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
import re
import xml.etree.ElementTree as ET
from rapidfuzz import process
import warnings
warnings.simplefilter("ignore")

model = tf.keras.models.load_model('pos_model.h5')
with open('word_encoder.pkl', 'rb') as f:
    word_encoder = pickle.load(f)
with open('tag_encoder.pkl', 'rb') as f:
    tag_encoder = pickle.load(f)
with open('max_seq_len.pkl', 'rb') as f:
    max_seq_len = pickle.load(f)

class Lemmatizer:
    def __init__(self, dictionary_path, score_cutoff=90):
        self.lemmas = {}
        self.score_cutoff = score_cutoff
        self.load_dictionary(dictionary_path)

    def load_dictionary(self, dictionary_path):
        tree = ET.parse(dictionary_path)
        root = tree.getroot()
        for lemma in root.findall("lemmata/lemma"):
            # Берем базовую (словарную) форму из элемента <l>
            lemma_text = lemma.find("l").get("t")
            for form in lemma.findall("f"):
                form_text = form.get("t").lower()
                self.lemmas[form_text] = lemma_text

    def find_best_match(self, word):
        # Используем RapidFuzz для нечеткого поиска
        best = process.extractOne(word, self.lemmas.keys(), score_cutoff=self.score_cutoff)
        if best:
            candidate = best[0]
            # Если кандидат слишком короткий, то считаем, что подходящего совпадения не найдено (Эвристика!)
            if len(candidate) < 0.6 * len(word):
                return None
            return candidate
        return None

    def lemmatize(self, text):
        # Токенизация с сохранением знаков препинания
        tokens = re.findall(r"\w+|[.,!?;]", text)
        lemmatized_tokens = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token in self.lemmas:
                lemma = self.lemmas[lower_token]
            else:
                candidate = self.find_best_match(lower_token)
                if candidate:
                    lemma = self.lemmas[candidate]
                else:
                    # Эвристика для глаголов: если слово заканчивается на "ет",
                    # попробуем заменить окончание на "ь" (например, "печет" -> "печь")
                    if lower_token.endswith("ет"):
                        new_word = lower_token[:-2] + "ь"
                        if new_word in self.lemmas:
                            lemma = self.lemmas[new_word]
                        else:
                            lemma = lower_token
                    else:
                        lemma = lower_token 
            lemmatized_tokens.append(f"{token}{{{lemma}=}}")
        return " ".join(lemmatized_tokens)


opencorpora_dict_path = "../dict.opcorpora.xml"
lemmatizer = Lemmatizer(opencorpora_dict_path, score_cutoff=90)

# Ввод текста
input_text = "я люблю русскую землю. Ежи гуляют по лесу! Очень быстро течет ручей, а бабушка печет пирог. Привет! Приветствую, друзья. Привет, макет, пакет"

lemmatized_output = lemmatizer.lemmatize(input_text)
print("Лемматизированное предложение:")
print(lemmatized_output)

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

words_only = [w for (w, _) in data[0]]
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

# Получение предсказаний модели (предсказание частей речи)
predictions = model.predict(X)
pred_indices = np.argmax(predictions[0], axis=-1)
pred_tags = tag_encoder.inverse_transform(pred_indices)

# Формирование итоговой строки вида "слово{лемма=тег}"
result_tokens = []
for (word, lemma), tag in zip(data[0], pred_tags):
    result_tokens.append(f"{word}{{{lemma}={tag}}}")
result_line = " ".join(result_tokens)
print("Результат:")
print(result_line)
