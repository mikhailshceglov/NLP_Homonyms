import re
import pickle
import xml.etree.ElementTree as ET
from rapidfuzz import process
from nltk.probability import LidstoneProbDist

# Функция-оценщик, необходимая для сериализации HMM-модели
def lidstone_estimator(fd, bins):
    return LidstoneProbDist(fd, 0.1, bins)

class Lemmatizer:
    def __init__(self, dictionary_path, score_cutoff=80):
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
        best = process.extractOne(word, self.lemmas.keys(), score_cutoff=self.score_cutoff)
        if best:
            return best[0]
        return None

    def lemmatize_token(self, token):
        lower_token = token.lower()
        if lower_token in self.lemmas:
            return self.lemmas[lower_token]
        else:
            candidate = self.find_best_match(lower_token)
            if candidate:
                return self.lemmas[candidate]
            else:
                return lower_token

    def lemmatize(self, text):
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
                    lemma = lower_token
            lemmatized_tokens.append(f"{token}{{{lemma}=}}")
        return " ".join(lemmatized_tokens)

# Функция для обработки одного предложения:
# 1. Токенизация (оставляем только слова)
# 2. Определение частей речи с помощью загруженной HMM-модели
# 3. Лемматизация токенов с использованием словаря OpenCorpora
# 4. Формирование строки вида: "токен{лемма=POS}"
def process_sentence(sentence, lemmatizer, hmm_tagger):
    tokens = re.findall(r'\w+', sentence)
    tagged = hmm_tagger.tag(tokens)
    output_tokens = []
    for token, pos in tagged:
        lemma = lemmatizer.lemmatize_token(token)
        output_tokens.append(f"{token}{{{lemma}={pos}}}")
    return " ".join(output_tokens)

def main():
    dictionary_path = "dict.opcorpora.xml"
    model_path = "hmm_tagger.pkl"
    
    with open(model_path, "rb") as f:
        hmm_tagger = pickle.load(f)
    print("Модель загружена из файла", model_path)
    
    # Создаем экземпляр лемматизатора
    lemmatizer = Lemmatizer(dictionary_path)
    
    # Пример входного текста
    input_text = "Русская печь. Печь пироги. Еж гуляет по лесу, а другие ежи купаются в ручье. Печем пироги!"

    sentences = [s.strip() for s in re.split(r'\.+', input_text) if s.strip()]
    print("\nПример разметки предложений:")
    for sentence in sentences:
        processed = process_sentence(sentence, lemmatizer, hmm_tagger)
        print(processed)

if __name__ == '__main__':
    main()
