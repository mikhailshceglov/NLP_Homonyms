import re
import xml.etree.ElementTree as ET
from rapidfuzz import process

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
            return best[0]
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
                    lemma = lower_token
            lemmatized_tokens.append(f"{token}{{{lemma}=}}")
        return " ".join(lemmatized_tokens)

# Пример использования
opencorpora_dict_path = "dict.opcorpora.xml"
lemmatizer = Lemmatizer(opencorpora_dict_path, score_cutoff=99)

text = "Пример текста, где может быть ошибка в форме слова, например, ежем вместо ежа. Попугай Кеша. Попугал кота"
print(lemmatizer.lemmatize(text))
