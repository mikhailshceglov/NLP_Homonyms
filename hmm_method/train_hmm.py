import re
import random
import xml.etree.ElementTree as ET
import nltk
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.probability import LidstoneProbDist
import pickle

def parse_token(token):
    if '{' in token and '}' in token:
        word = token.split('{')[0]
        inner = token.split('{')[1].split('}')[0]
        try:
            lemma, tag = inner.split('=')
            return word, tag
        except ValueError:
            return word, None
    return token, None

def load_training_data(filename):
    training_data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            sent = []
            for token in tokens:
                word, tag = parse_token(token)
                if tag is not None:
                    sent.append((word, tag))
            if sent:
                training_data.append(sent)
    return training_data

# Определяем функцию оценщика на верхнем уровне, чтобы pickle мог её сериализовать
def lidstone_estimator(fd, bins):
    return LidstoneProbDist(fd, 0.1, bins)

def train_hmm_tagger(train_data):
    hmm_tagger = HiddenMarkovModelTagger.train(train_data, estimator=lidstone_estimator)
    return hmm_tagger

def main():
    training_file = "../processed_corpus.txt" 
    all_data = load_training_data(training_file)
    if not all_data:
        print("Обучающие данные не найдены или файл пуст.")
        return

    # Разбиваем данные на обучающую и тестовую выборки 80/20
    random.shuffle(all_data)
    split_index = int(0.8 * len(all_data))
    train_data = all_data[:split_index]
    test_data = all_data[split_index:]
    
    # Обучаем HMM-теггер на обучающих данных
    hmm_tagger = train_hmm_tagger(train_data)
    
    # Оцениваем точность модели на тестовой выборке
    accuracy = hmm_tagger.evaluate(test_data)
    print(f"Accuracy on test set: {accuracy:.2%}")

    # Формируем списки истинных и предсказанных тегов для расчёта метрик
    true_tags = []
    pred_tags = []
    for sent in test_data:
        words = [w for (w, t) in sent]
        gold = [t for (w, t) in sent]
        predicted = [tag for (w, tag) in hmm_tagger.tag(words)]
        true_tags.extend(gold)
        pred_tags.extend(predicted)

    try:
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(true_tags, pred_tags))
    except ImportError:
        print("\nscikit-learn не установлен, classification report не может быть выведен.")

    with open("hmm_tagger.pkl", "wb") as f:
        pickle.dump(hmm_tagger, f)
    print("Модель сохранена в файл hmm_tagger.pkl")

if __name__ == '__main__':
    main()
