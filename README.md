# NLP Homonyms

## Краткое описание
Проект решает задачу автоматического определения части речи (**POS-tagging**) и снятия омонимии в русском языке. В нём сочетаются два подхода:
- **HMM-теггер** (Hidden Markov Model)
- **Нейронная сеть** (LSTM + fastText эмбеддинги)

---

## Установка и настройка

### 1️. Клонирование репозитория
```bash
git clone https://github.com/mikhailshceglov/NLP_Homonyms.git
cd NLP_Homonyms
```

### 2️. Создание виртуального окружения
```bash
python3 -m venv venv
source venv/bin/activate  # Для Linux/Mac
# Для Windows:
# venv\Scripts\activate
```

### 3️. Установка зависимостей
```bash
pip install -r requirements.txt
```
Список основных библиотек (указан в `requirements.txt`):
- `tensorflow==2.10.0`
- `numpy==1.21.6`
- `scikit-learn==1.0.2`
- `keras==2.6.0`
- `fasttext==0.9.2`
- `nltk==3.7`
- `rapidfuzz==2.13.7`

---

## Структура проекта

```text
NLP_Homonyms/
│
├─ hmm_method/
│   ├─ train_hmm.py        # Обучение HMM-модели
│   ├─ use_hmm.py          # Использование обученной HMM-модели
│   └─ hmm_tagger.pkl      # Сериализованная модель HMM
│
├─ neural_network/
│   ├─ learning_saving.py  # Обучение нейронной сети (LSTM + fastText)
│   └─ without_learning.py # Использование обученной нейронной сети
│
├─ make_processed_txt.py    # Преобразование XML-файла OpenCorpora в processed_corpus.txt
├─ processed_corpus.txt     # Пример корпуса (результат make_processed_txt.py)
├─ opcorpora.zip            # Архив с данными OpenCorpora
├─ requirements.txt         # Список необходимых библиотек
└─ README.md                # Текущее описание проекта
```

---

## Использование

### 1. Подготовка данных
Если нужно сгенерировать корпус из файла OpenCorpora (например, `annot.opcorpora.xml`), запустите:
```bash
python3 make_processed_txt.py
```
Скрипт создаст/обновит файл `processed_corpus.txt`.

### 2. Обучение HMM-теггера
```bash
cd hmm_method
python3 train_hmm.py
```
Скрипт обучит HMM-модель и сохранит её в файл `hmm_tagger.pkl`.

### 3. Использование HMM-теггера
```bash
python3 use_hmm.py
```
Скрипт загрузит модель `hmm_tagger.pkl` и словарь OpenCorpora (`dict.opcorpora.xml`), затем выведет разметку и лемматизацию текста.

### 4. Обучение нейронной сети
```bash
cd neural_network
python3 learning_saving.py
```
Скрипт использует файл `processed_corpus.txt` для обучения LSTM-модели с fastText-эмбеддингами.  
Результат: файлы `pos_model.h5`, `word_encoder.pkl`, `tag_encoder.pkl` и `max_seq_len.pkl` будут сохранены в папке.

### 5. Использование нейронной сети без обучения
```bash
python3 without_learning.py
```
Скрипт загрузит готовую модель (`pos_model.h5` и другие файлы) и выведет результат POS-теггинга и лемматизации для примерного текста.

