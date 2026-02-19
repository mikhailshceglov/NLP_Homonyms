1) venv
```
  pip install transformers accelerate seqeval tqdm numpy
```

2) Запуск
2.1 Подготовка датасетов
```
  python src/prepare_data.py
```
2.2 Обучение POS-теггера (трансформер, GPU)
```
  python src/train_pos_transformer.py --epochs 3 --batch 8
```
2.3 Проверка POS на тексте
```
  python src/predict_pos.py --text "Бабушка печет пирожки. Очень горячая печь!"
```
2.4 Обучение LSTM-лемматизатора (GPU)
```
  python src/train_lemma_lstm.py --epochs 8 --batch 256
```
2.5 Проверка лемматизации на тексте
```
  python src/lemmatize_lstm.py --text "Бабушка печет пирожки. Очень горячая печь!"
```
2.6 Полный пайплайн: POS + lemma в формат слово{лемма=POS}
```
  python src/pipeline.py --text "Бабушка печет пирожки. Очень горячая печь!"
```
