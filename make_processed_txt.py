import xml.etree.ElementTree as ET

input_file = "annot.opcorpora.xml"

output_file = "processed_corpus.txt"

def parse_opencorpora(input_path, output_path):
    tree = ET.parse(input_path)
    root = tree.getroot()

    with open(output_path, "w", encoding="utf-8") as f_out:
        for sentence in root.findall(".//sentence"):
            tokens = [] 
            # Проходимся по всем токенам в предложении
            for token in sentence.findall(".//token"):
                token_text = token.get("text") 

                # Извлекаем лемму и её часть речи
                lemma = None
                pos = None
                for l in token.findall(".//l"):
                    lemma = l.get("t")  # Лемма
                    for g in l.findall(".//g"):
                        if g.get("v").isalpha():  # Определяем часть речи
                            pos = g.get("v")
                            break
                    if lemma and pos:  # Если нашли лемму и POS, выходим
                        break

                if lemma and pos:  # Собираем формат токен{лемма=POS}
                    tokens.append(f"{token_text}{{{lemma}={pos}}}")

            if tokens:
                f_out.write(" ".join(tokens) + "\n")


parse_opencorpora(input_file, output_file)
print(f"Обработка завершена. Результаты сохранены в {output_file}")

