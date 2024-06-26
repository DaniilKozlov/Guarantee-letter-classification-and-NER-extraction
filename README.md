# Guarantee-letter-classification-and-NER-extraction

##  Введение
Добрый день! 

Здесь я публикую часть моей работы над классификацией гарантийных писем и выделением из них названий услуг. В работе представлен ход моих мыслей. 
К сожалению, из-за политики о неразглашении я не имею права публиковать ее полностью. В публикуемых примерах данных я заменил кодировки медицинских услуг и некоторые наименования медицинских услуг для избежания утечки данных. 
Надеюсь, с такими ограничениями у меня удалось показать замысел и аспекты реализации проекта.

## Инструменты
* Spacy
* Label-Studio

## Цель
Необходимо разработать алгоритм, который будет способен автоматически читать и обрабатывать гарантийные письма, получаемые от страховой компании больницей. Гарантийные письма содержат большое количество текстовой информации о предоставляемых услугах.

![image](https://github.com/DaniilKozlov/Guarantee-letter-classification-and-NER-extraction/assets/120337443/61e6a362-b8a3-4948-8477-372433948ce7)

Целесообразность выделения услуг определяется наличием этих услуг в письме. 

## Данные

В тренировочных данных находятся 162 гарантийных письма, где класс pos представлен 73%.

![image](https://github.com/DaniilKozlov/Guarantee-letter-classification-and-NER-extraction/assets/120337443/93a2eecd-828b-43b4-8e97-3d9da8535568)

Гарантийные письма содержат наименования медицинских услуг, записанные как шаблоном, так и в свободной форме. В некоторых письмах указываются также, и кодировки услуг согласно справочнику
Зеленым в данном случае выделены наименования, а желтым - кодировки.

![image](https://github.com/DaniilKozlov/Guarantee-letter-classification-and-NER-extraction/assets/120337443/e7594ef1-50dd-4ae6-82eb-a0c0ba58660e)

## Подход

Было решено поменять этапы классификации и выделения услуг местами, а классификацию провести пост-фактум по наличию услуг в получаемом массиве. Если массив оказывается пустым - это class neg. 

1) Первым этапом было выделение кодировок. Так как все они были написаны по одному правилу, то здесь подошло выделение с помощью регулярных выражений.

`import re`

`pattern = r'\b\w{1}\d{2}\.\d{2}\.\d{2}\.\d{1}\.\d{3}\b'`

`lst_of_codes = [re.findall(pattern, i) for i in data_train['guarantee_letter_text']]`

2) Предобработка. Удаление символов \n внутри каждого гарантийного письма

   Формирование txt-файла с текстом, где тексты писем разделяются знаком переноса

3) Загрузка текста в Label-studio. Разметка спанов именованных сущностей.

    `!label-studio start`

   Интерфейс работы в Label Studio продемонстрирован на примере вырезки из статьи про GitHub в Википедии, поскольку я не имею права публиковать предоставленные мне текстовые данные.
   ![image](https://github.com/DaniilKozlov/Guarantee-letter-classification-and-NER-extraction/assets/120337443/cdf21507-d5e8-4c88-b1bf-4f70130bd881)

  В нашем примере я выбрал выделение двух видов сущностей - кодировок услуг и наименований услуг.
  
  После разметки информация о сущностях выгружается в json формате.

  Чтение json

  ```
with open('ner.json', 'r') as f:
    ner = json.load(f)

training_data = []
for a in ner:
    temp_dict={}
    temp_dict['text'] = a['data']['text']
    temp_dict['entities'] = []
    for b in a['annotations'][0]['result']:
        start = b['value']['start']
        end = b['value']['end']
        label = b['value']['labels']
        for c in b['value']['labels']:
            temp_dict['entities'].append((
            start, end, c.upper()
        ))
    training_data.append(temp_dict)

print(training_data)
```
  
4) Инициализация модели spacy и подготовка к обучению.

Инициализация

```
nlp = spacy.blank('ru') 

doc_bin = DocBin()
```
Загрузка тренировочной информации в doc_bin - класс, в котором сериализована вся информация для обучения.
```
from spacy.util import filter_spans

for training_example  in training_data: 
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text) 
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents 
    doc_bin.add(doc)

doc_bin.to_disk("train.spacy") 
```

Следующим этапом идет подготовка config-файла по этой ссылке: https://spacy.io/usage/training#quickstart

Инициализация настроек

`!python -m spacy init fill-config base_config.cfg config.cfg`

Запуск обучения

`!python -m spacy train config.cfg --output ./ --paths.train ./train.spacy --paths.dev ./train.spacy `


Обучение

Заняло 31 минуту

![image](https://github.com/DaniilKozlov/Guarantee-letter-classification-and-NER-extraction/assets/120337443/f7d979c4-2eab-45c9-bf35-d43c320994c6)


## Результаты

![image](https://github.com/DaniilKozlov/Guarantee-letter-classification-and-NER-extraction/assets/120337443/f99c658a-683e-4fc4-8c24-8931cb1263c8)

В переменную class_list попали предсказания класса на основе наличия выделенных сущностей. 1 - есть сущности, класс pos; 0 - нет сущностей, класс neg.

Тестирование проводилось на другом датасете

![image](https://github.com/DaniilKozlov/Guarantee-letter-classification-and-NER-extraction/assets/120337443/1de4e41b-bac5-4931-96d7-733b633ce0a9)

Выделенные сущности

![image](https://github.com/DaniilKozlov/Guarantee-letter-classification-and-NER-extraction/assets/120337443/76e457e6-caa2-4ae2-8b09-7a7396ecc1ac)

Спаны кодировок я также решил включить в разметку. Результат выделения именно кодировок - accuracy 1.0.
Поэтому практическая разница между подходом с регулярными выражениями и выделением нейросетевой моделью отсутствует. Однако подход поиска на основе заданного паттерна логически обоснован лучше - формат кодировок не меняется, а остается одним и тем же от услуги к услуге.


## Вывод

Мне видятся неплохими полученные метрики классификации с учетом небольшого количества обучающих примеров. Однако выделение самих сущностей требует доработки.
Возможным вариантом решения проблемы является сравнение схожести полученной выделенной сущности с формулировкой из справочника услуг.

## Источники информации

Spacy: https://spacy.io/usage/training#quickstart
Хабр: https://habr.com/ru/articles/531940/


