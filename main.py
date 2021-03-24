import nltk
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split

with open('content/BIG_BOT_CONFIG.json', 'r') as f:
    BOT_CONFIG = json.load(f) # читаем json в переменную BOT_CONFIG

def cleaner(text): # функция очистки текста
    cleaned_text = ''
    for ch in text.lower():
        if ch in ('абвгдеёжзийклмнопрстуфхцчшщъыьэюяqwertyuiopasdfghjklzxcvbnm'):
            cleaned_text = cleaned_text + ch
    return cleaned_text

def match(text, example): # гибкая функция сравнения текстов
    return nltk.edit_distance(text, example) / len(example) < 0.4

def get_intent(text): # функция определения интента текста
    for intent in BOT_CONFIG['intents']:
        for example in BOT_CONFIG['intents'][intent]['examples']:
             if match(cleaner(text), cleaner(example)):
                  return intent

X = []
y = []

for intent in BOT_CONFIG['intents']:
     if 'examples' in BOT_CONFIG['intents'][intent]:
          X += BOT_CONFIG['intents'][intent]['examples']
          y += [intent for i in range(len(BOT_CONFIG['intents'][intent]['examples']))]

# Создаем обучающую выборку для ML-модели
vectorizer = CountVectorizer(preprocessor=cleaner, ngram_range=(1,3), stop_words=['а', 'и'])
# Создаем векторайзер – объект для превращения текста в вектора
vectorizer.fit(X)
X_vect = vectorizer.transform(X)
# Обучаем векторайзер на нашей выборке
X_train_vect, X_test_vect, y_train, y_test = train_test_split(X_vect, y, test_size=0.3)
# Разбиваем выборку на train и на test
sgd = SGDClassifier() # Создаем модель
# sgd.fit(X_train_vect, y_train) # Обучаем модель
# sgd.score(X_test_vect, y_test) # Проверяем качество модели на тестовой выборке

sgd.fit(X_vect, y)

sgd.score(X_vect, y) # Смотрим качество классификации

def get_intent_by_model(text):  # Функция определяющая интент текста с помощью ML-модели
    return sgd.predict(vectorizer.transform([text]))[0]


def bot(text):  # функция бота
    intent = get_intent(text)  # 1. попытаться понять намерение сравнением по Левинштейну

    if intent is None:
        intent = get_intent_by_model(text)  # 2. попытаться понять намерение с помощью ML-модели

    return random.choice(BOT_CONFIG['intents'][intent]['responses'])

question = ''
while question not in ['выход', 'выключайся']:
    question = input()
    answer = bot(question)
    print(answer)