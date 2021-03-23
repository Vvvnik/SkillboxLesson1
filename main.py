import nltk
import random

BOT_CONFIG = {
    'intents': {
        'hello': {
            'examples': ['Привет!', 'Здравсвуйте!))', 'Хай!!'],
            'responses': ['Прив!', 'Хеллоу', 'Как жизнь?']
        },
        'bye': {
            'examples': ['Пока!', 'До свиданья!', 'Увидимся!!'],
            'responses': ['Чао!', 'Будь здоров', 'Сайонара']
        },
        'science': {
            'examples': ['наука', 'исследования', 'учение'],
            'responses': ['естествознание', 'образование', 'академия']
        },
        'century': {
            'examples': ['век', 'эра', 'время'],
            'responses': ['вечность!', 'жизнь', 'бесконечность']
        },
        'ice': {
            'examples': ['gtr', 'мороз', 'лед'],
            'responses': ['зима', 'холодильник', 'заморозки']
        },
        'water': {
            'examples': ['дождь', 'жидкость', 'кипяток'],
            'responses': ['вода', 'пар', 'сырость']
        },
        'space': {
            'examples': ['dsf', 'космос', 'вселенная'],
            'responses': ['высота', 'высь', 'небосвод']
        },
        'auto': {
            'examples': ['авто', 'машина', 'транспорт'],
            'responses': ['автомобиль', 'автобус', 'доставка']
        }
    },
    'default_answers': ['Извините, я тупой', 'Переформулируйте, меня еще не обучили']
}  # "знания" бота

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

def bot(text): # функция бота
    intent = get_intent(text)  # 1. попытаться понять намерение
    if intent is not None:
        return random.choice(BOT_CONFIG['intents'][intent]['responses']) # 2. если удалось, ответить в соответствии намерением
    else:
        return random.choice(BOT_CONFIG['default_answers']) # 3. если не удалось, ответить заглушкой

question = ''
while question not in ['выход', 'выключайся']:
    question = input()
    answer = bot(question)
    print(answer)