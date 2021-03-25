import nltk
import random
import json
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def cleaner(text): # функция очистки текста
    cleaned_text = ''
    for ch in text.lower():
        if ch in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz ':
            cleaned_text = cleaned_text + ch
    return cleaned_text

def match(text, example): # гибкая функция сравнения текстов
    return nltk.edit_distance(text, example) / len(example) < 0.4 if len(example) > 0 else False

def get_intent(text): # функция определения интента текста
    for intent in BOT_CONFIG['intents']:
        if 'examples' in BOT_CONFIG['intents'][intent]:
             for example in BOT_CONFIG['intents'][intent]['examples']:
                  if match(cleaner(text), cleaner(example)):
                       return intent

with open('content/BIG_BOT_CONFIG.json', 'r') as f:
    BOT_CONFIG = json.load(f)  # читаем json в переменную BOT_CONFIG

X = []
y = []

for intent in BOT_CONFIG['intents']:
    if 'examples' in BOT_CONFIG['intents'][intent]:
        X += BOT_CONFIG['intents'][intent]['examples']
        y += [intent for i in range(len(BOT_CONFIG['intents'][intent]['examples']))]

vectorizer = TfidfVectorizer(preprocessor=cleaner, ngram_range=(1, 3))
# Создаем векторайзер – объект для превращения текста в вектора
vectorizer.fit(X)
X_vect = vectorizer.transform(X)

# Обучаем векторайзер на нашей выборке
X_train_vect, X_test_vect, y_train, y_test = train_test_split(X_vect, y, test_size=0.3)
# Разбиваем выборку на train и на test

ExtraTree = ExtraTreesClassifier(n_estimators=10)  # Создаем модель0.806598712446352   , stop_words=['а', 'и']
ExtraTree.fit(X_train_vect, y_train)  # Обучаем модель
print(ExtraTree.score(X_test_vect, y_test))  # Проверяем качество модели на тестовой выборке 0,022
ExtraTree.fit(X_vect, y)
print(ExtraTree.score(X_vect, y))  # Смотрим качество классификации 0,802


# sgd = SGDClassifier() # Создаем модель
# sgd.fit(X_train_vect, y_train) # Обучаем модель
# print('SGDClassifier train = '+ str(sgd.score(X_test_vect, y_test))) # Проверяем качество модели на тестовой выборке
# sgd.fit(X_vect, y)
# print('SGDClassifier = '+ str(sgd.score(X_vect, y))) # Смотрим качество классификации

# 0.7824570815450643


def get_intent_by_model(text):  # Функция определяющая интент текста с помощью ML-модели
    return ExtraTree.predict(vectorizer.transform([text]))[0]


def bot(text):  # функция бота
    intent = get_intent(text)  # 1. попытаться понять намерение сравнением по Левинштейну
    if intent is None:
        intent = get_intent_by_model(text)  # 2. попытаться понять намерение с помощью ML-модели
    return random.choice(BOT_CONFIG['intents'][intent]['responses'])


# question = ''
# while question not in ['выход', 'выключайся']:
#     question = input()
#     answer = bot(question)
#     print(answer)


# Enable logging
import logging

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help_command(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update: Update, _: CallbackContext) -> None:
    """Echo the user message."""
    update.message.reply_text(bot(update.message.text))


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("1673172061:AAFS7dyJsDTNViMg2HugWBSfv3KAc2iiwnQ")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()