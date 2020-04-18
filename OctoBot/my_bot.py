import os
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize, regexp_tokenize
from string import punctuation
import pymorphy2
import requests
import json


# Чтение файла


def get_text(url, encoding='utf-8', to_lower=True):
    url = str(url)
    if url.startswith('http'):
        r = requests.get(url)
        if not r.ok:
            r.raise_for_status()
        return r.text.lower() if to_lower else r.text
    elif os.path.exists(url):
        with open(url, encoding=encoding, errors='ignore') as f:
            return f.read().lower() if to_lower else f.read()
    else:
        raise Exception('parameter [url] can be either URL or a filename')


# Получение стоп слов
url_stopwords_ru = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"
stopwords_ru = get_text(url_stopwords_ru).splitlines() + \
    [char for char in punctuation]+['*', 'т.д.']


# Нормализация слов

def normalize_tokens(tokens):
    morph = pymorphy2.MorphAnalyzer()
    return [morph.parse(tok)[0].normal_form for tok in tokens]


# Удаление частоиспользуемых слов и слов > 3 символов

def remove_stopwords(tokens, stopwords=None):
    if not stopwords:
        return tokens
    stopwords = set(stopwords)
    tokens = [tok for tok in tokens if tok not in stopwords and len(tok) > 3]
    return tokens


# Токенизация текста и нормализация слов

def tokenize_n_lemmatize(text, stopwords=None, normalize=True):
    words = [sent for sent in nltk.word_tokenize(text)]
    if normalize:
        words = normalize_tokens(words)
    if stopwords:
        words = remove_stopwords(words, stopwords)
    return words


# Получение множества слов из файла

def get_text_words_set(path):
    words = set(tokenize_n_lemmatize(get_text(path), stopwords_ru))
    return words

#Множество слов в сообщении пользователя

def get_user_words_set(msg):
    words = set(tokenize_n_lemmatize(msg.lower(), stopwords_ru))
    return words



# Добавление файла в json файл


def json_file_update(text_path, text_name, json_file='OctoData.json'):

    words = list(get_text_words_set(text_path))
    text = {
        'text_name': text_name,
        'words': words
    }
    try:
        data = json.load(open(json_file))
    except:
        data = []
    data.append(text)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Загрузка json данных

def data_load(json_file='OctoData.json'):
    data = json.load(open(json_file))
    for i in data:
        i['words'] = set(i['words'])
    return data

# Анализ запроса пользователя
def bot_response(user_response):
    set_response = get_user_words_set(user_response)
    cor = [len(i['words']&set_response) for i in data]
    answer_i = cor.index(max(cor))
    return data[answer_i]['text_name']

flag = True
data = data_load()
print("Привет! Я Октопус - твой юридический помощник в вузе. Опиши свою проблему или задай мне вопрос. \
Если моя помощь тебе не нужна - напиши 'Мяу'")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'мяу'):
        x = bot_response(user_response)
        print("Может быть это поможет?")
        print(x)
    else:
        flag = False
        print("Удачи! Не заблудись в 'замке' ;-)")

