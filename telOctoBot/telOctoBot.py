import os
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize, regexp_tokenize
from string import punctuation
import pymorphy2
import requests
import json
import config
import math
import telebot
import re

bot = telebot.TeleBot(config.token)
# telebot.apihelper.proxy = {
#     'http': '127.0.0.1:9050',
#     'https': '127.0.0.1:9050'
#     }


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


def all_files(dir_path):
    directory = os.listdir(dir_path)
    for file in directory:
        json_file_update(dir_path+'/'+file)

def updateDataBase(dir_path):
    tf, df, idf = {}, {}, {}
    all_files(dir_path)
    data = data_load()
    total_dictionary_json(data, tf, df, idf)
    dictionary = dictionary_load()
    words_vectors_file_json(data)
    words_vectors = words_vectors_load()
    magn_file_json(data.keys(), words_vectors)
    magn = magnitudes_load()
    tf_df_idf_file_json(data, dictionary, tf, df, idf, magn)
    files_vectors_file_json(data.keys(), dictionary, tf, idf)



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

# input = [word1, word2, ...]
# output = {word1: [pos1, pos2], word2: [pos2, pos434], ...}


def index_one_file(termlist):
    fileIndex = {}
    for index, word in enumerate(termlist):
        if word in fileIndex.keys():
            fileIndex[word].append(index)
        else:
            fileIndex[word] = [index]
    return fileIndex

# input = {filename: [word1, word2, ...], ...}
# res = {filename: {word: [pos1, pos2, ...]}, ...}


def make_indices(termlists):
    total = {}
    for i in termlists:
        total[termlists['text_name']] = index_one_file(termlists['words'])
    return total


def get_text_words_with_index(path):
    index_words = index_one_file(
        tokenize_n_lemmatize(get_text(path), stopwords_ru))
    return index_words


# input = {filename: {word: [pos1, pos2, ...], ... }}
# res = {word: {filename: [pos1, pos2]}, ...}, ...}
def fullIndex(regdex, tf, df, idf):
	total_index = {}
	indie_indices = regdex
	for filename in indie_indices.keys():
		tf[filename] = {}
		for word in indie_indices[filename].keys():
			tf[filename][word] = len(indie_indices[filename][word])
			if word in df.keys():
				df[word] += 1
			else:
				df[word] = 1
			if word in total_index.keys():
				if filename in total_index[word].keys():
					total_index[word][filename].append(indie_indices[filename][word][:])
				else:
					total_index[word][filename] = indie_indices[filename][word]
			else:
				total_index[word] = {filename: indie_indices[filename][word]}
	return total_index

# input = {filename: {word: [pos1, pos2, ...], ... }}
# res = {word: {filename: [pos1, pos2]}, ...}, ...}


def tf_n_df(regdex):
    tf = {}
    df = {}
    indie_indices = regdex
    for filename in indie_indices.keys():
        tf[filename] = {}
        for word in indie_indices[filename].keys():
            tf[filename][word] = len(indie_indices[filename][word])
            if word in df.keys():
                df[word] += 1
            else:
                df[word] = 1
        return tf, df




def one_word_query(word):
	pattern = re.compile('[\W_]+')
	word = pattern.sub(' ',word)
	if word in dictionary.keys():
		return rankResults([filename for filename in data.keys()], word)
	else:
		return []

def free_text_query(string):
	pattern = re.compile('[\W_]+')
	string = pattern.sub(' ',string)
	result = []
	for word in string.split():
		result += one_word_query(word)
	return rankResults(list(set(result)), string)

def phrase_query(string):
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ', string)
    listOfLists, result = [], []
    for word in string.split():
        listOfLists.append(one_word_query(word))
    setted = set(listOfLists[0]).intersection(*listOfLists)
    for filename in setted:
        temp = []
        for word in string.split():
            temp.append(dictionary[word][filename][:])
        for i in range(len(temp)):
            for ind in range(len(temp[i])):
                temp[i][ind] -= i
        if set(temp[0]).intersection(*temp):
            result.append(filename)
    return rankResults(result, string)

def queryFreq(term, query):
    count = 0
    # queryls = tokenize_n_lemmatize(query)
    for word in query:
        if word == term:
            count += 1
    return count

def termfreq(terms, query):
	temp = [0]*len(terms)
	for i,term in enumerate(terms):
		temp[i] = queryFreq(term, query)
	return temp


def query_vec(query):
	pattern = re.compile('[\W_]+')
	query = pattern.sub(' ',query)
	queryls = tokenize_n_lemmatize(query)
	queryVec = [0]*len(queryls)
	index = 0
	for ind, word in enumerate(queryls):
		queryVec[index] = queryFreq(word, queryls)
		index += 1
	queryidf = [idf[word] for word in dictionary.keys()]
	magnitude = pow(sum(map(lambda x: x**2, queryVec)),.5)
	freq = termfreq(dictionary.keys(), queryls)
	#print('THIS IS THE FREQ')
	tf = [x/magnitude for x in freq]
	final = [tf[i]*queryidf[i] for i in range(len(dictionary.keys()))]
	#print(len([x for x in queryidf if x != 0]) - len(queryidf))
	return final

def document_frequency(term):
	if term in dictionary.keys():
		return len(dictionary[term].keys())
	else:
		return 0

def collection_size(data):
	return len(data)

def term_frequency(term, document, tf, mags):
	return tf[document][term]/mags[document] if term in tf[document].keys() else 0

def populateScores(data, dictionary, tf, df, idf, mags):
	for filename in data:
		for term in dictionary.keys():
			tf[filename][term] = term_frequency(term, filename, tf, mags)
			if term in df.keys():
				idf[term] = idf_func(collection_size(data), df[term])
			else:
				idf[term] = 0
	return df, tf, idf

def idf_func(N, N_t):
	if N_t != 0:
		return math.log(N/N_t)
	else:
	 	return 0

def generateScore(term, document, tf, idf):
		return tf[document][term] * idf[term]

def vectorize(data):
    vectors = {}
    for filename in data.keys():
        vectors[filename] = [len(data[filename][word])
                             for word in data[filename].keys()]
    return vectors

def make_vectors(documents, dictionary, tf, idf):
	vecs = {}
	for doc in documents:
		docVec = [0]*len(dictionary.keys())
		for ind, term in enumerate(dictionary.keys()):
			docVec[ind] = generateScore(term, doc, tf, idf)
		vecs[doc] = docVec
	return vecs


def magnitudes(documents, vectors):
    mags = {}
    for document in documents:
        mags[document] = pow(
            sum(map(lambda x: x**2, vectors[document])), .5)
    return mags


# Получение множества слов из файла


def get_text_words_set(path):
    words = set(tokenize_n_lemmatize(get_text(path), stopwords_ru))
    return words

# Множество слов в сообщении пользователя


def get_user_words_set(msg):
    words = set(tokenize_n_lemmatize(msg.lower(), stopwords_ru))
    return words


# Добавление файла в json файл


def json_file_update(text_path, json_file='OctoData.json'):

    words = index_one_file(tokenize_n_lemmatize(
        get_text(text_path), stopwords_ru))
    try:
        data = json.load(open(json_file))
    except:
        data = {}
    data[text_path] = words
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def total_dictionary_json(data, tf, df, idf, json_file='OctoDict.json'):
    data = fullIndex(data, tf, df, idf)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def words_vectors_file_json(data, json_file='OctoWordsVect.json'):
    vectors = vectorize(data)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(vectors, f, indent = 2, ensure_ascii=False)

def magn_file_json(data_keys, vectors, json_file='OctoMagn.json'):
    magn = magnitudes(data_keys, vectors)
    with open (json_file, 'w', encoding='utf-8') as f:
        json.dump(magn, f, indent =2, ensure_ascii = False)

def tf_df_idf_file_json(data, dictionary, tf, df, idf, magn, json_file='OctoParam.json'):
    tf, df, idf = populateScores(data, dictionary, tf, df, idf, magn)
    with open (json_file, 'w', encoding='utf-8') as f:
        json.dump([tf, df, idf], f, indent =2, ensure_ascii = False)

def files_vectors_file_json(files, dictionary, tf, idf, json_file='OctoFilesVec.json'):
    files_vectors = make_vectors(files, dictionary, tf, idf)
    with open (json_file, 'w', encoding ='utf-8') as f:
        json.dump(files_vectors, f, indent =2, ensure_ascii = False)


# Загрузка json данных


def data_load(json_file='OctoData.json'):
    data = json.load(open(json_file))
    return data

def dictionary_load(json_file='OctoDict.json'):
    dic = json.load(open(json_file))
    return dic

def words_vectors_load(json_file = 'OctoWordsVect.json'):
    vectors = json.load(open(json_file))
    return vectors

def magnitudes_load(json_file = 'OctoMagn.json'):
    magn = json.load(open(json_file))
    return magn

def tf_df_idf_load(json_file = 'OctoParam.json'):
    temp = json.load(open(json_file))
    return temp[0], temp[1], temp[2]

def files_vectors_load(json_file ='OctoFilesVec.json'):
    vect = json.load(open(json_file))
    return vect



# Анализ запроса пользователя
data = data_load()
dictionary = dictionary_load()
vectors = words_vectors_load()
mags = magnitudes_load()
tf, df, idf = tf_df_idf_load()
files_vectors = files_vectors_load()

def dotProduct(doc1, doc2):
	if len(doc1) != len(doc2):
		return 0
	return sum([x*y for x,y in zip(doc1, doc2)])

def rankResults(resultDocs, query):
	# print(vectors)
	queryVec = query_vec(query)
	# print(queryVec)
	results = [[dotProduct(files_vectors[result], queryVec), result]
	                       for result in resultDocs]
	# print(results)
	results.sort(key=lambda x: x[0])
	# print(results)
	results = [x[1] for x in results]
	return results

# def bot_response(user_response):
#     set_response = get_user_words_set(user_response)
#     cor = [len(i['words'] & set_response) for i in data]
#     answer = cor.index(max(cor))
#     return data[answer]['text_path']

def bot_response(user_response):
    res = free_text_query(user_response).pop()
    return res

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет!')


@bot.message_handler(content_types=['text'])
def handle_text_messages(message):
    res_message = message.text.lower()
    if res_message == 'привет':
        bot.send_message(message.from_user.id, "Я Октопус - твой юридический помощник в вузе. Опиши свою проблему или задай мне вопрос. \
			Если моя помощь тебе не нужна - напиши 'Мяу'")
    elif res_message == 'мяу':
        bot.send_message(message.from_user.id, 'Мур :* \n До встречи!')
    else:
        path = bot_response(res_message)
        bot.send_message(message.from_user.id, "Может быть это поможет?")
        doc = open(path, 'rb')
        bot.send_document(message.from_user.id, doc)


# if __name__ =='__main__':
bot.polling()
