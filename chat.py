
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: bag[i] = 1
    return(np.array(bag))

def predict(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]

    results = [ [i, r] for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results: return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            intent = i['tag']
            break
    return result, intent

def chatbot_response(msg):
    ints = predict(msg, model)
    res, intent = response(ints, intents)
    return res, intent

print('*** Start conversation *** \n\n')
msg = ''
intent = ''

while intent != 'goodbye':
    msg = input('You: ')
    res, intent = chatbot_response(msg)
    print('Chatbot: ', res)

print('\n\n *** Finished conversation! ***')
