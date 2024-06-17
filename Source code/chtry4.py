import random
import json
import pickle
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense ,Activation, Dropout
from tensorflow.keras.optimizers import SGD
import json
lemmatizer =WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words=[]
classes=[]
documents=[]
ignore_letters= ['?','!','.',',']
for intent in intents['intents']:
  for pattern in intent['patterns']:
    word_list = nltk.word_tokenize(pattern)
    words.extend(word_list)
    documents.append((word_list,intent['tag']))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])
print(documents)
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))
#print(words)

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]
model = Sequential()
model.add(Dense(128 ,input_shape=(len(trainX[0]),), activation = 'relu') )
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(trainY[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
  
print('Done')

lemmatizer =WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words =pickle.load( open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('chatbot_model.h5')
def clean_up_sentence(sentence):
  sentense_words = nltk.word_tokenize(sentence)
  sentense_words = [lemmatizer.lemmatize(word) for word in sentense_words]
  return sentense_words
def bag_of_words(sentence):
  sentence_words = clean_up_sentence(sentence)
  bag = [0]*len(words)
  for w in sentence_words:
    for i ,word in enumerate(words):
      if word == w :
        bag[i]=1
  return np.array(bag)
def predict_class(sentence):
  bow = bag_of_words(sentence)
  res = model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD = 0.25
  results =[[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

  results.sort(key =lambda x: x[1] ,reverse=True)
  return_list =[]
  for r in results:
    return_list.append({'intent': classes[r[0]],'probability': str(r[1]) })
  return return_list
def get_response(intents_list,intents_json):
  tag = intents_list[0]['intent']
  list_of_intents = intents_json['intents']
  for i in list_of_intents:
    if i['tag'] == tag :
      result = random.choices(i['responses'])
      break
  return result

def process(massage):
  inp= massage
  ints = predict_class(inp)
  res = get_response(ints , intents)
  return (res)



from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def home():
  return render_template("index.html")

@app.route("/get")
def get_bot_reponse():
    userText = request.args.get('msg')
    return str(process(userText))

if __name__ == "__main__":
    app.run(debug=True)





# print ("GO! bot is running!")
# while True:
#   message = input("")
#   ints = predict_class(message)
#   res = get_response(ints , intents)
#   print (res)


