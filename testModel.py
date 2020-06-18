import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer=WordNetLemmatizer()

model=load_model("chabotModel.h5")

intents=json.loads(open("intents.json").read())
words=pickle.load(open("words.pkl","rb"))
classes=pickle.load(open("classes.pkl","rb"))

def clean_up_sentence(msg):
	msg_words=nltk.word_tokenize(msg)
	msg_words=[lemmatizer.lemmatize(word.lower()) for word in msg_words]
	return msg_words

def bow(msg,words,show_details=True):
	msg_words=clean_up_sentence(msg)

	bag=[0]*len(words)

	for s in msg_words:
		for i,w in enumerate(words):
			if w==s:
				bag[i]=1
			if show_details:
				print(f"found in bag:{w}")
	return np.array(bag)

def predict_class(msg,model):
	p=bow(msg,words,show_details=False)

	p=p.reshape(1,88)

	res=model.predict([p])[0]
	
	ERROR_THRESHOLD=.40

	results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

	#print(results)
	
	results.sort(key=lambda X:X[1], reverse=True)

	#print(results)

	return_list=[]

	for r in results:
		return_list.append({"intent":classes[r[0]],"probability":str(r[1])})

	return return_list


def getResponse(ints,intents):
	tag=ints[0]["intent"]
	list_of_intents=intents["intents"]

	result=""
	#print(tag)
	for i in list_of_intents:
		#print(i["tag"])
		if i["tag"]==tag:
			result=random.choice(i["responses"])
			break
	return result

print("\n\n\n\n--------------------------------------------------------------------------------------")
while True:
	msg=input("You:")

	if msg=="exit()":
		break

	ints=predict_class(msg,model)
	print(f"/*-----{ints}-----*/")
	res=getResponse(ints,intents)

	print("Robot:",res)


print("\n\n\n\n--------------------------------------------------------------------------------------")




