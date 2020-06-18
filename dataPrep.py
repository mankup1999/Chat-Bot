import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle


lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore_words=["?","!"]

data_file=open("intents.json").read()
intents=json.loads(data_file)

for intent in intents["intents"]:
	for pattern in intent["patterns"]:
		w=nltk.word_tokenize(pattern)

		words.extend(w)
		
		documents.append((w,intent["tag"]))

		if intent["tag"] not in classes:
			classes.append(intent["tag"])


words=[lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

words=sorted(list(set(words)))

classes=sorted(list(set(classes)))

pickle.dump(words,open("words.pkl","wb"))
pickle.dump(classes,open("classes.pkl","wb"))
pickle.dump(documents,open("documents.pkl","wb"))







	
