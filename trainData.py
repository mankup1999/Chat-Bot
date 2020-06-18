import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import random
import numpy as np

lemmatizer=WordNetLemmatizer()

words=pickle.load(open("words.pkl","rb"))
classes=pickle.load(open("classes.pkl","rb"))
documents=pickle.load(open("documents.pkl","rb"))

training=[]

output_empty=[0]*len(classes)

for doc in documents:
	bag=[]
	pattern_words=doc[0]
	pattern_words=[lemmatizer.lemmatize(word.lower()) for word in pattern_words]

	for w in words:
		if w in pattern_words:
			bag.append(1)
		else:
			bag.append(0)

	output_row=list(output_empty)
	output_row[classes.index(doc[1])]=1

	training.append([bag,output_row])

random.shuffle(training)
training=np.array(training)

train_x=list(training[:,0])
train_y=list(training[:,1])

pickle.dump(train_x,open("trainX.pkl","wb"))
pickle.dump(train_y,open("trainY.pkl","wb"))



