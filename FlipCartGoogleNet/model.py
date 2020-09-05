from numpy import array
from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from keras.layers.merge import add
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
%matplotlib inline
import tensorflow as tf
import imageio
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
from pickle import dump
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from os import listdir

def GoogleNet():
  
    #installing CNN
    model = Sequential()
    
    #Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)
    model.add(Conv2D(64, (7, 7),activation ='sigmoid',padding='same',input_shape = (640,435,3)))
    
    #step-2 pooling
    model.add(MaxPooling2D (pool_size=(3,3), strides=(1,1)))
    
    #step3
    model.add(Conv2D(192, (3, 3),activation ='sigmoid',padding='same',strides = (1,1)))
    
    #step:4
    model.add(MaxPooling2D (pool_size=(3,3), strides=(2,2)))
    
    #step:5 inception(3a)
    model.add(Conv2D(filters = 256, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #step:6 (adding pooling layer to inception(3a))
    model.add(MaxPooling2D (pool_size=(3,3)))
    
    #step7
    model.add(Conv2D(filters = 480, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 480, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 480, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #step:8 (adding pooling layer to inception(3))
    #model.add(MaxPooling2D (pool_size=(3,3)))
    
    #step9
   # model.add(MaxPooling2D (pool_size=(3,3), strides=(1,1)))
    
    #step10
    model.add(Conv2D(filters = 512, kernel_size = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same'))
    model.add(Conv2D(filters = 512, kernel_size = (5,5), padding = 'same'))
    #(adding pooling layer to inception(3))
    #model.add(MaxPooling2D (pool_size=(3,3)))
    
    #step11
    model.add(Conv2D(filters = 512, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 512, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #(adding pooling layer to inception(3))
    #model.add(MaxPooling2D (pool_size=(2,2)))
    
    #step12
    model.add(Conv2D(filters = 512, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 512, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #(adding pooling layer to inception(3))
    #model.add(MaxPooling2D (pool_size=(100,100)))
    
    #step13
    model.add(Conv2D(filters = 528, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 528, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 528, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #(adding pooling layer to inception(3))
    #model.add(MaxPooling2D (pool_size=(3,3)))
    
    #step14
    model.add(Conv2D(filters = 832, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 832, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 832, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    
    model.add(Conv2D(filters = 832, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 832, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 832, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #(adding pooling layer to inception(3))
    #model.add(MaxPooling2D (pool_size=(3,3)))
    
    #step15
    model.add(MaxPooling2D (pool_size=(3,3)))
    
    #step16
    model.add(Conv2D(filters = 832, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 832, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 832, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #(adding pooling layer to inception(3))
    model.add(MaxPooling2D (pool_size=(3,3)))
    
    #step17
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
   # model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #(adding pooling layer to inception(3))
    #model.add(MaxPooling2D (pool_size=(3,3)))
    
    #step18
    model.add(AveragePooling2D(pool_size=(1,1), strides =(1,1)))
    
    #step19
    #step17
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #(adding pooling layer to inception(3))
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #(adding pooling layer to inception(3))
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    
    
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #(adding pooling layer to inception(3))
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    #(adding pooling layer to inception(3))
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    
    model.add(Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Conv2D(filters = 1024, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    
    model.add(MaxPooling2D (pool_size=(3,3)))
    
    #step20
    model.add(Flatten())
    
    #128
    model.add(Dense(activation = 'sigmoid', units=128))
   
    print(model.summary())
	
    return model

# extract features from each photo in the directory
# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model = GoogleNet()
	# summarize
	print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(500,330))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=1)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features


# extract features from all images
directory = 'Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))





from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)

# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(128,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

# train dataset

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)

# dev dataset

# load test set
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)

# fit model

# define the model
model = define_model(vocab_size, max_length)
# define checkpoint callback
filepath = 'model-ep003-loss3.614-val_loss3.845.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=25, verbose=1, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))




from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc_list in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# prepare tokenizer on train set

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare test set

# load test set
filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
filename = 'model-ep003-loss3.614-val_loss3.845.h5'
model = load_model(filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

