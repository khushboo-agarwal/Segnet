''' Import all the necessary libaries'''

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Layer, Activation, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, UpSampling1D
from keras.layers.normalization import BatchNormalization
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
import keras.models as models
from keras.optimizers import SGD
import cv2, numpy as np
from keras import backend as K
import h5py
import matplotlib.pyplot as plt
#import keras.utils.visualize_util as vutil
#from IPython.display import SVG


''' Changed the data shape to 224 * 224. Use image_resize.py to change the shape of the images.
	Defined the number of classes for this dataset
	Define the path to CamVid folder according to your machine'''
data_shape = 224*224
nb_class = 12
path = '/Users/administrator/PDFS/MachineLearning/december/Segnet/CamVid/'



'''
	Input:    Image (rgb) 
	Output:   Normalized Image
	Function: Normalized the Input Image
	'''
def normalized(rgb):
	
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

'''
	Input:	 Input Image of shape (3, 224, 224)
	Output:  Output Volume of shape (224*224, 12)
	Functon: It splits each and every pixel into its classes. This function labels all the pixels passed. 
	'''
def binarylab(labels):

    x = np.zeros([224,224,12])    
    for i in range(224):
        for j in range(224):
            x[i,j,labels[i][j]]=1
    #print x[:,:,0]

    return x
'''
	Input:   -
	Output:   Array of data and labels
	Function: reads the text file, with all the path to the images. Normalizes the images and makes an array of image, also labels all the pixels and outputs a volume of the same as an array of labels. 
	'''
def prep_data():

    train_data = []
    train_label = []
    import os
    with open('/Users/administrator/PDFS/MachineLearning/december/Segnet/CamVid/'+'train.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        ## these paths are very specific to my machine
        train_data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
        train_label.append(binarylab(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
        print "train"
    return np.array(train_data), np.array(train_label)
'''
prepares the train data and train labels
reshape labels to (samples, data shape, classes)
'''
train_data, train_label = prep_data()
train_label = np.reshape(train_label,(367,data_shape,12))
#print train_label.shape

#this comes from VGG 16 network for each class weighting
class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

def create_encoding_layers():
	'''
	Input: 
	Output:   
	Function:  Adds the first 13 layers of the VGG 16 into the encoding part of the auntoencoder. 

	'''
	model.add(ZeroPadding2D(padding = (1,1), dim_ordering='th'))
	model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid', dim_ordering='th'))
	model.add( BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering='th'))
	model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid', dim_ordering='th'))
	model.add( BatchNormalization())
	model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering='th'))

	model.add(ZeroPadding2D((1,1), dim_ordering='th'))
	model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='valid', dim_ordering='th'))
	model.add( BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering='th'))
	model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='valid', dim_ordering='th'))
	model.add( BatchNormalization())
	model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering='th'))

	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='valid', dim_ordering = 'th'))
	model.add( BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='valid', dim_ordering = 'th'))
	model.add( BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='valid', dim_ordering = 'th'))
	model.add( BatchNormalization())
	model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering = 'th'))

	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='valid', dim_ordering = 'th'))
	model.add( BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='valid', dim_ordering = 'th'))
	model.add( BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='valid', dim_ordering = 'th'))
	model.add( BatchNormalization())
	model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering = 'th'))


	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='valid', dim_ordering = 'th'))
	model.add( BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='valid', dim_ordering = 'th'))
	model.add( BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='valid', dim_ordering = 'th'))
	model.add( BatchNormalization())
	model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering = 'th'))



def create_decoding_layers():
	'''
	Input:
	Output:
	Function:  Adds the corresponding upsampling layers to the decoder of the autoencoder
	'''
	model.add(UpSampling2D(size = (2, 2), dim_ordering = 'th'))
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, border_mode='valid', dim_ordering = 'th'))
   	model.add(BatchNormalization())
   	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, border_mode='valid', dim_ordering = 'th'))
	model.add(BatchNormalization())
   	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, border_mode='valid', dim_ordering = 'th'))
	model.add(BatchNormalization())


	model.add(UpSampling2D(size = (2,2), dim_ordering = 'th'))
	model.add(ZeroPadding2D(padding=(1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, border_mode='valid', dim_ordering = 'th'))
	model.add(BatchNormalization())
   	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, border_mode='valid', dim_ordering = 'th'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(512, 3, 3, border_mode='valid' , dim_ordering = 'th'))
	model.add(BatchNormalization())

	model.add(UpSampling2D(size = (2, 2), dim_ordering = 'th'))
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(256, 3, 3, border_mode='valid', dim_ordering = 'th'))
	model.add(BatchNormalization())
   	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(256, 3, 3, border_mode='valid', dim_ordering = 'th'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(256, 3, 3, border_mode='valid', dim_ordering = 'th'))
	model.add(BatchNormalization())

	model.add(UpSampling2D(size = (2, 2), dim_ordering = 'th'))
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(128, 3, 3, border_mode='valid', dim_ordering = 'th'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(128, 3, 3, border_mode='valid', dim_ordering = 'th'))
	model.add(BatchNormalization())

	model.add(UpSampling2D(size = (2, 2), dim_ordering = 'th'))
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(64, 3, 3, border_mode='valid', dim_ordering = 'th'))
	model.add(BatchNormalization())
	model.add(ZeroPadding2D((1,1), dim_ordering = 'th'))
	model.add(Convolution2D(64, 3, 3, border_mode='valid', dim_ordering = 'th'))
	model.add(BatchNormalization())

'''
Initialize the model as Sequential
'''
model = models.Sequential()

'''
Add a denoising layers
'''

model.add(Layer(input_shape = (3, 224, 224)))

'''
Create the autoencoder by adding the layers sequentially. First the encoder and then the decoder. 
'''
model.encoding_layers = create_encoding_layers()
model.decoding_layers = create_decoding_layers()

'''
Add a softmax layer, reshape it and permute it. 
'''
model.add(Convolution2D(nb_class, 1, 1, border_mode = 'valid', dim_ordering = 'th'))
model.add(Reshape((12,224*224), input_shape=(12,224,224)))
model.add(Permute((2, 1)))
model.add(Activation('softmax'))

''' Initializing the model with SGD, and compiling it
	Also calling the model.summary function to represent the output shape of each layer and its connection'''
from keras.optimizers import SGD
sgd = SGD(lr=0.01, momentum=0.8, decay=1e-6, nesterov=False)
model.compile(loss="categorical_crossentropy", optimizer= sgd, metrics = ['accuracy'])
model.summary()

'''
Define the number of epoch
Define the batch size for each epoch
'''

nb_epoch = 5
batch_size = 32


'''
Uncomment the following the to save the weight file and fit the program. 
'''
# history = model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch,
#                    show_accuracy=True, verbose=1, class_weight=class_weighting )#, validation_data=(X_test, X_test))
# model.save_weights('modelKusTun_weight_ep100.hdf5')

'''
Load the weights in the weight file. 
Define color sequence for each and every class.
Load them in an array  
'''
model.load_weights('modelKusTun_weight_ep100.hdf5')
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
#Unlabelled = [0,0,0]

label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist])#, Unlabelled])

def visualize(temp, plot = True):
	'''
	Input: shape of the input is (224, 224, 12)
	Output: visualized image
	Functiom: Asssigns colors to each and every class. 
	'''

	r = temp.copy()
	g = temp.copy()
	b = temp.copy()

	for l in range(0, 11):
		r[temp == l] = label_colours[l, 0]
		g[temp == l] = label_colours[l, 1]
		b[temp == l] = label_colours[l, 2]
	rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
	rgb[:,:,0] = (r/255.0)#[:,:,0]
	rgb[:,:,1] = (g/255.0)#[:,:,1]
	rgb[:,:,2] = (b/255.0)#[:,:,2]
	if plot:
		plt.imshow(rgb)
	else:
		# print rgb
		return rgb

'''
	Input:   -
	Output:   Array of data and labels
	Function: reads the text file, with all the path to the images. Normalizes the images and makes an array of image, also labels all the pixels and outputs a volume of the same as an array of labels. 
	'''
def t_data():
	
    test_data = []
    test_label = []
    import os
    with open('/Users/administrator/PDFS/MachineLearning/december/Segnet/CamVid/'+'test.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
        print len(txt)
        print txt[1][0][7:]
        print os.getcwd()
    for i in range(len(txt)):
        ## these paths are very specific to my machine
        #print os.getcwd(path)
        print os.getcwd() + txt[i][1][7:][:-1]
        test_data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
        test_label.append(binarylab(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
        #print('.',end='')
        print "test"
    return np.array(test_data), np.array(test_label)

test_data, test_label = t_data()

gt = []
with open(path+'test.txt') as f:
    txt = f.readlines()
    txt = [line.split(' ') for line in txt]
for i in range(len(txt)):
    gt.append(cv2.imread(os.getcwd() + txt[i][0][7:]))



output = model.predict_proba(test_data[2:3])

#print 'test_data shape is:',test_data.shape 
print 'output_data shape is:',output.shape
print output[0]
print len(output[0])
#output = model.predict_proba(train_data[2:3])
pred = visualize(np.argmax(output[0],axis=1).reshape((224,224)), False)
# pred = visualize(np.argmax(output[0],axis=1).reshape((height,width)), False)
plt.imshow(pred)
plt.figure(2)
plt.imshow(gt[0])
plt.show()









