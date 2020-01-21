#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libaries
from keras.models import Sequential
from keras.layers.core import Dense, Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


import pandas as pd
import io

data = pd.read_csv(io.StringIO(uploaded['train.csv'].decode('utf-8')))


# In[ ]:


data.shape


# In[ ]:


data.shape
data=np.array(data)


# In[ ]:


X=data[:,1:785]


# In[ ]:


Y=data[:,0]


# In[ ]:


X.shape


# In[ ]:


Y.shape


# In[ ]:


X=X.reshape(60000, 28, 28)
X.shape


# In[ ]:


plt.subplot(221)
plt.imshow(X[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


# In[ ]:


num_pixels = X.shape[1] * X.shape[2]
X_train = X.reshape(X.shape[0], num_pixels).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')


# In[ ]:


X=X.reshape(60000, 784)
X.shape


# In[ ]:


X_train = X/255


# In[ ]:


# one hot encode outputs
y_train = np_utils.to_categorical(Y)
#y_test = np_utils.to_categorical(y_test)
#num_classes = Y.shape[1]


# In[ ]:


# define baseline model(simple dense neural network)
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# In[ ]:


# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=200, verbose=2)


# In[ ]:





# In[ ]:


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#K.set_image_dim_ordering('th')


# In[ ]:


X_train = X_train.reshape(X_train.shape[0], 28, 28,1).astype('float32')


# In[ ]:


def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(28, 28,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# In[ ]:


model = baseline_model()
# Fit the model
model.fit(X_train, y_train,validation_split=0.2 , epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


import pandas as pd
import io

data_1 = pd.read_csv(io.StringIO(uploaded['test.csv'].decode('utf-8')))


# In[ ]:


data_1.shape
data_1=np.array(data_1)


# In[ ]:


data_1.shape


# In[ ]:


data_1[1:20,784]


# In[ ]:


X_1=data_1[:,1:785]
Y_1=data_1[:,0]


# In[ ]:


X_1.shape
#Y_1.shape


# In[ ]:


Y_1.shape


# In[ ]:


Y_1


# In[ ]:


X_test = X_1/255


# In[ ]:


y_test = np_utils.to_categorical(Y_1)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
color_ohe = OneHotEncoder()
make_ohe = OneHotEncoder()
y_test = color_ohe.fit_transform(Y_1.reshape(-1,1)).toarray()


# In[ ]:


y_test.shape


# In[ ]:


X_test = X_test.reshape(X_test.shape[0], 28, 28,1).astype('float32')


# In[ ]:


scores = model.predict(X_test)
#print("CNN Error: %.2f%%" % scores[1]*100))


# In[ ]:


scores


# In[ ]:


aa=np.argmax(scores,axis=1)


# In[ ]:


aa[1:10]


# In[ ]:





# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


import pandas as pd
import io

data_2 = pd.read_csv(io.StringIO(uploaded['sample_submission.csv'].decode('utf-8')))


# In[ ]:


data_2.shape


# In[ ]:


data_2[1:10]


# In[ ]:


submt = pd.DataFrame({'id': Y_1, 'label':aa})


# In[ ]:



#submt= pd.DataFrame[[Y_1], [aa]]


# In[ ]:


submt[1:10]


# In[ ]:


submt.set_index(['id'], inplace=True)
submt[1:10]


# In[ ]:





# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


import pandas as pd
import io

data_3 = pd.read_csv(io.StringIO(uploaded['Dig-MNIST.csv'].decode('utf-8')))


# In[ ]:


data_3.shape


# In[ ]:


data_3=np.array(data_3)


# In[ ]:


XX=data_3[:,1:785]


# In[ ]:


XX=XX/255


# In[ ]:


XX=XX.reshape(10240,28,28,1).astype('float32')


# In[ ]:


YY=data_3[:,0]


# In[ ]:


YY.shape


# In[ ]:


YY = np_utils.to_categorical(YY)


# In[ ]:


scores = model.evaluate(XX, YY, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


# In[ ]:


model.predict(XX)


# In[ ]:





# In[ ]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Kannda MNIST Predictions 1.csv'

submt.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[ ]:


scaler =StandardScaler()


# In[ ]:


X_scaled=scaler.fit_transform(X)


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca=PCA(n_components=256)


# In[ ]:


pca.fit(X_scaled)


# In[ ]:


X_pca=pca.transform(X_scaled)


# In[ ]:


X_pca.shape


# In[ ]:


X_pca_res=X_pca.reshape(60000,16,16)
X_pca_res.shape


# In[ ]:


plt.subplot(221)
plt.imshow(X_pca_res[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_pca_res[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_pca_res[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_pca_res[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


# In[ ]:


nb_epoch = 25
num_classes = 10
batch_size = 128
train_size = 60000
test_size = 10000
v_length = 256


# In[ ]:


X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, 28, 28).astype('float32')


# In[ ]:


from keras.utils import to_categorical
Y = to_categorical(Y)


# In[ ]:


import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, UpSampling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#K.set_image_dim_ordering('th')
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization


# In[ ]:




