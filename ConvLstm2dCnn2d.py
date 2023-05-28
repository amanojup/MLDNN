# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 01:02:03 2020

@author: MANOJ KUMAR
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from matplotlib import pyplot
import pandas as pd
import numpy as np
import keras
from keras.layers import Input, Dense, Dropout, Concatenate,Flatten, BatchNormalization
from keras.layers import Conv1D, Conv2D, MaxPooling2D,MaxPooling3D, ConvLSTM2D, Conv3D,LSTM, AveragePooling2D, AveragePooling3D
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from keras import regularizers
import math
import time
# load the new file
spe = pd.read_csv('speedcatSEx2.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
print(spe.head())
print(spe.shape)
sample=len(spe)
print(sample)
# save
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler1 = MinMaxScaler(feature_range=(0, 1))
spe=scaler1.fit_transform(spe)
spe=np.reshape(spe,(sample,1,1,1,14))


# load the new file
target = pd.read_csv('IStargetSExNew.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
print(target.head())
print(target.shape)
# save
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler8 = MinMaxScaler(feature_range=(0, 1))
target=scaler8.fit_transform(target)
target=np.reshape(target,(sample,1))
print(target.shape)

#-------Set the prediction horizon and input time window--------------------------------------------

#--------------Create the input data set------------------------------------------------------------
train_spe= spe

                 
test_spe= target

#the dataset was divided into two parts: the training dataset and the testing dataset
train_size = int(len(train_spe) * 0.834)
X1=train_spe[0:train_size,:]                 


Y1=test_spe[0:train_size,:]                


y1=Y1


X1_test=train_spe[train_size:,:]                 




Y1_test=test_spe[train_size:,:]                 


y1_test=Y1_test

look_back=1
flters_no=10
flters_no1=10
fsize = 3
testsize = len(X1_test)
print(testsize)
#------------learn spatio-temporal feature from the speed data-----------------------------------------
spe_input = Input(shape=(look_back,1,1,14))

spe_input1 = BatchNormalization()(spe_input)
layer4 = ConvLSTM2D(filters=flters_no, kernel_size=(fsize, fsize),padding='same',kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01),data_format='channels_last', return_sequences=False)(spe_input1 )

layer4 = BatchNormalization()(layer4)
layer5 = Conv2D(filters=flters_no1, kernel_size=(fsize, fsize), data_format='channels_last',kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01),activation='relu', padding='same')(layer4)

flat1 = Flatten()(layer5)

              

               
#------------Combining the spatio-temporal information using a fusion layer----------------------------------
#merged_output = keras.layers.concatenate([layer2, layer4, layer6, layer8, layer10, layer12, layer14])
merged_output = flat1
out = keras.layers.Dense(1)(merged_output)
model = Model(inputs=spe_input, outputs=out)
model.compile(loss='mean_absolute_error', optimizer='Adamax')
start = time.time()
#-----------------------Record training history---------------------------------------------------------------
train_history = model.fit(X1, y1, epochs=80, batch_size=64, verbose=1,validation_data=(X1_test, y1_test))
print(X1.shape)
print(y1.shape)
#print(X1)
#print(y1)
#model.save('D:/4_Article_2021_DCP/Model_python/model270.h1')
#%%
#train_history = load_model('D:/4_Article_2021_DCP/Model_python/model270.h1')
#%%
loss = train_history.history['loss']
val_loss=train_history.history['val_loss']
end = time.time()
print (end-start)
plt.plot(train_history.history['loss'], label='train')
plt.plot(train_history.history['val_loss'], label='test')
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.show()
#--------------------------------Make prediction----------------------------------------------------------------
y1_pre = model.predict(X1_test)

print(y1_test.shape)
print(y1_pre.shape)

y1_test1 = scaler8.inverse_transform(y1_test)
y1_pre1 = scaler8.inverse_transform(y1_pre)
#y1_test1 = y1_test
#y1_pre1 = y1_pre
y1_test2=np.reshape(y1_test1,(1,testsize))
y1_pre2=np.reshape(y1_pre1,(1,testsize))



MSE=mean_squared_error(y1_pre1,y1_test1)
MAE=mean_absolute_error(y1_pre1,y1_test1)
# save the prediction values and the real values
#np.savetxt( 'test.txt',y1_test1)
# save the prediction values and the real values
#np.savetxt( 'pre.txt',y1_pre1 )
#--------------------------------Calculate evaluation index-----------------------------------------------------
mape= np.mean((abs(y1_test1- y1_pre1)) /y1_test1)
rmse=(y1_test1- y1_pre1)*(y1_test1- y1_pre1)
rm=np.sum(rmse)
RMSE=math.sqrt(rm/(rmse.size))
ape2=(abs(y1_test1- y1_pre1)) /y1_test1
ape22=ape2*ape2
summape2=np.sum(ape2)
summape22=np.sum(ape22)
len2=ape2.size
vape=math.sqrt((len2*summape22-summape2*summape2)/(len2*(len2-1)))
ec=(math.sqrt((np.sum((y1_test1- y1_pre1)**2))/len(y1_test1)))/(math.sqrt((np.sum(y1_test1**2))/len(y1_test1))+math.sqrt((np.sum(y1_pre1**2))/len(y1_test1)))
tic = (math.sqrt( (np.sum((y1_test1- y1_pre1)**2)) / len(y1_test1) )) / (math.sqrt((np.sum((y1_pre1)**2)) / len(y1_test1) ) + math.sqrt((np.sum((y1_test1)**2)) / len(y1_test1)))
cc = np.corrcoef(y1_test2, y1_pre2)
#print('MSE:', MSE)
print('RMSE:', RMSE)
print('MAE:', MAE)
print('MAPE' , mape)
#print('EC' , ec)
print('TIC' , tic)
print('cc' , cc)
print('Train Score: %.4f VAPE' % (vape))
# save the prediction values and the real values
#np.savetxt( 'test.txt',y1_test1)
#df1 = pd.DataFrame(y1_test1/2)
#df1.to_csv('Stest390.csv',index=False)
# save the prediction values and the real values
#np.savetxt( 'pre.txt',y1_pre1 )
#df = pd.DataFrame(y1_pre1/2)
#df.to_csv('Spre390.csv',index=False)