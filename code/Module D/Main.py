from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply,GRU
from keras.models import Model

from attention_utils import get_activations
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

import  pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

SINGLE_ATTENTION_VECTOR = False

def attention_3d_block2(inputs, single_attention_vector=False):
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
  
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i + look_back,:])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y

def NormalizeMult(data):
    #normalize 用于反归一化
    data = np.array(data)
    normalize = np.arange(2*data.shape[1],dtype='float64')
    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0,data.shape[1]):
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    #np.save("./normalize.npy",normalize)
    return  data,normalize

def FNormalizeMult(data,normalize):
    data = np.array(data)
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow
    return data

def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
    x = Dropout(0.3)(x)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='relu')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

predicted_result=[]
filepath=r'...\MOHHO-MVMD_angle4.xlsx'
f=pd.ExcelFile(filepath)

for ii in f.sheet_names:
    dataset = pd.read_excel(filepath,sheet_name=ii,header=None)	

    INPUT_DIMS = dataset.shape[1]
    TIME_STEPS = 2
    epochs_number=100
    lstm_units = 64

    #Normalization
    data,normalize = NormalizeMult(dataset)
    pollution_data = data[:,0].reshape(len(data),1)

    train_X, _ = create_dataset(data,TIME_STEPS)
    _ , train_Y = create_dataset(pollution_data,TIME_STEPS)

    print(train_X.shape,train_Y.shape)

    m = attention_model()
    m.summary()
    m.compile(optimizer='adam', loss='mse')
    # division of trainsets and testsets
    division=0.2
    historyhistory,testtest_x,testtest_y=m.fit([train_X], train_Y, epochs=epochs_number, batch_size=64, validation_split=division,verbose=2)
    # prediction
    predict_result = m.predict(testtest_x)
    final_result=FNormalizeMult(predict_result,normalize)
    (hang,lie,duoyu)=np.array(testtest_y).shape
    x=list(range(0,lie,1))
    # figure
    plt.figure(figsize=(8,4))
    plt.plot(x,final_result,"b",label='Predicted wind power')
    dataset=dataset.values
    final_label = dataset[(len(dataset)-lie):,0].reshape(lie,1)
    plt.plot(x,final_label,"r",label='Actual wind power')
    plt.title('Wind farm prediction')
    plt.xlabel('Time')
    plt.ylabel('Wind power')
    plt.legend()
    predicted_result.append(final_result)
aaa=np.array(predicted_result)
ccc=0
for i in range(aaa.shape[0]):
    ccc=ccc+aaa[i]
# transform wind_data_values to pandas DataFrame
a_pd = pd.DataFrame(ccc)
# create writer to write an excel file
fileout=r'...\Mali_angle4.xlsx'
writer = pd.ExcelWriter(fileout)
# write in ro file, 'sheet1' is the page title, float_format is the accuracy of data
a_pd.to_excel(writer, 'sheet1', float_format='%.6f')
# save file
writer.save()
# close writer
writer.close()
end=time.time()
print(str(end-start))