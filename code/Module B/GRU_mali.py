import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.keras.layers import Dropout, Dense, GRU
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

input_num=2
epochs_number=20
predicted_result=[]
for ii in range(12):
    wind_data = pd.read_excel(r"...\windpower_angle4.xlsx",sheet_name=ii,header=None)
    wind_data_values_ndarray = wind_data.values
    wind_data_values_array = np.array(wind_data_values_ndarray)
    zhuanzhi=wind_data_values_array[:,0]
    maotai = zhuanzhi.reshape((len(zhuanzhi), 1))

    train_size = int(len(maotai) * 0.80)
    training_set = maotai[0:train_size,:] 
    test_set_set = maotai[train_size:,:]  

    # Normalization
    sc = MinMaxScaler(feature_range=(0, 1))  
    training_set_scaled = sc.fit_transform(training_set) 
    test_set = sc.transform(test_set_set)  

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for i in range(input_num, len(training_set_scaled)):
        x_train.append(training_set_scaled[i - input_num:i, 0])
        y_train.append(training_set_scaled[i, 0])

    np.random.seed(7)
    np.random.shuffle(x_train)
    np.random.seed(7)
    np.random.shuffle(y_train)
    tf.compat.v1.random.set_random_seed(7)
   
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], input_num, 1))
    for i in range(input_num, len(test_set)):
        x_test.append(test_set[i - input_num:i, 0])
        y_test.append(test_set[i, 0])
   
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], input_num, 1))

    model = tf.keras.Sequential([
        GRU(80, return_sequences=True),
        Dropout(0.2),
        GRU(100),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='mean_squared_error')  
    checkpoint_save_path = "./checkpoint/stock.ckpt"

    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     monitor='val_loss')

    history = model.fit(x_train, y_train, batch_size=64, epochs=epochs_number, validation_data=(x_test, y_test), validation_freq=1,
                        callbacks=[cp_callback],verbose=2)

    model.summary()

    file = open('./weights.txt', 'w') 
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # plt.figure()
    # plt.plot(loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()

    ################## predict ######################
    predicted_stock_price = model.predict(x_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    real_stock_price = test_set_set[input_num:]
    # Plot
    plt.figure()
    plt.plot(real_stock_price, color='red', label='Actual wind direction')
    plt.plot(predicted_stock_price, color='blue', label='Predicted wind direction')
    plt.title('Wind direction prediction')
    plt.xlabel('Time')
    plt.ylabel('Wind direction')
    plt.legend()
    # ##########evaluate##############
    # mse = mean_squared_error(predicted_stock_price, real_stock_price)
    # rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
    # mae = mean_absolute_error(predicted_stock_price, real_stock_price)
    # print('MSE: %.6f' % mse)
    # print('RMSE: %.6f' % rmse)
    # print('MAE: %.6f' % mae)
    predicted_result.append(predicted_stock_price)
aaa=np.array(predicted_result)
ccc=aaa[0]+aaa[1]+aaa[2]+aaa[3]+aaa[4]+aaa[5]+aaa[6]+aaa[7]+aaa[8]+aaa[9]+aaa[10]+aaa[11]

season_angle = pd.read_excel(r"...\windpower_angle4.xlsx",usecols=[2],header=None)
season_angle_values = season_angle.values

plt.figure()
plt.plot(season_angle_values[train_size+input_num:], color='red', label='Actual wind direction all')
plt.plot(ccc, color='blue', label='Predicted wind direction all')
plt.title('Wind direction prediction')
plt.xlabel('Time')
plt.ylabel('Wind direction')
plt.legend()
plt.show()

# transform wind_data_values to pandas DataFrame
a_pd = pd.DataFrame(ccc)
# create writer to write an excel file
writer = pd.ExcelWriter(r'...\mali.xlsx')
# write in ro file, 'sheet1' is the page title, float_format is the accuracy of data
a_pd.to_excel(writer, 'sheet1', float_format='%.6f')
# save file
writer.save()
# close writer
writer.close()