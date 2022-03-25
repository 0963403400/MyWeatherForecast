
# from DataAPI import DataAPIs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .Tuan_Forescast import forecast
# file_loc = "DataUT/Daily_rainfall_Exp_Student.xlsx"
# df = pd.read_excel(file_loc,sheet_name='Sheet1', usecols="B")
#
# series = np.array(df)
# series=series.reshape(1,1,-1)
# print(series.shape[2])
#
# series_Input= np.array()


data = pd.read_csv('DataUT/rainfall.csv')

data_end = int(np.floor(0.7*(data.shape[0])))
train = data[0:data_end]['48805'].values.reshape(-1)
test = data[data_end:]['48805'].values.reshape(-1)
date_test = data[data_end:]['TimeVN'].values.reshape(-1)

def get_data(train, test, time_step, num_predict, date):
    x_train = list()
    y_train = list()
    x_test = list()
    y_test = list()
    date_test = list()

    for i in range(0, len(train) - time_step - num_predict):
        x_train.append(train[i:i + time_step])
        y_train.append(train[i + time_step:i + time_step + num_predict])

    for i in range(0, len(test) - time_step - num_predict):
        x_test.append(test[i:i + time_step])
        y_test.append(test[i + time_step:i + time_step + num_predict])
        date_test.append(date[i + time_step:i + time_step + num_predict])

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), np.asarray(date_test)
x_train, y_train, x_test, y_test, date_test = get_data(train,test,360,1, date_test)

Tuan=forecast()
print(Tuan.Model_predict("4831",360,7,x_test[0],1,train))
# y_pred= list()
# for x in x_test:
#     y_pred.append(Tuan.Model_predict("4831",360,7,x,1,train))
# y_pred=np.asarray(y_pred).reshape(-1,1)
# plt.plot(y_test, color='b')
# plt.plot(y_pred ,color='r')
# plt.title("Biểu đồ dự báo lượng mưa trạm 48/31")
# plt.xlabel("Thời gian")
# plt.ylabel("Lượng mưa")
# plt.legend(('Thực tế', 'Dự báo'),loc='upper right')
# plt.show()

