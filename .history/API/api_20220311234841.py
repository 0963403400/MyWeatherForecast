from crypt import methods
from distutils.log import debug
from flask import Flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .Tuan_Forescast import forecast
from .__init__ import get_data


app=Flask(__name__)

# @app.route('/',methods=['GET'])
# def api():   
#     data = pd.read_csv('DataUT/rainfall.csv')

#     data_end = int(np.floor(0.7*(data.shape[0])))
#     train = data[0:data_end]['48805'].values.reshape(-1)
#     test = data[data_end:]['48805'].values.reshape(-1)
#     date_test = data[data_end:]['TimeVN'].values.reshape(-1)
#     x_train, y_train, x_test, y_test, date_test = get_data(train,test,360,1, date_test)

#     Tuan=forecast()
#     # print(Tuan.Model_predict("4831",360,7,x_test,1,train))
#     return{
#         "Tuan":Tuan.Model_predict("4831",360,7,x_test[0],1,train)[0],
#         "Tittle":"VipPro",
#     }

@app.route('/Position/<Province>')
def api(Province):   
    data = pd.read_csv('DataUT/rainfall.csv')
    print(Province)

    # data_end = int(np.floor(0.7*(data.shape[0])))
    # train = data[0:data_end]['48805'].values.reshape(-1)
    # test = data[data_end:]['48805'].values.reshape(-1)
    # date_test = data[data_end:]['TimeVN'].values.reshape(-1)
    # x_train, y_train, x_test, y_test, date_test = get_data(train,test,360,1, date_test)

    # Tuan=forecast()
    # print(Tuan.Model_predict("4831",360,7,x_test,1,train))
    return{
        # "Tuan":Tuan.Model_predict("4831",360,7,x_test[0],1,train)[0],
        "Tuan":Province,
        "Tittle":"VipPro",
    }