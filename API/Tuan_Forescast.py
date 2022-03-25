import sys
import numpy as np
import os
# import yaml

# from DataAPI import DataAPIs
from .ATM_Model_API import ATM_Model
# from core.logger import Logger

import xlrd

class forecast:

    def Model_train(self, tid, series, enc_len, dec_len):
        'Train model'
        # if tid not in self.data.List_ATM_Series:
        #     return None, None, None

        model10 = ATM_Model(enc_len=enc_len, dec_len=dec_len, epoch=1000, valid_size=199)
        # series = np.array(self.data.List_ATM_Series[tid])
        # print(series)
        print("TRAINING...")
        return model10.Create_and_Train_Model(series, "ATM_Models/" + str(tid))

    def FineTune_Model(self, tid, enc_len, dec_len,series):
        'Fine tune model'
        model10 = ATM_Model(enc_len=enc_len, dec_len=dec_len, epoch=1000, valid_size=199)
        # series = np.array(self.data.List_ATM_Series[tid])  # need to update this series for FineTune
        # print(series)
        print("Finetuning...")
        return model10.FineTune_Model(series, "ATM_Models/" + str(tid))

    def Model_predict(self, tid, enc_len, dec_len, serires_input, future, series):
        'Predict'

        # if tid not in self.data.List_ATM_Series:
        #     return None
        # Tạo model
        model10 = ATM_Model(enc_len=enc_len, dec_len=dec_len, epoch=1000, valid_size=199)
        # series = np.array(self.data.List_ATM_Series[tid])

        model_dir = "ATM_Models/" + str(tid) + "/checkpoints"

        # Kiểm tra xem trong đường dẫn đã có model trên hay chưa, nếu chưa có model train rồi thì tiến hành train
        if (not os.path.exists(model_dir)) or (len(os.listdir(model_dir)) == 0):
            print(f'Model for "{tid}" is not existed. Now training..')
            # print(f'Model for "{tid}" is not existed. Now training..')

            # Tạo và lưu model vào đường dẫn có tid
            self.Model_train(tid ,series, enc_len=enc_len, dec_len=dec_len)
            print(f'Training finished')
            print('Done')
        print('Now predicting..')

        predicter, _mean, _std = model10.Create_and_rLoad_Model(series, "ATM_Models/" + str(tid))
        # PREDICTION
        print("Predict with best epoch ", predicter.best_epoch)
        print("Epoch of Ler ", predicter.epochs)
        return model10._predict(predicter, _mean, _std, serires_input, future)