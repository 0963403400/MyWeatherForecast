import sys
from .deepseries.wave2wave import *
from .deepseries.train import Learner
from .deepseries.data import Value, create_seq2seq_data_loader, forward_split
from .deepseries.nn.loss import RMSE, MSE, MAPE, SMAPE, MAPES
from .deepseries import functional as F
import numpy as np
import torch
from torch.optim import Adam

# from core.logger import Logger


# Class build atms model
class ATM_Model:

  def __init__(self, enc_len=360, dec_len=1, epoch=1000, batch_size=16, lr=0.001, valid_size=99, test_size=0):
    self.batch_size = 16
    self.enc_len = enc_len
    self.dec_len = dec_len
    self.epoch = epoch
    self.lr = lr
    self.valid_size = valid_size
    self.test_size = test_size

  def Create_and_Train_Model(self, series, model_save_path, verbow =0, early_stopping=True):
    'Init and train model'
    model = self.Create_model(model_save_path, verbow=verbow, loss_function=RMSE())
    return self.Fit_model(series, model)

  def FineTune_Model(self, series, model_save_path, verbow =0, early_stopping=True):
    'FineTune model: Reload and train model with new data'
    model = self.Create_model(model_save_path, verbow=verbow, loss_function=RMSE())
    wave_learner, _, _ = self.re_load_model(series,model)
    return self.Fit_model(series, wave_learner)

  def Create_and_rLoad_Model(self, series, model_save_path, verbow =0, early_stopping=True):
    'Reload model'
    model = self.Create_model(model_save_path, verbow=verbow, loss_function=RMSE())
    return self.re_load_model(series, model)
  
    
  def Create_model(self, model_save_path, verbow, num_layers = 12, loss_function=RMSE()):
    'Create model'    
    wave = Wave2Wave(target_size=1, num_layers=num_layers, num_blocks=2, dropout=0.01, loss_fn=loss_function)
    wave.cpu()
    opt = torch.optim.Adam(wave.parameters(), lr=self.lr)
    wave_learner = Learner(wave, opt, root_dir=model_save_path, verbose=verbow)
    return wave_learner  

  def Fit_model(self, series, wave_learner, early_stopping=True):
    'Create and train a model for predict ATM cashflow with input is: series'
    'This function will return a model for predict, mean and standard diviation'    
    #Reshape affter compute mean and std
    series = series.reshape(1, 1, -1)
    train_idx, valid_idx = forward_split(np.arange(series.shape[2]), enc_len=self.enc_len, valid_size=self.valid_size + self.test_size)
    test_idx=[]
    if self.test_size>0:
      valid_idx, test_idx = forward_split(valid_idx, self.enc_len, self.test_size)

    # mask test, will not be used for calculating mean/std.
    mask = np.zeros_like(series).astype(bool)
    if len(test_idx)>0:
      mask[:, :, test_idx] = False
    series, mu, std = F.normalize(series, axis=2, fillna=True, mask=mask)

    # wave2wave train
    train_dl = create_seq2seq_data_loader(series[:, :, train_idx], self.enc_len, self.dec_len, sampling_rate=1,
                                          batch_size=self.batch_size, seq_last=True, device='cpu')
    valid_dl = create_seq2seq_data_loader(series[:, :, valid_idx], self.enc_len, self.dec_len,
                                          batch_size=self.batch_size, seq_last=True, device='cpu')

    wave = Wave2Wave(target_size=1, num_layers=12, num_blocks=2, dropout=0.01, loss_fn=RMSE())
    wave.cpu()
    wave_learner.fit(max_epochs=self.epoch, train_dl=train_dl, valid_dl=valid_dl, early_stopping=early_stopping, patient=32)
    #wave_learner.load(wave_learner.best_epoch)
    wave_learner.remove_useless_model()
    wave_learner.loadBestModel()
    print("Best Epoch: ", wave_learner.best_epoch)
    print("Epoch : ", wave_learner.epochs)
    #Log MAPE and SMAPE 00520270
    try:
      period = 3
      predict_nums = int((len(valid_idx)-self.enc_len)/period)    
      real_values = np.zeros(predict_nums * period, float)
      pred_values = np.zeros(predict_nums * period, float)
      for localtion in range(predict_nums):
        pred = self._predict(wave_learner, mu, std, series[ : , : , valid_idx[localtion*period : localtion*period + self.enc_len]].reshape(-1), 7)
        real_values[localtion*period : localtion*period + 7] = series[: , : , valid_idx[localtion*period+self.enc_len : localtion*period + self.enc_len + 7]].reshape(-1)
        pred_values[localtion*period : localtion *period +7] =pred
      mape_value = self.mean_absolute_percentage_error(real_values, pred_values)
      smape_value = self.symmetric_mean_absolute_percentage_error(real_values, pred_values)
      print(f'Model has been trained with MAPE: {mape_value} and SMAPE: {smape_value}')
    except:
      print(f'Model has been trained but cant calculate MAPE and SMAPE')

    return wave_learner, mu, std

  def re_load_model(self, series, wave_learner, early_stopping=True):
    'Creload'
    #Reshape affter compute mean and std
    series = series.reshape(1, 1, -1)
    _, valid_idx = forward_split(np.arange(series.shape[2]), enc_len=self.enc_len, valid_size=self.valid_size + self.test_size)
    test_idx=[]
    if self.test_size>0:
      valid_idx, test_idx = forward_split(valid_idx, self.enc_len, self.test_size)

    # mask test, will not be used for calculating mean/std.
    mask = np.zeros_like(series).astype(bool)
    if len(test_idx)>0:
      mask[:, :, test_idx] = False
    series, mu, std = F.normalize(series, axis=2, fillna=True, mask=mask)
    wave = Wave2Wave(target_size=1, num_layers=12, num_blocks=2, dropout=0.01, loss_fn=RMSE())
    wave.cpu()
    #wave_learner.load(wave_learner.best_epoch)
    wave_learner.loadBestModel()
    return wave_learner, mu, std

  def ATMs_Predict(self,ATMs_Model, mu, std, series_input=None, future_step=7):
    series_input= series_input.reshape(1,1,-1)
    series_input=(series_input-mu)/(std + 1e-6)
    rs=ATMs_Model.model.predict(torch.tensor(series_input).float().cpu(), future_step).cpu().numpy().reshape(-1,1)
    #xxx= np.array([mu for _ in range(future_step)])
    #return xxx.reshape(-1)
    return (rs*std+mu).reshape(-1)



  ###Evaluate
  def Create_Index(self, series):
    series = series.reshape(1, 1, -1)
    train_idx, valid_idx = forward_split(np.arange(series.shape[2]), enc_len=self.enc_len, valid_size=self.valid_size+self.test_size)
    valid_idx, test_idx = forward_split(valid_idx, self.enc_len, self.test_size)
    return train_idx, valid_idx, test_idx

  def _predict(self, predicter, _mean, _std, series_input, future):    
    return self.ATMs_Predict(predicter, _mean, _std, series_input, future)


  ####Metrics
  def MAPE(self, actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred)/(actual+1e-10))) * 100

  ################-------------------METRICS--------------##########
  def mean_square_error(self, y_true, y_pred):
    y_true = np.asmatrix(y_true).reshape(-1)
    y_pred = np.asmatrix(y_pred).reshape(-1)

    return np.square(np.subtract(y_true, y_pred)).mean()

  def root_mean_square_error(self, y_true, y_pred):

    return self.mean_square_error(y_true, y_pred)**0.5


  def mean_absolute_percentage_error(self, y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(np.where(y_true == 0)[0]) > 0:
        return np.inf
    else:
        return np.mean(np.abs((y_true - y_pred) / (y_true+0) )) * 100

  def symmetric_mean_absolute_percentage_error(self, y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.mean(np.abs((y_true - y_pred) / (( np.abs(y_true) + np.abs(y_pred) )/2) ))*100

  def mean_absolute_error(self, y_true, y_pred):
      
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.mean(np.abs(y_true - y_pred))

  # U of Theil Statistics
  def u_theil(self, y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred[0:(len(y_pred) - 1)], y_pred[1:(len(y_pred))])).sum()

    return error_sup / (error_inf+1e-10)


  def average_relative_variance(self, y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred, mean)).sum()

    return error_sup / error_inf


  def index_agreement(self, y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.abs(np.subtract(y_true, y_pred))).sum()

    error_inf = np.abs(np.subtract(y_pred, mean)) + np.abs(np.subtract(y_true, mean))
    error_inf = np.square(error_inf).sum()

    return 1 - (error_sup / error_inf)


  def prediction_of_change_in_direction(self, y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    true_sub = np.subtract(y_true[0:(len(y_true) - 1)], y_true[1:(len(y_true))])
    pred_sub = np.subtract(y_pred[0:(len(y_pred) - 1)], y_pred[1:(len(y_pred))])

    mult = true_sub * pred_sub
    result = 0
    for m in mult:
        if m > 0:
            result = result + 1
    return (100 * (result / len(y_true)))

  def gerenerate_metric_results(self, y_true, y_pred):
    return {'MSE': self.mean_square_error(y_true, y_pred), 
              'RMSE':self.root_mean_square_error(y_true, y_pred),
              'MAPE': self.mean_absolute_percentage_error(y_true, y_pred),
              'SMAPE':self.symmetric_mean_absolute_percentage_error(y_true, y_pred),
              'MAE': self.mean_absolute_error(y_true, y_pred),
              'theil': self.u_theil(y_true, y_pred),
              'ARV': self.average_relative_variance(y_true, y_pred),
              'IA': self.index_agreement(y_true, y_pred),
              'POCID': self.prediction_of_change_in_direction(y_true, y_pred)}