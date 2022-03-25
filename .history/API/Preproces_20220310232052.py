import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb
# from matplotlib.pyplot import figure
import numpy as np
# import statsmodels as sm
# from statsmodels.tsa.stattools import adfuller

class Preprocess:
    #Get data from DB

    def __init__(self, df):
        self.TS = self.ConvertDataFrameToTimeSerires(df)
        self.TS = self.removeNormalErr()
        "Ex: Proc = Preprocess(_dataframe)"
        "tsdf1=Proc.FixError992(tid)"
        "tsday=Proc.MakeTimeSeries(tsdf1, 'D')"

    #Describe the ATMs Dataset
    def Description(self):
        return self.TS.describe()
    #Convert Datafram to Time series
    def ConvertDataFrameToTimeSerires(self, df):
        atm=df.sort_values(by=["NGAY","GIO"], ascending=True)
        norm_atm = atm
        norm_atm["ds"]= pd.to_datetime(norm_atm["NGAY"].astype(int).astype(str) +" "+ norm_atm["GIO"].astype(int).astype(str).apply(lambda x: x.zfill(6)))
        TS80=pd.DataFrame()
        TS80['ds']=norm_atm["ds"]
        TS80['y']=norm_atm["REQAMT"] #Req Amount
        TS80['RESP']=norm_atm["RESP"] #Resp code
        TS80['CardNumber']=norm_atm['HPAN']
        TS80['MA_ATM']=norm_atm['MA_ATM'].astype(int)
        TS80S = TS80.set_index('ds')
        return TS80S
    # Remove normal Er
    def removeNormalErr(self):
        self.TS = self.TS[self.TS["RESP"].isin([992, -1])]
        return self.TS
  
    #Create time series for each ATM: opt = 'D' / 'W' / 'Y'
    def MakeTimeSeries_From_TID(self,tid, opt):
        return self.TS[self.TS["MA_ATM"]==tid].groupby(pd.Grouper(freq=opt)).sum()

    #Make time series from df of an atm
    def MakeTimeSeries(self, atmi, opt):
        return atmi.groupby(pd.Grouper(freq=opt)).sum()

    #Get list std of ATMs corresponding with TID tid
    def GetSTDsValues(self, listtid):
        std_DF = pd.DataFrame(columns=["TID", "STD"])
        std_DF.set_index(["TID"])
        for tid in listtid:
            di=self.MakeTimeSeries_From_TID(tid, 'D')
            std_DF.loc[tid]=di.y.std()
        return std_DF

  #Get list atm in minvalue to maxvalues
    def getListATM_Std(self, DFTotal_TXN):    
        print("Tổng quan về bộ dữ liệu theo tổng lượng tiền giao dịch: ")
        print(DFTotal_TXN.describe())
        Q1, Q2, Q3 =DFTotal_TXN["STD"].quantile([0.25, 0.50, 0.75])
        print("Các giá trị Q: ",Q1,Q2,Q3)
        boxwide=Q3-Q1
        value_min=Q1-1.5*boxwide
        value_max=Q3+1.5*boxwide
        Listmid=[]
        for tid in DFTotal_TXN.index.to_list():
            if DFTotal_TXN["STD"].loc[tid]>=value_min and DFTotal_TXN["STD"].loc[tid] <= value_max:
                Listmid.append(tid)
        print("Các giá trị trong khoảng min đến max: ")
        print(len(Listmid))
        print(Listmid)
        return Listmid
    #Fix 992 Er
    ##keep value and normalize outlier --> wrisker
    def FixError992(self, tid):
        TS=self.TS[self.TS["MA_ATM"]==tid]
        RS=TS.copy()
        i=0
        Q1, _, Q3 =TS["y"].quantile([0.25, 0.50, 0.75])
        maxTxn=Q3+1.0*(Q3-Q1)
        print(maxTxn)

        while i < TS['y'].count()-1:
            if TS['RESP'].iloc[i]==992:
                RS.y.iloc[i]=TS.y.iloc[i]*1.2
                if (RS.y.iloc[i]>maxTxn):
                    RS.y.iloc[i]=maxTxn  

                cardNum = TS["CardNumber"].iloc[i]
                
                j = i + 1

                # reset 0 neu mot the rut 992 lien tiep k lan 
                while j < TS['y'].count()-1 and TS['RESP'].iloc[j]==992 and TS["CardNumber"].iloc[j] == cardNum:
                    RS.y.iloc[j]=0
                    j = j + 1
                
                i = j
            else:         
                i=i+1

        return RS

    #Add week day
    def AddWeekDay(self, TSD):
        TSD['Weekday']=TSD.index
        TSD['WD']=TSD['Weekday'].dt.day_name()
        TSD.pop('Weekday')
        return TSD
    #Create ts days of week
    def MakeTimeSeries_DayOfWeek(self, TSi, dayofweek):
        TSi=TSi.groupby(pd.Grouper(freq='D')).sum()
        TSi = self.AddWeekDay(TSi)
        return TSi[TSi['WD']==dayofweek]