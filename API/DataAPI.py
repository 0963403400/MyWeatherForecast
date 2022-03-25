import sys
# sys.path.insert(0, "/home/ds1/projects/tpbams/api/")
import pandas as pd
import numpy as np

from core.logger import Logger

from api.model import TPBAMSRepository #Call put from this to push data into DB's app
from api.model.entities import DailyTransAmount
from .Preproces import Preprocess
import ibm_db, ibm_db_dbi
from datetime import datetime, timedelta

from api.model import TPBAMSRepository
# Get new Data from DB2WareHouse then "preprocess" and put in to DB's app
# Get TXN Data from DB's app and convert to Times series
class DataAPIs:

  def __init__(self, cfg):
    #Connect to DB2
    self.dsn_host = cfg['host']
    self.dsn_port = cfg['port']
    self.dsn_user = cfg['username']
    self.dsn_pwd = cfg['password']
    self.dsn_driver = cfg['driver']
    self.dsn_dbname = cfg['database_name']
    self.dsn_protocol = cfg['protocol']
    
    self.transtable = cfg['transtable']
    self.datefield = cfg['datefield']
    self.atmcodefield = cfg['atmcodefield']

    try:
      self.ibm_db_conn = self.ConnectDB2()
    except:
      self.ibm_db_conn=None
      Logger.instance().error(
            f'Cant connect to DB2')
    self.List_ATMs=[] 
    
    try:
      self.List_ATMs = np.array(TPBAMSRepository().execute(f'''
                    select "CODE"
                    from "ATMS"
                    ''').fetchall())
      self.List_ATMs = list(np.reshape(self.List_ATMs, (-1)))
    except:
      print('Can not get ATM list')
    
    self.List_ATM_Series=self.readListATMs() # read from DBApp up to date( Today() )

  #Disconnect
  def dispose(self):
    if self.ibm_db_conn is not None:
      self.ibm_db_conn.close()
    

  #Connect
  def ConnectDB2(self):    
    'Create connection to DB2'    
    connection_string = 'DRIVER = {0}; DATABASE={1};HOSTNAME={2};PORT={3};PROTOCOL={4};UID={5};PWD={6};'.format(self.dsn_driver, self.dsn_dbname, self.dsn_host, self.dsn_port, self.dsn_protocol, self.dsn_user, self.dsn_pwd)
    con = ibm_db.connect(connection_string, "", "")
    return ibm_db_dbi.Connection(con)
    
  
  #Update new data
  def UpdateDBapp(self):
    if self.ibm_db_conn is None:
      return
    'get new data from DB2 where day>=today, preprocess --> insert to DB_app '
    curr_day = int(datetime.today().strftime('%Y%m%d'))
    print("Connected to DB2")
    Logger.instance().info(f'DataPreprocessing: Connected to Data Mart. Now processing...')

    lastDaySql = 'select max("DATE") as DAT from dailytransamounts where realamount != 0'
    lastUpdateDay = TPBAMSRepository().execute(lastDaySql).scalar()
    
    if lastUpdateDay is not None:
      lastUpdateDay = lastUpdateDay.replace('/', '')
      sql = 'Select * from {0} where {1} > {2} and {3} in ({4})'.format(self.transtable, self.datefield, lastUpdateDay, self.atmcodefield, ','.join(self.List_ATMs))
    else:
      sql = 'Select * from {0} where {1} in ({2})'.format(self.transtable, self.atmcodefield, ','.join(self.List_ATMs))

    print("Query from DB2: ",sql)

    'Get data from DB2 with sql querry --> insert into DB_app'
    df = pd.read_sql(sql, self.ibm_db_conn)
    if df is None:
      Logger.instance().error(f'Get data from Data Mart failed')
      return None
    Preproc = Preprocess(df)
    for TID in self.List_ATMs:
      print("TID: ", TID)
      Logger.instance().info(f'DataPreprocessing: ATM \'{TID}\'')
      tsdf = Preproc.FixError992(TID.astype(int))
      tsday=Preproc.MakeTimeSeries(tsdf, 'D')
      data = []
      for i in range(tsday.shape[0]):
        date = tsday.index.strftime('%Y/%m/%d')[i]
        y = tsday.iloc[i]['y']
        data.append({
          'ATMCODE': TID,
          'DATE': date,
          'REALAMOUNT': y
        })
      TPBAMSRepository().put(entity_name='DailyTransAmount', values=data)
    Logger.instance().info(f'DataPreprocessing: Done')

    return tsday

  ############Using this for test with CSV file###############
  def readATM(self, tid):
    path = "core/forecasts/DataUT/"
    atm_series = pd.read_csv(path + str(tid) + ".csv")
    atm_series=atm_series.y.values
    atm_series=pd.DataFrame(atm_series)
    Q1=atm_series.quantile(0.25).values[0]
    Q3=atm_series.quantile(0.75).values[0]
    Thresh = Q1- (Q3-Q1)*0.75
    if Thresh <0:
      Thresh = Q1
    atm_series[atm_series<Thresh]=Thresh
    atm_series =np.array(atm_series)
    atm_series=np.log(atm_series)
    return atm_series
    #Get ATM data from DBapp with tid

  def LoadATMTrans(self, tid):
    today = datetime.today()
    to_date = today - timedelta(days=1)
    from_date = to_date - timedelta(days=365*5)

    filtered_atm = TPBAMSRepository().execute(f'''
        select "DATE", "REALAMOUNT"
        from "DAILYTRANSAMOUNTS"
        where ("DATE" between :from_date
                          and :to_date)
            and "ATMCODE" = :tid
        order by "DATE" ASC
      ''',
      from_date=from_date.strftime('%Y/%m/%d'),
      to_date=to_date.strftime('%Y/%m/%d'),
      tid=tid).fetchall()
    
    if len(filtered_atm) < 1:
      return None

    last_date = datetime.strptime(filtered_atm[-1][0], '%Y/%m/%d')
    series_input = np.array(list(map(lambda x: x[1], filtered_atm)))

    while len(series_input) > 0 and series_input[-1] == 0:
      series_input = series_input[0:-1]

    if len(series_input) < 365 * 2:
      return None

    atm_series=pd.DataFrame(series_input)
    Q1=atm_series.quantile(0.25).values[0]
    Q3=atm_series.quantile(0.75).values[0]
    Thresh = Q1- (Q3-Q1)*0.75
    if Thresh <0:
      Thresh = Q1
    print("Q1: ", Q1," Q3: ", Q3," Thresh: ", Thresh)
    Thresh=2100000
    atm_series[atm_series<Thresh]=Thresh
    atm_series =np.array(atm_series)
    return atm_series

  def readListATMs(self):
    series={}
    for tid in self.List_ATMs:
      res = self.LoadATMTrans(tid) 
      if res is not None:
        series[tid] = res
    return series