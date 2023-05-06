import pandas as pd
import time
import timeit
from math import log
from datetime import datetime, timedelta

def clean_data(input_file,output_file):
    df = pd.read_csv(input_file)
    df_ledger = pd.read_csv('ledger.csv')
    date_format1 = "%Y/%m/%d"
    date_format2 = "%m/%d/%y"
    index_data = 0
    for index, row in df_ledger.iterrows():
        date_ledger = datetime.strptime(df_ledger.iloc[index,2],date_format2).date()
        date_data = datetime.strptime(df.iloc[index,0],date_format1).date()
        while date_ledger != date_data :
            print('ledger:'+df_ledger.iloc[index,2]+" data:"+df.iloc[index,0])
            df.drop(index=index,inplace=True)
            df = df.reset_index(drop=True)
            date_data = datetime.strptime(df.iloc[index,0],date_format1).date()
            print('cleaned:'+df_ledger.iloc[index,2]+" "+df.iloc[index,0])
        if index == df_ledger.shape[0]-1 and index < df.shape[0]-1 :
            print('last:'+df_ledger.iloc[index,2]+" "+df.iloc[index,0])
            index_data = index + 1
    df.drop(df.index[index_data:],inplace=True)
    df.to_csv(output_file,index=False)

clean_data('hw4_data.csv','data_cleaned.csv')

# def clean_data(input_file,output_file):
#     df = pd.read_csv(input_file)
#     df_ledger = pd.read_csv('ledger.csv')
#     date_format = "%Y/%m/%d"
#     for index, row in df.iterrows():
#         date_str = row['Dates']
#         date = datetime.strptime(date_str, date_format).date()
#         if date.isoweekday() == 6 or date.isoweekday() == 7:
#             print('drop ' + date_str)
#             df.drop(index=index,inplace=True)
#
#     df = df.reset_index(drop=True)
#     df.to_csv(output_file,index=False)

