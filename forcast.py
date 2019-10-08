# import pandas as pd
# import numpy as np
# import json
# from pandas.io.json import json_normalize
# from datetime import datetime, timedelta,date
# import time
# import dateutil.parser
# from statsmodels.tsa.arima_model import ARIMA
# from pmdarima import auto_arima
# from pandas.tseries.offsets import DateOffset
# #from statsmodels.tsa.arima_model import ARIMA
# import statsmodels.api as sm

# def main(dataset,x_axis,y_axis,timestamp,start_time,end_time):
#     df = preprocessing(dataset,y_axis,timestamp)
    
#    # create_dump = generate_dataset(df,timestamp,start_time,end_time)
#     model = generate_dataset(df,timestamp,start_time,end_time,y_axis)
#     #print(df)
#     #print(df.columns)
#     print(model)

#     return model

# def preprocessing(df,y_axis,timestamp):
#     print("timestamp:",timestamp)
#     df = pd.DataFrame.from_records(df)
#     df.dropna(how='all',inplace=True)
#     df.fillna(method='ffill',inplace=True)
#     df = df[[timestamp,y_axis]]
#     df[timestamp] = pd.to_datetime(df[timestamp])
#     df.set_index(timestamp,inplace=True)

#     #df = df.to_json(orient='records') 
#     return df



# def auto_Arima(df,future_df,no_of_rows_df,no_of_rows_forcast,timestamp):
 
#     stepwise_fit = auto_arima(df, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
#                           start_P=0, seasonal=True, d=1, D=1, trace=True,
#                           error_action='ignore',  
#                           suppress_warnings=True,  
#                           stepwise=True) 
#     print(stepwise_fit.summary())
#     forcast = stepwise_fit.predict(n_periods=no_of_rows_forcast)
#     print("auto-arima",forcast)

#     future_df['Sales'] = forcast
#     col_names = df.columns
#     print(future_df.columns)
#     df = pd.concat([df,future_df])
#     df = pd.DataFrame(df)
#    # df = df.reset_index(inplace = True) 
#     return df

# def generate_dataset(df,timestamp,start_time,end_time,y_axis):
#     #print(type(start_time))

#     start_t = pd.to_datetime(start_time)
#     end_t = pd.to_datetime(end_time)
#     #model = arima_model(df,y_axis)
#     print(type(start_t))
#     future_df,no_of_rows_df,no_of_rows_forcast = timestamp_duration(df,timestamp,start_t,end_t,y_axis)
#     model_auto = auto_Arima(df,future_df,no_of_rows_df,no_of_rows_forcast,timestamp)
#     return model_auto
#     #return model


# def timestamp_duration(df,timestamp,start_time,end_time,y_axis):
    
#     if timestamp == "Week":
#         future_timestamp = ((end_time - start_time)/np.timedelta64(1, 'W'))
#         print(future_timestamp)
#         future_timestamp = int(future_timestamp)
#         future_dates = [df.index[-1] + DateOffset(weeks=i) for i in range(0,future_timestamp)]
 


#     elif timestamp == "Month":
#         future_timestamp = ((end_time - start_time)/np.timedelta64(1, 'M'))
#         future_timestamp = int(future_timestamp)
#         future_dates = [df.index[-1] + DateOffset(months=i) for i in range(0,future_timestamp)]

        
#     elif timestamp == "Year":
#         future_timestamp = ((end_time - start_time)/np.timedelta64(1, 'Y'))
#         future_timestamp = int(future_timestamp)
#         future_dates = [df.index[-1] + DateOffset(years=i) for i in range(0,future_timestamp)]

#     future_df = pd.DataFrame(index=future_dates[1:],columns=df.columns)
#     no_of_rows_forcast = future_df.shape[0]
#     no_of_rows_df = df.shape[0]
#     print(future_df)
#     return future_df,no_of_rows_df,no_of_rows_forcast

