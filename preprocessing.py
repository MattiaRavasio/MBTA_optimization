import numpy as np
import random
import pysindy as ps
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
pd.options.mode.chained_assignment = None

df = pd.read_csv("MBTA_data.csv")

data = df[(df.day_type_name == "weekday") & (df.season == "Fall 2019")]

data.stop_name = np.where(data.stop_id == 'place-stpul', 'St Pul Street', data.stop_name)

time_windows_order = ["VERY_EARLY_MORNING", "EARLY_AM", "AM_PEAK", "MIDDAY_SCHOOL",
                     "MIDDAY_BASE", "PM_PEAK", "EVENING", "LATE_EVENING", "NIGHT"]

#### No offpeak for the blue line
Blue1_dataset = data[(data.route_name == "Blue Line") & (data.direction_id == 0)].drop(['ObjectId', 'mode', 'season', 'route_id', 
       'day_type_id', 'time_period_id',  'stop_id', 'total_ons', 'total_offs',
       'number_service_days', 'average_ons', 'average_offs', ], axis=1)

Blue1_dataset = Blue1_dataset.pivot(index='stop_name', values='average_flow', columns = "time_period_name").reset_index()[time_windows_order]

Blue2_dataset = data[(data.route_name == "Blue Line") & (data.direction_id == 1)].drop(['ObjectId', 'mode', 'season', 'route_id', 
       'day_type_id', 'time_period_id',  'stop_id', 'total_ons', 'total_offs',
       'number_service_days', 'average_ons', 'average_offs', ], axis=1)

Blue2_dataset = Blue2_dataset.pivot(index='stop_name', values='average_flow', columns = "time_period_name").reset_index()[time_windows_order]

Orange1_dataset = data[ (data.route_name == "Orange Line") & (data.direction_id == 0)].drop(['ObjectId', 'mode', 'season', 'route_id', 
       'day_type_id', 'time_period_id',  'stop_id', 'total_ons', 'total_offs',
       'number_service_days', 'average_ons', 'average_offs', ], axis=1)

Orange1_dataset = Orange1_dataset.pivot(index='stop_name', values='average_flow', columns = "time_period_name").reset_index()[time_windows_order]

Orange2_dataset = data[(data.route_name == "Orange Line") & (data.direction_id == 1)].drop(['ObjectId', 'mode', 'season', 'route_id', 
       'day_type_id', 'time_period_id',  'stop_id', 'total_ons', 'total_offs',
       'number_service_days', 'average_ons', 'average_offs', ], axis=1)

Orange2_dataset = Orange2_dataset.pivot(index='stop_name', values='average_flow', columns = "time_period_name").reset_index()[time_windows_order]

Red1_dataset = data[(data.route_name == "Red Line") & (data.direction_id == 1)].drop(['ObjectId', 'mode', 'season', 'route_id', 
       'day_type_id', 'time_period_id',  'stop_id', 'total_ons', 'total_offs',
       'number_service_days', 'average_ons', 'average_offs', ], axis=1)

Red1_dataset = Red1_dataset.pivot(index='stop_name', values='average_flow', columns = "time_period_name").reset_index()[time_windows_order]

Red2_dataset = data[(data.route_name == "Red Line") & (data.direction_id == 0)].drop(['ObjectId', 'mode', 'season', 'route_id', 
       'day_type_id', 'time_period_id',  'stop_id', 'total_ons', 'total_offs',
       'number_service_days', 'average_ons', 'average_offs', ], axis=1)

Red2_dataset = Red2_dataset.pivot(index='stop_name', values='average_flow', columns = "time_period_name").reset_index()[time_windows_order]

Green1_dataset = data[(data.route_name == "Green Line") & (data.direction_id == 1)].drop(['ObjectId', 'mode', 'season', 'route_id', 
       'day_type_id', 'time_period_id',  'stop_id', 'total_ons', 'total_offs',
       'number_service_days', 'average_ons', 'average_offs', ], axis=1)

Green1_dataset = Green1_dataset.pivot(index='stop_name', values='average_flow', columns = "time_period_name").reset_index()[time_windows_order]

Green2_dataset = data[(data.route_name == "Green Line") & (data.direction_id == 0)].drop(['ObjectId', 'mode', 'season', 'route_id',
         'day_type_id', 'time_period_id',  'stop_id', 'total_ons', 'total_offs',
         'number_service_days', 'average_ons', 'average_offs', ], axis=1)   

Green2_dataset = Green2_dataset.pivot(index='stop_name', values='average_flow', columns = "time_period_name").reset_index()[time_windows_order]

Blue1_dataset.columns.name = None
Blue2_dataset.columns.name = None

Orange1_dataset.columns.name = None
Orange2_dataset.columns.name = None

Red1_dataset.columns.name = None
Red2_dataset.columns.name = None

Green1_dataset.columns.name = None
Green2_dataset.columns.name = None

Blue1_dataset.to_csv("processed_data/Blue1_dataset.csv", index=False)
Blue2_dataset.to_csv("processed_data/Blue2_dataset.csv", index=False)

Orange1_dataset.to_csv("processed_data/Orange1_dataset.csv", index=False)
Orange2_dataset.to_csv("processed_data/Orange2_dataset.csv", index=False)

Red1_dataset.to_csv("processed_data/Red1_dataset.csv", index=False)
Red2_dataset.to_csv("processed_data/Red2_dataset.csv", index=False)

Green1_dataset.to_csv("processed_data/Green1_dataset.csv", index=False)
Green2_dataset.to_csv("processed_data/Green2_dataset.csv", index=False)

Blue1_std = np.random.uniform(0.02, 0.06,Blue1_dataset.shape) * Blue1_dataset
Blue2_std = np.random.uniform(0.02, 0.06,Blue2_dataset.shape) * Blue2_dataset

Orange1_std = np.random.uniform(0.02, 0.06,Orange1_dataset.shape) * Orange1_dataset
Orange2_std = np.random.uniform(0.02, 0.06,Orange2_dataset.shape) * Orange2_dataset

Red1_std = np.random.uniform(0.02, 0.06,Red1_dataset.shape) * Red1_dataset
Red2_std = np.random.uniform(0.02, 0.06,Red2_dataset.shape) * Red2_dataset

Green1_std = np.random.uniform(0.02, 0.06,Green1_dataset.shape) * Green1_dataset
Green2_std = np.random.uniform(0.02, 0.06,Green2_dataset.shape) * Green2_dataset

Blue1_std.astype(int).to_csv("processed_data/Blue1_std.csv", index=False)
Blue2_std.astype(int).to_csv("processed_data/Blue2_std.csv", index=False)

Orange1_std.astype(int).to_csv("processed_data/Orange1_std.csv", index=False)
Orange2_std.astype(int).to_csv("processed_data/Orange2_std.csv", index=False)

Red1_std.astype(int).to_csv("processed_data/Red1_std.csv", index=False)
Red2_std.astype(int).to_csv("processed_data/Red2_std.csv", index=False)

Green1_std.astype(int).to_csv("processed_data/Green1_std.csv", index=False)
Green2_std.astype(int).to_csv("processed_data/Green2_std.csv", index=False)