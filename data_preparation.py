import csv
import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG

def get_data(train):
    " Extract data from input csv file and create train and test data"
      "Return values as per input is train or not"
        
    temps = []

    with open(data_path_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            temps.append(float(row[1]))

    series = np.array(temps)
    
    plt.plot(series, label='Temperature', color='green')
    plt.legend(loc='best')
    plt.xlabel('Day')
    plt.ylabel('Mean Temperature')
    plt.show()

    split_time = int(len(series)* CONFIG.train_test_split)
    x_train = series[:split_time] 
    x_test = series[split_time:] 
    
    if train:
        return x_train
    else:
        return x_test,split_time,series
