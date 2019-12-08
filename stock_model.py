#!/usr/bin/env python
# coding: utf-8

# In[1]:


### import modules
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def analyze_raw(data_df, ticker = 'SPY'):
    analyze_df = pd.DataFrame(columns = ['ticker','start','end','days','low','high','now','annual_return', 'total_return'])
    analyze_df.at[0,'ticker'] = ticker
    analyze_df.at[0,'start'] = data_df.index[0]
    analyze_df.at[0,'end'] = data_df.index[-1]
    analyze_df.at[0,'days'] = len(data_df)
    analyze_df.at[0,'low'] = data_df.close.min()
    analyze_df.at[0,'high'] = data_df.close.max()
    analyze_df.at[0,'now'] = data_df.at[data_df.index[-1], 'close']
    analyze_df.at[0,'annual_return'] = ((data_df.at[data_df.index[-1], 'close'] / data_df.at[data_df.index[0], 'open']) ** (1 / len(data_df))) ** 252 -1
    analyze_df.at[0,'total_return'] = data_df.at[data_df.index[-1], 'close'] / data_df.at[data_df.index[0], 'open']
                                        
    return analyze_df

def plot_series(data_df, target = 'return'):
    pd.Series(data_df[target]).plot()
    plt.show()

def plot_hist(data_df, target = 'return', bins = 50):
    pd.Series(data_df[target]).hist(bins = bins)
    plt.show()
    
def plot_auto(data_df, target = 'return', lag = 100, ran = 0.1):
    pd.plotting.autocorrelation_plot(pd.Series(data_df[target]))
    plt.axis([0,lag,-ran,ran])
    plt.show()
    
def plot_all(data_df, target = 'return', bins = 50, lag = 100, ran = 0.1):
    #plt.figure(figsize=(15, 4))
    #plt.subplot(131)
    plot_series(data_df, target = target)
    #plt.subplot(132)
    plot_hist(data_df, target = target, bins = bins)
    #plt.subplot(133)
    plot_auto(data_df, target = target, lag = lag, ran = ran)


# In[3]:


def data_get(ticker = 'SPY', start = '1990-01-01', end = '2019-11-15'):
    data_df = yf.download(ticker)
    data_df = data_df.rename(columns={'Open':'open','Close':'close', 'High':'high', 'Low':'low'})
    data_df = data_df[['open','high','low','close']]
    data_df = data_df[start : end]
    data_df = data_df.dropna()
    
    return data_df

def data_add_return(data_df):
    data_df['return'] = 0.0
    for i in range(1,len(data_df)):
        data_df.at[data_df.index[i],'return'] = np.log(data_df.at[data_df.index[i],'close'] /data_df.at[data_df.index[i-1],'close'])
    data_df = data_df
    
    return data_df

def data_add_direction(data_df):
    data_df['direction'] = 0
    for i in range(1,len(data_df)):
        if data_df.at[data_df.index[i],'close']>data_df.at[data_df.index[i-1],'close']:
            data_df.at[data_df.index[i],'direction'] = 1

    return data_df

def data_ta(data_df):
    import ta
    # RSI
    data_df['rsi'] = ta.momentum.rsi(data_df['close'],n=14)
    data_df['rsi_singal'] = [ 1 if x>=70 else -1 if x<=30 else 0 for x in data_df['rsi'] ]
    #data_df['rrt'] = data_df['close'].pct_change()
    data_df['rrt'] = np.log(data_df['close']).diff()

    # MACD setup
    data_df['macd'] = ta.trend.macd(data_df['close'], n_fast=12, n_slow=26, fillna=False)
    data_df['macd_xover']=data_df['macd'] - data_df['macd'].ewm(span=9).mean()
    data_df['macd_xover_signal'] = [ 1 if x>0 else -1 if x<0 else 0 for x in data_df['macd_xover'] ]
    data_df['macd_signal'] = (np.sign(data_df['macd_xover_signal'] - data_df['macd_xover_signal'].shift(1)))

    # ADX setup
    data_df['adx'] = ta.trend.adx(data_df['high'], data_df['low'], data_df['close'], n=14).values

    # Bollinger Band Moving average setup
    data_df['bband_up'] = ta.volatility.bollinger_hband(data_df['rrt'], n=20, ndev=0.5)
    data_df['bband_dn'] = ta.volatility.bollinger_lband(data_df['rrt'], n=20, ndev=0.5)
    data_df['ma'] = ta.volatility.bollinger_mavg(data_df['rrt'], n=20)
    data_df['ma'] = data_df['ma'].shift(1)
    data_df = data_df.replace([np.inf, -np.inf], np.nan)

    # shift rate-of-return 1-period earlier for next period prediction
    ystd = data_df['rrt'].std()
    data_df['rrt']=data_df['rrt'].shift(-1)
    data_df.dropna(inplace=True)
    
    return data_df

def data_split(data_df, test_size = 0.3):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(data_df.loc[:, (data_df.columns != 'return') * (data_df.columns != 'direction') * (data_df.columns != 'price')],data_df[['return','direction', 'price']],test_size=test_size,shuffle=False)
    
    return x_train, x_test, y_train, y_test

def data_normalize(x_train, x_test):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, x_test

def data_complie(x_train, x_test):
    x_np = np.vstack((x_train, x_test))
    
    return x_np

def data_to_model(x_np, lag = 50, test_size = 0.3, target = 'return'):
    train_len = int(x_np.shape[0] * (1-test_size))
    test_len = x_np.shape[0] - train_len
    x_train = np.zeros((train_len-lag, x_np.shape[1]*lag))
    x_test = np.zeros((test_len, x_np.shape[1]*lag))
    for i in range(lag,train_len):
        if target == 'return':
            x_train[i-lag] = x_np[i-lag:i].reshape(x_np.shape[1]*lag) / (x_np[i-1, 3] + 1e-06)
        else:
            x_train[i-lag] = x_np[i-lag:i].reshape(x_np.shape[1]*lag)
    for i in range(train_len,test_len+train_len):
        if target == 'return':
            x_test[i-train_len] = x_np[i-lag:i].reshape(x_np.shape[1]*lag) / (x_np[i-1, 3] + 1e-06)
        else:
            x_test[i-train_len] = x_np[i-lag:i].reshape(x_np.shape[1]*lag)
            
    return x_train, x_test

def data_to_cnn(x_np, lag = 50, test_size = 0.3, target = 'return'):
    train_len = int(x_np.shape[0] * (1-test_size))
    test_len = x_np.shape[0] - train_len
    x_np=x_np.reshape((x_np.shape[0],x_np.shape[1],1))
    x_train = np.zeros((train_len-lag, lag, x_np.shape[1], 1))
    x_test = np.zeros((test_len, lag, x_np.shape[1], 1))
    for i in range(lag,train_len):
        if target == 'return':
            x_train[i-lag] = x_np[i-lag:i] / (x_np[i-1, 3] + 1e-06)
        else:
            x_train[i-lag] = x_np[i-lag:i]
    for i in range(train_len,test_len+train_len):
        if target == 'return':
            x_test[i-train_len] = x_np[i-lag:i] / (x_np[i-1, 3] + 1e-06)
        else:
            x_test[i-train_len] = x_np[i-lag:i]
    
    return x_train, x_test

def data_to_rnn(x_np, lag = 50, test_size = 0.3, target = 'return'):
    x_train, x_test = data_to_cnn(x_np, lag = lag, test_size = test_size, target = target)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])    
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
    
    return x_train, x_test

def data_y_cut(y_train, lag = 50):
    y_train = y_train[lag:]
    
    return y_train


# In[4]:


def model_build_sk(model_name_string, task_type = 'classification', model_para = ''):
    if task_type == 'classification':
        if model_name_string == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            model = eval('RandomForestClassifier(' + model_para +')')
        elif model_name_string == 'lr':
            from sklearn.linear_model import LogisticRegression
            model = eval('LogisticRegression(' + model_para +')')
        elif model_name_string == 'adb':
            from sklearn.ensemble import AdaBoostClassifier
            model = eval('AdaBoostClassifier(' + model_para +')')
        elif model_name_string == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            model = eval('KNeighborsClassifier(' + model_para +')')
    else:
        if model_name_string == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model = eval('RandomForestRegressor(' + model_para +')')
        elif model_name_string == 'knn':
            from sklearn.neighbors import KNeighborsRegressor
            model = eval('KNeighborsRegressor(' + model_para +')')
        elif model_name_string == 'adb':
            from sklearn.ensemble import AdaBoostRegressor
            model = eval('AdaBoostRegressor(' + model_para +')')
        elif model_name_string == 'lasso':
            from sklearn.linear_model import Lasso
            model = eval('Lasso(' + model_para +')')
        elif model_name_string == 'ridge':
            from sklearn.linear_model import Ridge
            model = eval('Ridge(' + model_para +')')     
    
    return model

def model_build_mlp(x_train, task_type = 'classification', number_layer = 3, width = 50, number_drop_out = 0.2, optimizer = 'adam'):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import BatchNormalization
    model = Sequential()
    if number_layer == 1:
        model.add(Dense(units = width, input_shape = (x_train.shape[1], )))
    else:
        model.add(Dense(units = width, input_shape = (x_train.shape[1], )))
        for i in range(number_layer-1):
            model.add(BatchNormalization())
            model.add(Dense(units = width))
            model.add(Dropout(number_drop_out))
        model.add(BatchNormalization())
        model.add(Dense(units = width))
        model.add(Dropout(number_drop_out))
    if task_type == 'classification':
        model.add(BatchNormalization())
        model.add(Dense(units = 2, activation='softmax'))
        model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    else:
        model.add(BatchNormalization())
        model.add(Dense(units = 1, activation='relu'))
        model.compile(optimizer = optimizer, loss = 'mean_squared_error')
        
    return model

def model_build_lstm(x_train, task_type = 'classification', number_layer = 3, width = 50, number_drop_out = 0.2, optimizer = 'adam'):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import BatchNormalization
    model = Sequential()
    if number_layer == 1:
        model.add(LSTM(units = width, input_shape = (x_train.shape[1], x_train.shape[2])))
    elif number_layer == 2:
        model.add(LSTM(units = width, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(units = width, dropout=number_drop_out))
    else:
        model.add(LSTM(units = width, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
        for i in range(number_layer-2):
            model.add(BatchNormalization())
            model.add(LSTM(units = width, dropout=number_drop_out, return_sequences = True))
        model.add(BatchNormalization())
        model.add(LSTM(units = width, dropout=number_drop_out))
    if task_type == 'classification':
        model.add(BatchNormalization())
        model.add(Dense(units = 2, activation='softmax'))
        model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    else:
        model.add(BatchNormalization())
        model.add(Dense(units = 1, activation='relu'))
        model.compile(optimizer = optimizer, loss = 'mean_squared_error')
        
    return model

def model_build_cnn(x_train, task_type = 'classification', number_layer = 3, width = 50, number_drop_out = 0.2, optimizer = 'adam'):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import BatchNormalization
    model = Sequential()
    if number_layer == 1:
        model.add(Conv2D(width, (3, 3), input_shape=x_train.shape[1:]))
    elif number_layer == 2:
        model.add(Conv2D(width, (3, 3), padding = 'same', input_shape=x_train.shape[1:]))
        model.add(BatchNormalization())
        model.add(Dropout(number_drop_out))
        model.add(Conv2D(width, (3, 3), padding = 'same'))
    else:
        model.add(Conv2D(width, (3, 3), padding = 'same', input_shape=x_train.shape[1:]))
        for i in range(number_layer-1):
            model.add(BatchNormalization())
            model.add(Dropout(number_drop_out))
            model.add(Conv2D(width, (3, 3), padding = 'same'))
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    if task_type == 'classification':
        model.add(Dense(units = 2, activation='softmax'))
        model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    else:
        model.add(Dense(units = 1, activation='relu'))
        model.compile(optimizer = optimizer, loss = 'mean_squared_error')
        
    return model

def model_train_sk(model, x_train, y_train):
    model.fit(x_train, y_train)
    
    return model

def model_train_tf(model, x_train, y_train, epoch = 50, batch_size = 64):
    history = model.fit(x_train, y_train, epochs = epoch, batch_size = batch_size)
    
    return history, model


# In[5]:


def score_model(model, x_test, y_test, model_type = 'sk', task_type = 'classification'):
    if model_type == 'sk':
        return model.score(x_test, y_test)
    else:
        if task_type == 'classification':
            return model.evaluate(x_test, y_test)[1]
        else:
            return model.evaluate(x_test, y_test)

def score_auc(model, x_test, y_test, model_type = 'sk'):
    from sklearn.metrics import roc_auc_score
    if model_type == 'sk':
        score = model.predict_proba(x_test)[:,1]
    else:
        score = model.predict(x_test)[:,1]
        
    return roc_auc_score(y_test, score)

def score_r2(model, x_test, y_test, x_test_close_original, target):
    from sklearn.metrics import r2_score
    if target == 'price':
        price_prediction = model.predict(x_test).ravel()
        price_real = y_test.ravel()
    else:
        prediction = model.predict(x_test)
        price_prediction = []
        price_real = []
        for i in range(1, len(y_test)):
            price_prediction.append(x_test_close_original[i] * np.exp(prediction[i]))
            price_real.append(x_test_close_original[i] * np.exp(y_test[i]))
        
    return r2_score(price_prediction, price_real)

def score_r2_base(y_close, drift = 0):
    from sklearn.metrics import r2_score
    prediction = np.array(y_close)[:-1]
    prediction *= np.exp(drift)
           
    return r2_score(y_close[1:], prediction)

def score_calculate_drift(y_close, lag = 100000):
    y_close = np.array(y_close)
    if lag < len(y_close):
        y_close = y_close[-lag:]
    drift = (np.log(y_close[-1]) - np.log(y_close[0])) /lag
    
    return drift

def score_plot_roc(model, x_test, y_test, model_type = 'sk'):
    from sklearn.metrics import roc_curve
    if model_type == 'sk':
        score = model.predict_proba(x_test)[:,1]
    else:
        score = model.predict(x_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, score)
    plt.plot(fpr, tpr, label='model')
    plt.plot(np.arange(0, 1.01, 0.01), np.arange(0, 1.01, 0.01), linestyle='--', label='base')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

def score_plot_prediction(model, x_test, y_test, x_test_close_original, target):
    if target == 'price':
        price_prediction = model.predict(x_test)[1:]
        price_real = x_test_close_original
    else:
        prediction = model.predict(x_test)
        price_prediction = []
        price_real = []
        for i in range(1, len(y_test)):
            price_prediction.append(x_test_close_original[i] * np.exp(prediction[i]))
            price_real.append(x_test_close_original[i] * np.exp(y_test[i]))
    plt.plot(price_prediction, label = 'prediction')
    plt.plot(x_test_close_original[1:], label = 'real', alpha = 0.5)
    plt.ylabel('price')
    plt.legend()
    plt.show()
    
def score_plot_return(model, x_test, y_test, target):
    if target == 'return':
        return_real = y_test
        return_prediction = model.predict(x_test)
    else:
        price_prediction = model.predict(x_test).ravel()
        return_real = []
        return_prediction = []
        for i in range(1, len(price_prediction)):
            return_prediction.append((np.log(price_prediction[i] / price_prediction[i-1])))
        for i in range(1, len(price_prediction)):
            return_real.append((np.log(y_test[i] / y_test[i-1])))  
    plt.plot(return_prediction, label = 'prediction')
    plt.plot(return_real, label = 'real', alpha = 0.5)
    plt.ylabel('return')
    plt.legend()
    plt.show()
    
def score_to_strategy(model, x_test, y_test, task_type = 'classification', model_type = 'sk', target = 'return'):
    prediction = model.predict(x_test)
    if task_type == 'classification':
        if model_type == 'sk':
            strategy = prediction
            strategy[strategy == 0] = -1
        else:
            strategy = prediction[:, 1]
            strategy[strategy > 0.5] = 1
            strategy[strategy <= 0.5] = -1
    else:
        prediction = prediction.ravel()
        if target == 'return':
            strategy = np.sign(prediction)
            strategy[strategy == 0] = -1
        else:
            strategy = np.sign(prediction[1:] - y_test[:-1])
            strategy = strategy.reshape((len(strategy)))
            strategy = np.hstack((1, strategy))
    strategy = strategy.ravel()
    
    return strategy

def score_accuracy(strategy, y_test, task_type = 'classification', target = 'return'):
    y_test = y_test.copy()
    if target == 'return':
        y_test[y_test == 0] = -1
        accuracy = len(strategy[strategy == np.sign(y_test)]) / len(strategy)
    else:
        strategy = strategy[1:]
        real = np.sign(y_test[1:] - y_test[:-1])
        accuracy = len(strategy[strategy == real]) / len(strategy)
    
    return accuracy
  
def score_total_return(strategy, y_open, y_close, commission = 0.01):
    asset = 100
    commission /= 100
    for i in range(1, len(strategy)):
        if strategy[i] == 1:
            asset *= y_close[i] / y_close[i - 1] * (1-commission)
        elif strategy[i] == -1:
            asset *= y_close[i - 1] / y_close[i] * (1-commission)
    
    return asset/100 - 1

def score_annual_return(strategy, y_open, y_close, commission = 0.01):
    total_return = score_total_return(strategy, y_open, y_close, commission = commission)
    
    return ((total_return+1) ** (1/len(strategy))) ** 252 - 1

def score_analyze(model, x_test, y_test, y_open, y_close, model_name, ticker = 'SPY', task_type = 'classification', model_type = 'sk', commission = 0.01, drift = 0, lag = 100000, target = 'return'):
    score = score_model(model, x_test, y_test, model_type = model_type, task_type = task_type)
    strategy = score_to_strategy(model, x_test, y_test, task_type = task_type, model_type = model_type)
    accuracy = score_accuracy(strategy, y_test, task_type = task_type, target = target)
    total_return = score_total_return(strategy, y_open, y_close, commission = commission)
    annual_return = score_annual_return(strategy, y_open, y_close, commission = commission)
    if task_type == 'classification':
        auc = score_auc(model, x_test, y_test, model_type = model_type)
    else:
        r2 = score_r2(model, x_test, y_test, y_close, target)
        if drift >= 1:
            drift = score_calculate_drift(y_close, lag = lag)
        r2_base = score_r2_base(y_close, drift = drift)
    if task_type == 'classification':
        analyze_df = pd.DataFrame(columns = ['ticker', 'model_name','score','accuracy', 'auc', 'realized_total_return', 'total_return', 'annual_return'])
        analyze_df.at[0, 'ticker'] = ticker
        analyze_df.at[0, 'model_name'] = model_name
        #analyze_df.at[0, 'score'] = score
        analyze_df.at[0, 'accuracy'] = accuracy
        analyze_df.at[0, 'auc'] = auc
        analyze_df.at[0, 'realized_total_return'] = y_close[-1] / y_open[0] - 1
        analyze_df.at[0, 'total_return'] = total_return
        analyze_df.at[0, 'annual_return'] = annual_return
    else:
        analyze_df = pd.DataFrame()
        #analyze_df = pd.DataFrame(columns = ['ticker', 'model_name','score', 'accuracy', 'r2', 'r2_base', 'realized_total_return', 'total_return', 'annual_return'])
        analyze_df.at[0, 'ticker'] = ticker
        analyze_df.at[0, 'model_name'] = model_name
        '''
        if model_type == 'tf':
            analyze_df.at[0, 'loss'] = score
        else:
            analyze_df.at[0, 'score'] = score
        '''
        analyze_df.at[0, 'accuracy'] = accuracy
        analyze_df.at[0, 'r2'] = r2
        analyze_df.at[0, 'r2_base'] = r2_base
        analyze_df.at[0, 'realized_total_return'] = y_close[-1] / y_open[0] - 1
        analyze_df.at[0, 'total_return'] = total_return
        analyze_df.at[0, 'annual_return'] = annual_return
    
    return analyze_df


# In[6]:


class stock_model:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    def __init__(self):
        ### parameter
        self.ticker = 'SPY'
        self.start = '1990-01-01'
        self.end = '2019-11-15'
        self.task_type = 'classification'
        self.target = 'return'
        self.model_type = 'sk'
        self.model_name = 'rf'
        self.test_size = 0.05
        self.lag = 50
        self.ta = False
        self.normalization = True
        self.drift_include = False
        self.commission = 0.00
        self.para_list = ['ticker','start','end','task_type','target','model_type','model_name','test_size','lag',
                          'ta','normalization','drift_include','commission']
        self.para_exp = {'ticker':'ticker of the asset, default SPY',
                             'start':'start time of data, default 1990-01-01',
                             'end':'end time of data, default 2019-11-15',
                             'task_type':'task type, either classification or regression, default classification',
                             'target':'target feature of modeling, default return',
                             'model_type':'whether model in sklearn or tensorflow, default sk, can change to tf',
                             'model_name':'abbreviated name of the model, default rf',
                             'test_size':'split proportion of the test set, default 0.3',
                             'lag':'time lag to consider for the model, default 50',
                             'ta':'whether to add technical tickers for the data, default False',
                             'normalization':'whether apply normalization for the data, default True',
                             'drift_include':'whether to include drift for r2_base score , default False',
                             'commission':'commission for testing trading strategy return, default 0.01 in percentage'}
        
        ### variable
        self.data_raw = None
        self.data_added = None
        self.data_taed = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_train_nor = None
        self.x_test_nor = None
        self.x_complie = None
        self.x_train_model = None
        self.x_test_model = None
        self.y_train_model = None
        self.y_test_model = None
        self.analyze_df = None
        self.model = None
        self.history = None
        self.score = None
        self.auc = None
        self.r2 = None
        self.r2_base = None
        self.strategy = None
        self.accuracy = None
        self.total_return = None
        self.annual_return = None
        self.analyze_result = None
        self.drift = None

    ### loading and processing data
    
    def para(self):
        for name in self.para_list:
            print(name + ': ' + str(eval('self.' + name)))
    
    def para_explain(self, para_exp = None):
        if para_exp == None:
            for name in self.para_list:
                print(name + ': ' + self.para_exp[name])
        else:
            for name in para_exp:
                print(name + ': ' + self.para_exp[name])   
    
    def para_change(self, para_dic):
        for i in range(len(para_dic.keys())):
            if type(list(para_dic.values())[i]) == str:
                exec('self.' + str(list(para_dic.keys())[i]) + '=' + "'" + str(list(para_dic.values())[i]) + "'")
            else:
                exec('self.' + str(list(para_dic.keys())[i]) + '=' + str(list(para_dic.values())[i]))
    
    def data_get(self):
        self.data_raw = data_get(self.ticker, self.start, self.end)
    
    def data_add(self):
        self.data_added = data_add_return(self.data_raw)
        self.data_added = data_add_direction(self.data_added)
        self.data_added['price'] = self.data_raw['close']
        self.data_added = self.data_added.loc[self.data_added.index[1:]]
    
    def data_ta(self):
        if self.ta == True:
            self.data_taed = data_ta(self.data_added)
    
    def data_split(self):
        if self.ta == True:
            self.x_train, self.x_test, self.y_train, self.y_test = data_split(self.data_taed, test_size = self.test_size)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = data_split(self.data_added, test_size = self.test_size)
            
    def data_normalize(self):
        if self.normalization:
            self.x_train_nor, self.x_test_nor = data_normalize(self.x_train, self.x_test)
        
    def data_complie(self):
        if self.normalization:
            self.x_complie = data_complie(self.x_train_nor, self.x_test_nor)
        else:
            self.x_complie = data_complie(self.x_train, self.x_test)
    
    def data_to_model(self):
        if self.model_type == 'sk' or self.model_name == 'mlp':
            self.x_train_model, self.x_test_model = data_to_model(self.x_complie, lag = self.lag, test_size = self.test_size, target = self.target)
        elif self.model_name == 'rnn':
            self.x_train_model, self.x_test_model = data_to_rnn(self.x_complie, lag = self.lag, test_size = self.test_size, target = self.target)
        elif self.model_name == 'cnn':
            self.x_train_model, self.x_test_model = data_to_cnn(self.x_complie, lag = self.lag, test_size = self.test_size, target = self.target)
        self.y_train_model = data_y_cut(self.y_train, lag = self.lag)
    
    def data_y_train_model_shrink(self):
        if self.task_type == 'classification':
            self.y_train_model = self.y_train_model['direction'].values
        elif self.target == 'return':
            self.y_train_model = self.y_train_model['return'].values
        else:
            self.y_train_model = self.y_train_model['price'].values
    
    def data_y_test_model_shrink(self):
        if self.task_type == 'classification':
            self.y_test_model = self.y_test['direction'].values
        elif self.target == 'return':
            self.y_test_model = self.y_test['return'].values
        else:
            self.y_test_model = self.y_test['price'].values
    
    def data_prepare(self):
        self.data_get()
        self.data_add()
        self.data_ta()
        self.data_split()
        self.data_normalize()
        self.data_complie()
        self.data_to_model()
        self.data_y_train_model_shrink()
        self.data_y_test_model_shrink()

    ### analyze loaded data
    
    def analyze_raw(self):
        self.analyze_df = analyze_raw(self.data_added, self.ticker)
        
        return self.analyze_df
    
    def plot_series(self):
        plot_series(self.data_added, self.target)
    
    def plot_hist(self, bins = 50):
        plot_hist(self.data_added, self.target, bins)
        
    def plot_auto(self, lag = 100, ran = 0.1):
        plot_auto(self.data_added, self.target)
    
    def plot_all(self, bins = 50, lag = 100, ran = 0.1):
        if self.target == 'price':
            if ran == 0.1:
                ran = 1
            if lag == 100:
                lag = 1000
        plot_all(self.data_added, self.target, bins, lag, ran)
    
    ### build model
    
    def model_build_sk(self, model_para = ''):
        self.model = model_build_sk(self.model_name, task_type = self.task_type, model_para = model_para)
    
    def model_build_tf(self, number_layer = 3, width = 50, number_drop_out = 0.2, optimizer = 'adam'):
        if self.model_name == 'mlp':
            self.model = model_build_mlp(self.x_train_model, self.task_type, number_layer = number_layer, width = width, number_drop_out = number_drop_out, optimizer = optimizer)
        elif self.model_name == 'cnn':
            self.model = model_build_cnn(self.x_train_model, self.task_type, number_layer = number_layer, width = width, number_drop_out = number_drop_out, optimizer = optimizer)
        elif self.model_name == 'rnn':
            self.model = model_build_lstm(self.x_train_model, self.task_type, number_layer = number_layer, width = width, number_drop_out = number_drop_out, optimizer = optimizer)
    
    ### train model
    
    def model_train_sk(self):
        self.model = model_train_sk(self.model, self.x_train_model, self.y_train_model)
        
    def model_train_tf(self, epoch = 50, batch_size = 64):
        self.history, self.model = model_train_tf(self.model, self.x_train_model, self.y_train_model, epoch = epoch, batch_size = batch_size)
        
    ### test performance
    
    def score_model(self):
        self.score = score_model(self.model, self.x_test_model, self.y_test_model, model_type = self.model_type, task_type = self.task_type)
       
        return self.score
    
    def score_auc(self):
        self.auc = score_auc(self.model, self.x_test_model, self.y_test_model, model_type = self.model_type)

        return self.auc
    
    def score_r2(self):
        self.r2 = score_r2(self.model, self.x_test_model, self.y_test_model, self.x_test_model['close'].values, model_type = self.model_type, target = self.target)
        
        return self.r2
    
    def score_r2_base(self, lag_drift = 100000):
        if self.drife_include:
            self.drift = score_calculate_drift(self.x_train['close'].values, lag = lag_drift)
            self.r2_base = score_r2_base(self.x_train['close'].values, 0)
        else:
            self.r2_base = score_r2_base(self.x_train['close'].values.x_train, 0)
        
        return self.r2_base
    
    def score_plot_roc(self):
        score_plot_roc(self.model, self.x_test_model, self.y_test_model, model_type = self.model_type, target = self.target)
        
    def score_plot_prediction(self):
        score_plot_prediction(self.model, self.x_test_model, self.y_test_model, self.x_test['close'].values, target = self.target)
    
    def score_plot_return(self):
        score_plot_return(self.model, self.x_test_model, self.y_test_model, target = self.target)
    
    def score_to_strategy(self):
        self.strategy = score_to_strategy(self.model, self.x_test_model, self.y_test_model, task_type = self.task_type, model_type = self.model_type)
    
    def score_accuracy(self):
        self.accuracy = score_accuracy(self.strategy, self.y_test_model, task_type = self.task_type, target = self.target)
        
    def score_total_return(self):
        self.total_return = score_total_return(self.strategy, self.x_test['open'].values, self.x_test['close'].values, commission = self.commission)
        
    def score_annual_return(self):
        self.annual_return = score_annual_return(self.strategy, self.x_test['open'].values, self.x_test['close'].values, commission = self.commission)

    def score_analyze(self, lag_drift = 100000):
        drift = 0
        if self.drift_include:
            drift = score_calculate_drift(self.x_train['close'].values, lag = lag_drift)
        self.analyze_result = score_analyze(self.model, self.x_test_model, self.y_test_model, self.x_test['open'].values, self.x_test['close'].values, self.model_name, ticker = self.ticker, task_type = self.task_type, model_type = self.model_type, commission = self.commission, drift = drift, lag = lag_drift, target = self.target)
        
        return self.analyze_result           

