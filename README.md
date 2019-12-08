# stock_price_by_historical
A python pipeline of using models from sklearn and tensorflow to predict (classification or regression) stock price by its historical price.

Using machine learning and deep learning models to predict stock price is a hot topic. There are many papers and blogs show cases using historical stock price as features to predict future stock price, and they include prediction plots (predicted price vs. real price) to demonstrate the model is useful. However, even a last value model (use yesterday's close price as the prediction of today's close price) is powerful to generate a plausible plot, which is very close to the trend of real price. According to efficient market hypothesis, all the public infomation, e.g., historical stock price, should have been fully digested by the market and thus is useless to predict the future.

I built this pipeline to control the process of acquring stock price by yfinance api, data pre-processing, modeling and evaluating. You can change the parameters of the pipeline to build model and test its performance. The parameters include ticker, start or end time, model_type (scikit- learn or tensorflow), model_name (e.g. knn, rf, rnn), lag (how many days to look back) and many others. For details you may use .para_explain().

I defined the prediction task type as classification or regression. Classification is to predict whether the stock would go up or down; regression is to predict the exact stock price. The evaluation metric for classification is AUC, and for regression is r2 compared with r2_base (r2 of the last value model, usually 0.92- 0.97). A naive trading strategy, if the model predicts tomorrow as up, then long today's close and short tomorrow's close, is used to test the return of the model. Based on the experiments I did, no model are practically useful in predicting.

This is version 1.0 and I may more functions or make it more generalized in the future. Now you can use the pre-defined parameters to handle the pipeline, and you can assign the models you are interested in to customize (e.g. template.model = sklearn.svm.SVC()), or you can use it to download and pre-process data, and get them by data_x, data_y = template.x_test_model, template.y_test_model.

Have fun!

Here's the contents in the stock_model_using_template.ipynb:

Import the package.


```python
import stock_model
```

Apply stock_model class to a variable.


```python
template = stock_model.stock_model()
```

Report parameters and get explaination.


```python
template.para()
```

    ticker: SPY
    start: 1990-01-01
    end: 2019-11-15
    task_type: classification
    target: return
    model_type: sk
    model_name: rf
    test_size: 0.05
    lag: 50
    ta: False
    normalization: True
    drift_include: False
    commission: 0.0
    


```python
template.para_explain()
```

    ticker: ticker of the asset, default SPY. string
    start: start time of data, default 1990-01-01. string
    end: end time of data, default 2019-11-15. string
    task_type: task type, either classification or regression, default classification. string
    target: target feature of modeling, if model_name = "tf" use "price", default return. string
    model_type: whether model in sklearn or tensorflow, default sk, can change to tf. string
    model_name: abbreviated name of the model, default rf, for sk, can choose lasso, ridge, knn, rf(random forest), adb(adaboosting), for tf, can choose mlp, rnn, cnn. string
    test_size: split proportion of the test set, default 0.05. float
    lag: time lag to consider for the model, default 50. int
    ta: whether to add technical tickers for the data, RSI, MACD, ADX, Bollinger Band, default False. bool
    normalization: whether apply normalization for the data, default True. bool
    drift_include: whether to include drift for r2_base score , default False. bool
    commission: commission for testing trading strategy return, default 0.00 in percentage. float
    


```python
template.para_explain(['ticker','lag'])
```

    ticker: ticker of the asset, default SPY. string
    lag: time lag to consider for the model, default 50. int
    

Change parameters.


```python
template.ticker = 'aapl'
```


```python
template.para_change({'ticker':'aapl','task_type':'reg','target':'return','normalization':False})
```


```python
template.para()
```

    ticker: aapl
    start: 1990-01-01
    end: 2019-11-15
    task_type: reg
    target: return
    model_type: sk
    model_name: rf
    test_size: 0.05
    lag: 50
    ta: False
    normalization: False
    drift_include: False
    commission: 0.0
    

Getting and processing the data.


```python
template.target = 'price'
template.data_prepare()
```

    [*********************100%***********************]  1 of 1 completed
    

Descriptive analysis of the data.


```python
template.plot_all(lag = 2000, ran = 1)
```


![png](using_template_picutres/output_15_0.png)



![png](using_template_picutres/output_15_1.png)



![png](using_template_picutres/output_15_2.png)



```python
template.analyze_raw()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>start</th>
      <th>end</th>
      <th>days</th>
      <th>low</th>
      <th>high</th>
      <th>now</th>
      <th>annual_return</th>
      <th>total_return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>aapl</td>
      <td>1990-01-03 00:00:00</td>
      <td>2019-11-15 00:00:00</td>
      <td>7528</td>
      <td>0.462054</td>
      <td>265.76</td>
      <td>265.76</td>
      <td>0.193219</td>
      <td>195.823</td>
    </tr>
  </tbody>
</table>
</div>



Build the model.


```python
template.model_build_sk(model_para='n_jobs = -1')
```

Train the model.


```python
template.model_train_sk()
```

    c:\users\zfan2\anaconda3\envs\tf_gpu\lib\site-packages\sklearn\ensemble\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    

Evaluate the model.


```python
template.score_analyze()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>model_name</th>
      <th>accuracy</th>
      <th>r2</th>
      <th>r2_base</th>
      <th>realized_total_return</th>
      <th>total_return</th>
      <th>annual_return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>aapl</td>
      <td>rf</td>
      <td>0.550532</td>
      <td>-6.832646</td>
      <td>0.979166</td>
      <td>0.413617</td>
      <td>0.416405</td>
      <td>0.261999</td>
    </tr>
  </tbody>
</table>
</div>



Plot the model prediction.


```python
template.score_plot_prediction()
template.score_plot_return()
```


![png](using_template_picutres/output_24_0.png)



![png](using_template_picutres/output_24_1.png)


Quick Version


```python
import stock_model
template = stock_model.stock_model()
template.para_change({'ticker':'dia','task_type':'classification','target':'return','model_name':'rnn','model_type':'tf'})
template.para()
```

    ticker: dia
    start: 1990-01-01
    end: 2019-11-15
    task_type: classification
    target: return
    model_type: tf
    model_name: rnn
    test_size: 0.05
    lag: 50
    ta: False
    normalization: True
    drift_include: False
    commission: 0.0
    


```python
template.data_prepare()
template.plot_all()
```

    [*********************100%***********************]  1 of 1 completed
    


![png](using_template_picutres/output_27_1.png)



![png](using_template_picutres/output_27_2.png)



![png](using_template_picutres/output_27_3.png)



```python
template.model_build_tf(number_layer = 2, width = 50)
template.model_train_tf(epoch = 10, batch_size = 64)
template.score_analyze()
```

    Epoch 1/10
    5168/5168 [==============================] - 15s 3ms/sample - loss: 0.7158 - acc: 0.5064
    Epoch 2/10
    5168/5168 [==============================] - 14s 3ms/sample - loss: 0.6999 - acc: 0.51241s - loss: 0.6986 - 
    Epoch 3/10
    5168/5168 [==============================] - 14s 3ms/sample - loss: 0.7002 - acc: 0.5155
    Epoch 4/10
    5168/5168 [==============================] - 14s 3ms/sample - loss: 0.6985 - acc: 0.50432s - loss: 0.6990 -
    Epoch 5/10
    5168/5168 [==============================] - 14s 3ms/sample - loss: 0.6955 - acc: 0.51929s - loss: 0.6 - 
    Epoch 6/10
    5168/5168 [==============================] - 14s 3ms/sample - loss: 0.6958 - acc: 0.5037
    Epoch 7/10
    5168/5168 [==============================] - 14s 3ms/sample - loss: 0.6956 - acc: 0.5130
    Epoch 8/10
    5168/5168 [==============================] - 14s 3ms/sample - loss: 0.6931 - acc: 0.5213
    Epoch 9/10
    5168/5168 [==============================] - 14s 3ms/sample - loss: 0.6934 - acc: 0.5193
    Epoch 10/10
    5168/5168 [==============================] - 14s 3ms/sample - loss: 0.6944 - acc: 0.51453s - loss
    275/275 [==============================] - 1s 3ms/sample - loss: 0.6898 - acc: 0.5564
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>model_name</th>
      <th>accuracy</th>
      <th>auc</th>
      <th>realized_total_return</th>
      <th>total_return</th>
      <th>annual_return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>dia</td>
      <td>rnn</td>
      <td>0.556364</td>
      <td>0.482749</td>
      <td>0.107051</td>
      <td>0.109376</td>
      <td>0.0997871</td>
    </tr>
  </tbody>
</table>
</div>
