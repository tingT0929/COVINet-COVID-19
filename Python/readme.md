Before running the model, do download the data files and obtain data by running
```shell
python data_pro.py
```

For confirmed cases prediction, run the model by
```shell
python lstm_cov_factor_server.py --target 0
```

For confirmed deaths prediction, run the model by
```shell
python lstm_cov_factor_server.py --target 1
```

For altering seed(eg. setting seed to 1), run the model by
```shell
python lstm_cov_factor_server.py --target 1 --seed 1
```

For model comparison, run the model by
```shell
python lstm_cov_factor_server.py --target 1 --seed 1 --limit_date '2021-04-26'
```
