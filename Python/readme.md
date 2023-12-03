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
