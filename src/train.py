from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoLSTM
from neuralforecast.losses.pytorch import HuberLoss
from utils.helper_functions import my_lstm_config
import optuna
from pathlib import Path

# df will be loaded from our DB or Amazon S3 blob
def Train_and_save_predictor(df):
    root = Path().resolve().parent
    models = [

        AutoLSTM(
        h=7,  
        loss=HuberLoss(delta=1.35),  
        config=my_lstm_config,
        search_alg=optuna.samplers.TPESampler(),
        backend="optuna",
        num_samples=10,
        ),
    ]

    nf = NeuralForecast(models=models, freq='D')
    nf.fit(df)
    nf.save(path=f'{root}/models/LSTM',
        model_index=None, 
        overwrite=True,
        save_dataset=True) # supposed to be saved to an S3 bucket
    
def Train_and_save_cluster_mdl(df):
    pass
