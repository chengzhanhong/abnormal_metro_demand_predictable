# Forecastable
Code for "Anomalies in metro passenger demand are predictable -- causal inference with transformer"
The structure of the code is as follows:
```
├── models             <- Implementation of different models, including the proposed, PatchTST, Dlinear, LSTM, etc.>
├── exps               <- Scripts to run experiments for different models. ->
     ├── ABtranformer  <- Experiments for the proposed ABtranformer. ->
     ├── PatchTST      <- Experiments for PatchTST. ->
     ├── Dlinear       <- Experiments for Dlinear. ->
     ├── LSTM          <- Experiments for LSTM. ->
├── datasets           <- Scripts to prepare datasets for training and testing for different models. ->
├── utilites           <- Utilities like loss functions, learning rate schedulers, etc. ->
```