# Instructions

Run `python3 data/batch_download_zips.py` to download and unzip all the training data. Also download `Data_Entry_2017_v2020.csv` to get the labels for the training data. 

To train, run `python3 train.py`

To evaluate metrics, run `python3 test.py --data_dir data --model_dir model/{model_name}.pt --dataset {train|test|valid}`

My results are shown in the results folder. I mainly trained two different models to completion, one was more complex and reached 99% training accuracy but overfit and got 60% on validation accuracy. The other model was less complex but only reached 75% training accuracy and got 65% on validation accuracy.