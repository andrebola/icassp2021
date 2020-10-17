# icassp2021

Code to reproduce the paper. 


# Instructions

In order to run this code you first need to install the dependencies: `pip install -r requirements.txt`


Then, there are 3 python script to run in the following order:
 - `python cf_train.py`: generate all the splits, train the CF and generate the latent factors
 - `python autotagging_vgg.py`: train the audio model to predict the latent factors
 - `factors_eval.py`: use latent factors to generate the recommendations and evaluate them


Before running the script `autotagging_vgg.py` you need to set the location of the dataset `TMP_PATH` and a folder to save the checkpoints in `MODELS_PATH`. 
