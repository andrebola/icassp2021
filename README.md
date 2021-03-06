# Melon Playlist Dataset

Code to reproduce the paper: "**MELON PLAYLIST DATASET: A PUBLIC DATASET FOR AUDIO-BASED PLAYLIST GENERATION AND MUSIC TAGGING**" by Andres Ferraro, Yuntae Kim, Soohyeon Lee, Biho Kim, Namjun Jo, Semi Lim, Suyon Lim, Jungtaek Jang, Sehwan Kim, Xavier Serra and Dmitry Bogdanov


# Instructions

In order to run this code you first need to install the dependencies: `pip install -r requirements.txt`


Then, there are 3 python script to run in the following order:
 - `python cf_train.py`: generate all the splits, train the CF and generate the latent factors
 - `python autotagging_vgg.py`: train the audio model to predict the latent factors
 - `factors_eval.py`: use latent factors to generate the recommendations and evaluate them


Before running the script `autotagging_vgg.py` you need to set the location of the dataset `TMP_PATH` and a folder to save the checkpoints in `MODELS_PATH`. 


In this repository we also provide the code to generate the mel-spectrograms that are included in the dataset. For this purpose, the method `melspectrogram` of the file `audio_extract.py` was used.

To download the dataset follow the instructions in the repository: https://mtg.github.io/melon-music-dataset/
