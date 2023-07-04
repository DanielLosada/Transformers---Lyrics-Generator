# Transformers - Lyrics-Generator
## Dataset
The dataset used for this project is the [Lyrics Dataset](https://www.kaggle.com/datasets/mervedin/genius-lyrics) from Kaggle.

Or this one: [Lyrics Dataset](https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics)

Or this one: [Lyrics Dataset](https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres)

## Installation
### With Conda
Create a conda environment by running
```
conda create --name LyricsGenerator python=3.8
```
Then, activate the environment
```
conda activate LyricsGenerator
```
and install the dependencies
```
pip install -r requirements.txt
```
The first line of the file
```
--extra-index-url https://download.pytorch.org/whl/cu117
```
it's used to install the torch version compatible with cuda in windows.
