import argparse
import torch
import json
from dataset import LyricsDataset

# Load device to use eith GPU or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    # Load the configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Parse the command-line options
    parser = argparse.ArgumentParser(prog='Lyrics generator', description="Trains a lyrics generation model based on GPT-2 architecture.")
    parser.add_argument("-s", "--trainSingleArtist", dest='single_artist', type=str, help="Prepare the dataset of the choosen artist and train the model")
    parser.add_argument("-m", "--trainMultipleArtists", dest='multiple_artist', action="store_true", help="Prepare the dataset of all the artists and train the model")
    parser.add_argument("-d", "--databaseSelection", dest='database_selection', choices=["genious-lyrics","79-musical-genres"], help="Offers dataset selection between two choices")
    args = parser.parse_args()

    # Set default arguments
    if args.database_selection == None:
        args.database_selection="79-musical-genres"

    if(args.single_artist):
        print("Selected single-artist training: ", args.single_artist)
        lyrics_dataset = LyricsDataset(config, args.single_artist, args.database_selection)
    elif(args.multiple_artist):
        print("Selected multi-artist tranining")
    else:
        print("Unknown training option")
