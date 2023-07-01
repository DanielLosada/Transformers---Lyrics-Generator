import argparse
import torch
import json
import os
from pynvml import *
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from dataset import LyricsDataset
from generator import *

# Load device to use eith GPU or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def print_gpu_utilization():
    """Prints GPU usage while training"""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    """Prints training summary results"""
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    # print_gpu_utilization()

def train_model(dataset, tokenized_dataset, save_name):
    model = AutoModelForCausalLM.from_pretrained(config["model"]).to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{config['model']} size: {model_size/1000**2:.1f}M parameters")
    training_args = TrainingArguments("trainer", per_device_train_batch_size=4, evaluation_strategy="epoch", num_train_epochs=10, save_strategy="epoch", load_best_model_at_end=True)
    data_collator = DataCollatorForLanguageModeling(dataset.tokenizer, mlm=False)
    trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            tokenizer=dataset.tokenizer,
            #compute_metrics=compute_metrics,
        )
    # print_gpu_utilization()
    trainer.train()
    #print_summary(result)
    trainer.save_model("./models/" + save_name.replace(" ", "_"))

if __name__ == "__main__":
    os.chdir('/home/paurosci/gits/Transformers---Lyrics-Generator/gpt2-model')
    print(os.getcwd())

    # Load the configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Parse the command-line options
    parser = argparse.ArgumentParser(prog='Lyrics generator', description="Trains a lyrics generation model based on GPT-2 architecture.")
    parser.add_argument("-ts", "--trainSingleArtist", dest='train_single_artist', type=str, help="Prepare the dataset of the choosen artist and train the model")
    parser.add_argument("-tm", "--trainMultipleArtists", dest='train_multiple_artist', action="store_true", help="Prepare the dataset of all the artists and train the model")
    parser.add_argument("-tg", "--trainGenre", dest='train_genre', type=str, help="Pass the genre to train the model with songs of that genre")
    parser.add_argument("-gs", "--generateSingleArtist", dest='generate_single_artist', type=str, help="Pass the artist name to generate lyrics. Use the same name you used to train it.")
    parser.add_argument("-gm", "--generateMultipleArtist", dest='generate_multiple_artist',type=str, help="Pass the artist name to generate lyrics with the model trained with multiple artists. Use the same name you used to train it.")
    parser.add_argument("-gg", "--generateMultipleArtistGenre", dest='generate_multiple_artist_genre',type=str, help="Pass the artist name to generate lyrics with the model trained with multiple artists genre. Use the same name you used to train it.")
    parser.add_argument("-ds", "--datasetSelection", dest='dataset_selection', choices=["genious-lyrics","79-musical-genres"], help="Offers dataset selection between two choices")
    args = parser.parse_args()

    # Set default arguments
    if args.dataset_selection == None:
        args.dataset_selection="genious-lyrics"

    # Training options
    if(args.train_single_artist):
        print("Selected single-artist training: ", args.train_single_artist)
        lyrics_dataset = LyricsDataset(config, args.train_single_artist, args.dataset_selection)
        lyrics_dataset.load_dataset_single_artist()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, args.train_single_artist)
    elif(args.train_multiple_artist):
        print("Selected multi-artist tranining")
        lyrics_dataset = LyricsDataset(config, "multipleArtists", args.dataset_selection)
        lyrics_dataset.load_dataset_multiple_artists()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, "multipleArtists")
    elif(args.train_genre):
        print("Selected genre tranining")
        lyrics_dataset = LyricsDataset(config, args.train_genre, "79-musical-genres")
        lyrics_dataset.load_dataset_multiple_artists()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, "multipleArtistsGenre")

    # Generation options
    if(args.generate_single_artist):
        print("Selected single-artist generation: ", args.generate_single_artist)
        lyrics_generator_params = LyricsGeneratorParams
        lyrics_generator_params.max_length = 10
        lyrics_generator = LyricsGenerator(config, args.generate_single_artist, lyrics_generator_params)
        lyrics_generator.generate_lyrics(initial_prompt="My name is")
    elif(args.generate_multiple_artist):
        print("Selected multiple-artist generation")
        lyrics_generator_params = LyricsGeneratorParams
        lyrics_generator_params.max_length = 10
        initial_prompt="You are"
        lyrics_generator = LyricsGenerator(config, "multipleArtists", lyrics_generator_params)
        lyrics_generator.generate_lyrics(args.generate_multiple_artist + ': ' + initial_prompt)

    elif(args.generate_multiple_artist_genre):
            print("Selected multiple-artist genre generation")
            lyrics_generator_params = LyricsGeneratorParams
            lyrics_generator_params.max_length = 20
            initial_prompt="You are"
            lyrics_generator = LyricsGenerator(config, "multipleArtistsGenre", lyrics_generator_params)
            lyrics_generator.generate_lyrics(args.generate_multiple_artist_genre + ': ' + initial_prompt)


