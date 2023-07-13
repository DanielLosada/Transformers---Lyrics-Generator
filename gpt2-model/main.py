import argparse
import torch
import json
import os
import statistics
import wandb

from datetime import datetime
from pynvml import *
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from nltk.translate.bleu_score import sentence_bleu
from dataset import LyricsDataset
from generator import *

# Load device to use eith GPU or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#### MODEL TRAINING FUNCTIONS #### 
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

def train_model(dataset, tokenized_dataset, save_name=''):
    os.environ["WANDB_API_KEY"] = config["wandb"]["wandb_api_key"]
    wandb.init(project="Lyrics-Generator")
    #wandb.run.name = f'{save_name.replace(" ", "_")}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    model = AutoModelForCausalLM.from_pretrained(config["model"]).to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{config['model']} size: {model_size/1000**2:.1f}M parameters")
    training_args = TrainingArguments("trainer", report_to="wandb", run_name=f'{save_name.replace(" ", "_")}-{datetime.now().strftime("%Y%m%d-%H%M%S")}',per_device_train_batch_size=4, evaluation_strategy="epoch", num_train_epochs=config["epochs"], save_strategy="epoch", load_best_model_at_end=True)
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
    wandb.finish()
    if(save_name):
        trainer.save_model("./models/" + save_name.replace(" ", "_"))

#### MODEL PERFORMANCE EVALUATION FUNCTIONS ####
class BleuPerformanceParams:
    total_eval_datasets = 0
    prompt_words = []
    reference_words = []
    generated_words = []
    bleu_scores = []

def generate_performance_prompt(test_data, n_verses=None):
    '''Selects last n sentences as prompt'''
    prompt_texts=[]
    for i in range(len(test_data['test_trimmed_lyrics'])):
        split_dataset = test_data['test_trimmed_lyrics'][i].split('\n')
        # Remove empty words
        while("" in split_dataset): split_dataset.remove("")
        
        if(n_verses):
            # Remove verses
            for j in range(len(split_dataset)-n_verses):
                split_dataset.pop(0)
    
        # Join dataset
        prompt_texts.append('\n'.join(split_dataset))
    return prompt_texts

def word_counter(data):
    '''counts the amount of words present in data'''
    n_words = 0
    split_dataset = [i.split() for i in data.split('\n')]
    
    for i in range(len(split_dataset)):
        n_words += len(split_dataset[i])
    return n_words

def remove_last_words(data, n):
    '''removes the last n words from data'''
    split_data = [i.split() for i in data.split('\n')]
    word_count=0
    for j in reversed(range(0,len(split_data))):
        if split_data[j] != '[]':
            for k in reversed(range(len(split_data[j]))):
                if word_count == n:
                    break
                else:
                    # print("removed word count: ", word_count, " removed word: ", split_data[j][k])
                    split_data[j].pop(k)
                    word_count+=1
            else:
                continue
            break
    print('removed '  + str(word_count) + ' words from data')
    data = '\n'.join(' '.join(v) for v in split_data)
    return data

def compute_bleu_metric(artist, artist_generation=None):
     # Load stored test lyrics information to avoid training again
    f = open("./models/" + artist.replace(" ", "_") + "_performance/lyrics_test.json")
    test_data=json.load(f)

    # generate performance prompt
    initial_prompt = generate_performance_prompt(test_data)

    # perform lyrics generation 
    lyrics_generator_params = LyricsGeneratorParams
    lyrics_generator_params.num_sequences = 1
    lyrics_generator_params.max_length = 1024
    lyrics_generator_params.min_length = 1024
    # lyrics_generator_params.temperature = 1
    # lyrics_generator_params.top_p = 0.8
    lyrics_generator = LyricsGenerator(config, artist + "_performance", lyrics_generator_params)

    # Perform text generation and compute bleu scores
    performance_data = BleuPerformanceParams
    performance_data.total_eval_datasets = 0
    # generation_offset = 400
    for i in range(0,len(test_data['test_trimmed_lyrics'])):
        if(artist_generation):
            # modify prompt for multiArtists generation
            initial_prompt[i] = artist_generation + ': ' + initial_prompt[i]
        initial_prompt_words = word_counter(initial_prompt[i])
        true_lyrics_words = word_counter(test_data['test_true_lyrics'][i])

        # Check that true lyrics dataset contains the expected amount of verses
        # if( verse_counter(test_data['test_true_lyrics'][i]) != removed_verses):
        #     print("\n" + "#"*50 + " Test set number = ", str(i+1), "was bypassed...")
        #     continue
        # Drop long generations (after more than 1024 tokens it doesn't work)
        # if((initial_prompt_words + true_lyrics_words) > 1024):
        #     print("\n" + "#"*50 + " Test set number = ", str(i+1), "was bypassed...")
        #     continue

        # Check that prompt is not empty
        if( true_lyrics_words <= 0):
            print("Sequence could not be generated: true_lyrics data was empty")
            print("Test set number = ", str(i+1), "was bypassed...")
            continue

        # Generate sentence with trained model
        #lyrics_generator.params.max_length = (initial_prompt_words + true_lyrics_words) + generation_offset
        #lyrics_generator.params.min_length = (initial_prompt_words + true_lyrics_words) + generation_offset

        print("\n" + "#"*50 + " Test set number = ", str(i+1))
        print("generator selected max_length =", lyrics_generator.params.max_length)
        print("generator selected min_length =", lyrics_generator.params.min_length)
        try: 
            lyrics_generator.generate_lyrics(initial_prompt=initial_prompt[i])
        except Exception as error:
            print("Sequence could not be generated:", error)
            print("Test set number = ", str(i+1), "was bypassed...")
            continue

        print("\n"+"&"*20 + " Initial prompt " + "&"*20)
        print(initial_prompt[i])
        print("\n"+"&"*20 + "   True  text   " + "&"*20)
        print(initial_prompt[i] + '\n' + test_data['test_true_lyrics'][i])
        print("\n"+"&"*20 + " Generated text " + "&"*20)
        print(lyrics_generator.generated[0])
        print("\n" + "&"*25)
        
        # Postprocess generated data
        true_lyrics = test_data['test_true_lyrics'][i]
        # Remove prompt from generated data
        generated_lyrics = lyrics_generator.generated[0].replace(initial_prompt[i], '', 1)
        # Remove line breaks from generated data
        generated_lyrics = generated_lyrics.replace('\n', ' ')
        generated_words = word_counter(generated_lyrics)
        
        print("\n" + "&"*25)
        print("prompt words      =", initial_prompt_words)
        print("true_lyrics words =", true_lyrics_words)
        print("generated words   =", generated_words)
        print("&"*25)
        
        # Trim generated words
        if(generated_words > true_lyrics_words):
            generated_lyrics = remove_last_words(generated_lyrics, abs(true_lyrics_words-generated_words))
            generated_words = word_counter(generated_lyrics)
        
        # Store performance data
        performance_data.prompt_words.append(initial_prompt_words)
        performance_data.reference_words.append(true_lyrics_words)
        performance_data.generated_words.append(generated_words)

        print("\n"+"&"*20 + "  Reference  " + "&"*20)
        print(true_lyrics)
        print("\n"+"&"*20 + "  Candidate  " + "&"*20)
        print(generated_lyrics)
        print("\n"+"&"*20 + "  Bleu Score  " + "&"*20)

        # compute bleu metric
        reference = true_lyrics.split()
        candidate = generated_lyrics.split()
        performance_data.bleu_scores.append(sentence_bleu([reference], candidate, weights=(1,)))
        performance_data.total_eval_datasets+=1
        print("bleu score   =", performance_data.bleu_scores[-1])

    # Display final results
    print("\n" + "&"*20 + " General statistics " + "&"*20 + "\n")
    if(performance_data.total_eval_datasets > 0):
        print("Total number of evaluated datasets = ", performance_data.total_eval_datasets)
        print("Average number of prompt words     = {:.2f}".format(statistics.mean(performance_data.prompt_words)))
        print("Average number of reference words  = {:.2f}".format(statistics.mean(performance_data.reference_words)))
        print("Average number of generated words  = {:.2f}".format(statistics.mean(performance_data.generated_words)))
        print("Average bleu score                 = {:.2f}".format(statistics.mean(performance_data.bleu_scores)))
        print("\n" + "&"*60 + "\n")

    return performance_data

if __name__ == "__main__":
    # TODO: remove this
    # os.chdir('/home/paurosci/gits/Transformers---Lyrics-Generator/gpt2-model')

    # Load the configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Parse the command-line options
    parser = argparse.ArgumentParser(prog='Lyrics generator', description="Trains a lyrics generation model based on GPT-2 architecture.")
    parser.add_argument("-ts", "--trainSingleArtist", dest='train_single_artist', type=str, help="Prepare the dataset of the choosen artist and train the model")
    parser.add_argument("-tm", "--trainMultipleArtists", dest='train_multiple_artists', action="store_true", help="Prepare the dataset of all the artists and train the model")
    parser.add_argument("-tg", "--trainGenre", dest='train_genre', type=str, help="Pass the genre to train the model with songs of that genre")
    
    parser.add_argument("-gs", "--generateSingleArtist", dest='generate_single_artist', type=str, help="Pass the artist name to generate lyrics. Use the same name you used to train it.")
    parser.add_argument("-gm", "--generateMultipleArtist", dest='generate_multiple_artists',type=str, help="Pass the artist name to generate lyrics with the model trained with multiple artists. Use the same name you used to train it.")
    parser.add_argument("-gg", "--generateGenre", dest='generate_genre',type=str, help="Pass the artist name to generate lyrics with the model trained with multiple artists genre. Use the same name you used to train it.")
    parser.add_argument("-ds", "--datasetSelection", dest='dataset_selection', choices=["genious-lyrics","79-musical-genres"], default = "genious-lyrics", help="Offers dataset selection between two choices")
    
    parser.add_argument("-sp", "--singleArtistPerformance", nargs=3, dest='single_artist_performance', help="Computes the bleu metric to evaluate the single-artist model")
    parser.add_argument("-mp", "--multipleArtistsPerformance", nargs=3, dest='multiple_artists_performance', help="Computes the bleu metric to evaluate the multiple-artist model")
    parser.add_argument("-gp", "--genrePerformance", nargs=3, dest='genre_performance', help="Computes specified metric to evaluate the specified model")

    args = parser.parse_args()

    # TODO: remove this
    # args.single_artist_performance=['50 Cent', 'False', '2']
    # args.multiple_artists_performance=['Eminem10', 'False', '1']
    # args.train_multiple_artists = True
    # args.genre_performance = ['Rock', 'True', '1']
    
    # Training options
    if(args.train_single_artist):
        print("Selected single-artist training: ", args.train_single_artist)
        lyrics_dataset = LyricsDataset(config, args.train_single_artist, args.dataset_selection)
        lyrics_dataset.load_dataset_single_artist()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, args.train_single_artist + '_' + args.dataset_selection)
    elif(args.train_multiple_artists):
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
        train_model(lyrics_dataset, tokenized_dataset, args.train_genre)
   
    # Generation options
    if(args.generate_single_artist):
        print("Selected single-artist generation: ", args.generate_single_artist)
        lyrics_generator_params = LyricsGeneratorParams
        lyrics_generator_params.max_length = 10
        lyrics_generator = LyricsGenerator(config, args.generate_single_artist, lyrics_generator_params)
        lyrics_generator.generate_lyrics(initial_prompt="My name is")
    elif(args.generate_multiple_artists):
        print("Selected multiple-artist generation")
        lyrics_generator_params = LyricsGeneratorParams
        lyrics_generator_params.max_length = 10
        initial_prompt="You are"
        lyrics_generator = LyricsGenerator(config, "multipleArtists", lyrics_generator_params)
        lyrics_generator.generate_lyrics(args.generate_multiple_artists + ': ' + initial_prompt)
    elif(args.generate_genre):
        print("Selected multiple-artist genre generation")
        lyrics_generator_params = LyricsGeneratorParams
        lyrics_generator_params.max_length = 30
        initial_prompt="You are"
        lyrics_generator = LyricsGenerator(config, "multipleArtistsGenre", lyrics_generator_params)
        lyrics_generator.generate_lyrics(args.generate_genre + ': ' + initial_prompt)

    # Performance evaluation options
    if (args.single_artist_performance):
        print("Selected single artist performance evaluation")
        artist = args.single_artist_performance[0]
        train = args.single_artist_performance[1]
        n_verses = int(args.single_artist_performance[2])

        # Train model
        if(train == 'True'):
            lyrics_dataset = LyricsDataset(config, artist, "genious-lyrics", performance_evaluation_nverses=n_verses)
            lyrics_dataset.load_dataset_single_artist()
            tokenized_dataset = lyrics_dataset.dataset.map(
                lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
            )
            train_model(lyrics_dataset, tokenized_dataset, artist.replace(" ", "_") + "_performance")

            # Store test lyrics in json file in order to prevent training again
            test_lyrics = {}
            test_trimmed_lyrics = []
            for i in range(len(lyrics_dataset.dataset['test']['lyrics'])):
                test_trimmed_lyrics.extend([str(lyrics_dataset.dataset['test']['lyrics'][i])])
            test_lyrics['test_trimmed_lyrics'] = test_trimmed_lyrics
            test_lyrics['test_true_lyrics'] = lyrics_dataset.true_lyrics_dataset
            with open("./models/" + artist.replace(" ", "_") + "_performance/lyrics_test.json","w") as f:
                json.dump(test_lyrics, f)

        performance_data = compute_bleu_metric(artist)

    elif(args.multiple_artists_performance):
        print("Selected multiple artists performance evaluation")
        artist_generation = args.multiple_artists_performance[0]
        train = args.multiple_artists_performance[1]
        n_verses = int(args.multiple_artists_performance[2])
        artist = 'multipleArtists'

        # Train model
        if(train == 'True'):
            lyrics_dataset = LyricsDataset(config, 'multipleArtists', "genious-lyrics", performance_evaluation_nverses=n_verses)
            lyrics_dataset.load_dataset_multiple_artists()
            tokenized_dataset = lyrics_dataset.dataset.map(
                lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
            )
            train_model(lyrics_dataset, tokenized_dataset, artist + "_performance")
            
            # Store test lyrics in json file in order to prevent training again
            test_lyrics = {}
            test_trimmed_lyrics = []
            for i in range(len(lyrics_dataset.dataset['test']['lyrics'])):
                test_trimmed_lyrics.extend([str(lyrics_dataset.dataset['test']['lyrics'][i])])
            test_lyrics['test_trimmed_lyrics'] = test_trimmed_lyrics
            test_lyrics['test_true_lyrics'] = lyrics_dataset.true_lyrics_dataset
            with open("./models/" + artist + "_performance/lyrics_test.json","w") as f:
                json.dump(test_lyrics, f)

        performance_data = compute_bleu_metric(artist, artist_generation)

    elif(args.genre_performance):
        print("Selected genre performance evaluation")
        genre = args.genre_performance[0]
        train = args.genre_performance[1]
        n_verses = int(args.genre_performance[2])

        # Train model
        if(train == 'True'):
            lyrics_dataset = LyricsDataset(config, genre.replace(" ", "_"), "79-musical-genres", performance_evaluation_nverses=n_verses)
            lyrics_dataset.load_dataset_multiple_artists()
            # tokenized_dataset = lyrics_dataset.dataset.map(
            #     lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
            # )
            # train_model(lyrics_dataset, tokenized_dataset, genre.replace(" ", "_") + "_performance")

        pass
        # TODO



