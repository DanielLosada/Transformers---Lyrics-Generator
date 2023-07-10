import argparse
import torch
import json
import os
import statistics
from pynvml import *
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from nltk.translate.bleu_score import sentence_bleu
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

def train_model(dataset, tokenized_dataset, save_name=''):
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
    if(save_name):
        trainer.save_model("./models/" + save_name.replace(" ", "_"))

### POST-PROCESSING FUNCTIONS ####
def generate_performance_prompt(test_data, n_verses):
    '''Selects last n sentences as prompt'''
    prompt_texts=[]
    for i in range(len(test_data['test_trimmed_lyrics'])):
        split_dataset = test_data['test_trimmed_lyrics'][i].split('\n')
        # remove empty words
        while("" in split_dataset): split_dataset.remove("")
        
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
        n_words+= len(split_dataset[i])
    return n_words

def remove_last_words(data, n):
    '''removes the last n words from data'''
    split_data = [i.split() for i in data.split('\n')]
    word_count=0
    for j in range(0,len(split_data)):
        if split_data[-j] != '[]':
            for k in reversed(range(len(split_data[-j]))):
                if word_count == n:
                    break
                else:
                    print("removed word count: ", word_count, " removed word: ", split_data[-j][k])
                    split_data[-j].pop(k)
                    word_count+=1
            else:
                continue
            break
    data = '\n'.join(' '.join(v) for v in split_data)
    return data

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
    parser.add_argument("-ds", "--datasetSelection", dest='dataset_selection', choices=["genious-lyrics","79-musical-genres"], help="Offers dataset selection between two choices")
    
    parser.add_argument("-sp", "--singleArtistPerformance", nargs=3, dest='single_artist_performance', help="Computes the bleu metric to evaluate the single-artist model")
    parser.add_argument("-mp", "--multipleArtistsPerformance", dest='multiple_artists_performance', help="Computes specified metric to evaluate the specified model")
    parser.add_argument("-gp", "--genrePerformance", dest='genre_performance', help="Computes specified metric to evaluate the specified model")

    args = parser.parse_args()

    # Set default arguments
    if args.dataset_selection == None:
        args.dataset_selection="genious-lyrics"

    # TODO: remove this
    # args.single_artist_performance=['50 Cent', 'False', '1']

    # Training options
    if(args.train_single_artist):
        print("Selected single-artist training: ", args.train_single_artist)
        lyrics_dataset = LyricsDataset(config, args.train_single_artist, args.dataset_selection, args.performance_evaluation)
        lyrics_dataset.load_dataset_single_artist()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, args.train_single_artist)
    elif(args.train_multiple_artists):
        print("Selected multi-artist tranining")
        lyrics_dataset = LyricsDataset(config, "multipleArtists", args.dataset_selection, args.performance_evaluation)
        lyrics_dataset.load_dataset_multiple_artists()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, "multipleArtists")
    elif(args.train_genre):
        print("Selected genre tranining")
        lyrics_dataset = LyricsDataset(config, args.train_genre, "79-musical-genres", args.performance_evaluation)
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
        artist = args.single_artist_performance[0]+'_performance'
        train = args.single_artist_performance[1]
        n_verses = int(args.single_artist_performance[2])

        if(train == 'True'):
            lyrics_dataset = LyricsDataset(config, artist.replace(" ", "_"), "genious-lyrics", performance_evaluation_nverses=n_verses)
            lyrics_dataset.load_dataset_single_artist()
            tokenized_dataset = lyrics_dataset.dataset.map(
                lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
            )
            train_model(lyrics_dataset, tokenized_dataset, artist)

            # Store test lyrics in json file in order to prevent training again
            test_lyrics = {}
            test_trimmed_lyrics = []
            for i in range(len(lyrics_dataset.dataset['test']['lyrics'])):
                test_trimmed_lyrics.extend([str(lyrics_dataset.dataset['test']['lyrics'][i])])
            test_lyrics['test_trimmed_lyrics'] = test_trimmed_lyrics
            test_lyrics['test_true_lyrics'] = lyrics_dataset.true_lyrics_dataset
            with open("./models/" + artist.replace(" ", "_")+"/lyrics_test.json","w") as f:
                json.dump(test_lyrics, f)

        # load test lyrics information for performing text generation
        f = open("./models/" + artist.replace(" ", "_")+"/lyrics_test.json")
        test_data=json.load(f)

        # generate performance prompt
        initial_prompt = generate_performance_prompt(test_data, n_verses=1)

        # perform lyrics generation 
        lyrics_generator_params = LyricsGeneratorParams
        lyrics_generator_params.num_sequences = 1
        lyrics_generator = LyricsGenerator(config, artist, lyrics_generator_params)

        # Perform generation and compute scores
        bleu_scores=[]
        for i in range(0, len(test_data['test_trimmed_lyrics'])):
            initial_prompt_words = word_counter(initial_prompt[i])
            true_lyrics_words = word_counter(test_data['test_true_lyrics'][i])

            # create sentence with latest n words
            lyrics_generator.params.max_length = (initial_prompt_words + true_lyrics_words) + 10
            lyrics_generator.params.min_length = (initial_prompt_words + true_lyrics_words) + 10
            print("################################# iteration num = ", str(i))
            print("generator selected max_length =", lyrics_generator.params.max_length)
            print("generator selected min_length =", lyrics_generator.params.min_length)
            
            lyrics_generator.generate_lyrics(initial_prompt=initial_prompt[i])
            print("\n"+"&"*20 + " Initial prompt " + "&"*20)
            print(initial_prompt[i])
            print("\n"+"&"*20 + " True text " + "&"*20)
            print(initial_prompt[i] + ' ' + test_data['test_true_lyrics'][i])
            print("\n"+"&"*20 + " Generated text " + "&"*20)
            for text in lyrics_generator.generated:
                print(text)
            print("\n" + "&"*20)
            
            # remove prompt from generated output
            true_lyrics = test_data['test_true_lyrics'][i]
            generated_lyrics = lyrics_generator.generated[0].replace(initial_prompt[i], '')
            generated_words = word_counter(generated_lyrics)
            
            print("\n" + "&"*20)
            print("prompt words      =", initial_prompt_words)
            print("true_lyrics words =", true_lyrics_words)
            print("generated words   =", generated_words)
            print("\n" + "&"*20)

            # TODO: timming commented for now
            # trim generated or true_lyrics if lengths are not compatible
            # if(generated_words < true_lyrics_words):
            #     true_lyrics = remove_last_words(true_lyrics, abs(true_lyrics_words-generated_words))
            #     true_lyrics_words = word_counter(true_lyrics)
            #     print("\n" + "&"*20)
            #     print("prompt words      =", initial_prompt_words)
            #     print("true_lyrics words =", true_lyrics_words)
            #     print("generated words   =", generated_words)
            #     print("\n" + "&"*20)
            #     print("\n"+"&"*20 + " Initial prompt " + "&"*20)
            #     print(initial_prompt[i])
            #     print("\n"+"&"*20 + " True text " + "&"*20)
            #     print(test_data['test_true_lyrics'][i])
            #     print("\n"+"&"*20 + " Generated text " + "&"*20)
            #     print(generated_lyrics)
            #     print("\n" + "&"*20)
            # elif(generated_words > true_lyrics_words):
            #     generated_lyrics = remove_last_words(generated_lyrics, abs(true_lyrics_words-generated_words))
            #     generated_words = word_counter(generated_lyrics)
            #     print("\n" + "&"*20)
            #     print("prompt words      =", initial_prompt_words)
            #     print("true_lyrics words =", true_lyrics_words)
            #     print("generated words   =", generated_words)
            #     print("\n" + "&"*20)
            #     print("\n"+"&"*20 + " Initial prompt " + "&"*20)
            #     print(initial_prompt[i])
            #     print("\n"+"&"*20 + " True text " + "&"*20)
            #     print(test_data['test_true_lyrics'][i])
            #     print("\n"+"&"*20 + " Generated text " + "&"*20)
            #     print(generated_lyrics)
            #     print("\n" + "&"*20)

            # TODO: make a list of words instead of a string
            # Compute and display bleu metric
            # scores=[]
            # for j in range(len(test_data['test_true_lyrics'])):
            # reference = [i.split() for i in true_lyrics.split('\n')]
            # candidate = generated_lyrics
            # candidate = reference
            # scores.append(sentence_bleu('hello world!!', 'hello world!!'))

            # bleu_scores.append(statistics.mean(scores))
            # print("\n" + "&"*20)
            # print("bleu scores   =", bleu_scores[i])
            # print("\n" + "&"*20)

    elif(args.multiple_artists_performance):
        print("Selected multiple artists performance evaluation")
        pass
        # TODO
    elif(args.genre_performance):
        print("Selected genre performance evaluation")
        pass
        # TODO



