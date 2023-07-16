import json
import wandb
import statistics
from generator import LyricsGeneratorParams, LyricsGenerator
from nltk.translate.bleu_score import sentence_bleu

class BleuPerformanceParams:
    total_eval_datasets = 0
    prompt_words = []
    reference_words = []
    generated_words = []
    bleu_scores = []

def generate_performance_prompt(test_data):
    """
    Selects last n sentences as prompt.

    Args:
        test_data (Dictionary):
            Dictonary contaning the array of data to be used as prompt.
    Returns:
        prompt_texts (Array):
            Array of filtered performance prompts.
    """
    prompt_texts=[]
    for i in range(len(test_data['test_trimmed_lyrics'])):
        split_dataset = test_data['test_trimmed_lyrics'][i].split('\n')
        # Remove empty words
        while("" in split_dataset): split_dataset.remove("")
        # Join dataset
        prompt_texts.append('\n'.join(split_dataset))
    return prompt_texts

def word_counter(data):
    """
    Counts the amount of words present in data.

    Args:
        data (`str`):
            String of data
    Returns:
        n_words: (`int`):
            Amount of words present in data 
    """
    n_words = 0
    split_dataset = [i.split() for i in data.split('\n')]
    
    for i in range(len(split_dataset)):
        n_words += len(split_dataset[i])
    return n_words

def remove_last_words(data, n):
    """
    Removes the last n words from data.
    
    Args:
        data (`str`):
            String of data
    Returns:
        n: (`int`):
            Amount of words to remove from data 
    """
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

def compute_bleu_metric(config, n_words, filter, filter_generation=None):
    """
    Computes bleu evaluation metric related to a selected trained model. It uses this model to generate some text and 
    compare it against the real text. Maximum bleu score is 1 and minimum is 0.

    Args:
        config:
            The training model configuration parameters. Essentialy used to select the generator model.
        n_words (`int`):
            Number of words removed from the test set before training which will be compared with the generated ones.
        filter (`str`):
            Indicates which model to load in order to perform the metric calculation to.
        filter_generation (`str`, *optional*):
            If set will indicate that the selected model is the multiArtists one. It is used for its specific generation.
    Returns:
        performance_data (`BleuPerformanceParams`):
            Returns the computed metric for each test set.
    """
    # Load stored test lyrics information to avoid training again
    f = open("./models/" + filter.replace(" ", "_") + "_performance/lyrics_test.json")
    test_data=json.load(f)

    # Create table to store results
    table = wandb.Table(columns=["prompt", "reference", "generation", "bleu score"])

    # generate performance prompt
    initial_prompt = generate_performance_prompt(test_data)

    # perform lyrics generation 
    lyrics_generator_params = LyricsGeneratorParams
    lyrics_generator_params.num_sequences = 1
    lyrics_generator_params.max_length = 1024
    lyrics_generator_params.min_length = 1024
    # lyrics_generator_params.temperature = 1
    # lyrics_generator_params.top_p = 0.8
    lyrics_generator = LyricsGenerator(config, filter + "_performance", lyrics_generator_params, True)

    # Perform text generation and compute bleu scores
    performance_data = BleuPerformanceParams
    performance_data.total_eval_datasets = 0
    for i in range(0,len(test_data['test_trimmed_lyrics'])):
        if(filter_generation):
            # modify prompt for multiArtists generation
            initial_prompt[i] = filter_generation + ': ' + initial_prompt[i]
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
        if( true_lyrics_words != n_words):
            print("Sequence could not be generated: true_lyrics length was too short")
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

        # Log performance data
        table.add_data(initial_prompt[i], true_lyrics, generated_lyrics,
            "Number of prompt words    = {:.2f}".format(performance_data.prompt_words[-1]) + "\n" + 
            "Number of reference words = {:.2f}".format(performance_data.reference_words[-1]) + "\n" + 
            "Number of generated words = {:.2f}".format(performance_data.generated_words[-1]) + "\n" +
            "Computed bleu score       = {:.2f}".format(performance_data.bleu_scores[-1])
            )

    # Display final results
    print("\n" + "&"*20 + " General statistics " + "&"*20 + "\n")
    if(performance_data.total_eval_datasets > 0):
        print("Total number of evaluated datasets = ", performance_data.total_eval_datasets)
        print("Average number of prompt words     = {:.2f}".format(statistics.mean(performance_data.prompt_words)))
        print("Average number of reference words  = {:.2f}".format(statistics.mean(performance_data.reference_words)))
        print("Average number of generated words  = {:.2f}".format(statistics.mean(performance_data.generated_words)))
        print("Average bleu score                 = {:.2f}".format(statistics.mean(performance_data.bleu_scores)))
        print("\n" + "&"*60 + "\n")

    wandb.log({filter.replace(" ", "_") + "_performance": table})
    wandb.finish()

    return performance_data