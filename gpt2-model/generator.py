import wandb
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch import device, cuda
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load device to use eith GPU or CPU 
device = device("cuda") if cuda.is_available() else device("cpu")

@dataclass
class LyricsGeneratorParams:
    num_sequences: int = 3
    min_length: int = 100
    max_length: int = 500
    temperature: float = 1
    top_p: float = 1
    top_k: int = 70
    max_repeat: int = 2

class LyricsGenerator():
    def __init__(self, config, artist, params=LyricsGeneratorParams, pretrained=True):
        self.config = config
        self.artist = artist
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.params = params
        self.pretrained = pretrained
        self.generated=""
    
    def generate_lyrics(self, initial_prompt, table_name = "default table name", condition = ""):
        """Generates lyrics text using trained model from /models/ and specified generator parameters"""
        encoded_prompt = self.tokenizer(initial_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        encoded_prompt.to(device)

        attention_mask = encoded_prompt.clone().fill_(1)

        print("encoded_prompt: ", encoded_prompt)
        print("attention_mask: ", attention_mask)
        print("table_name: ", table_name)
        print("Loading generation model with params...")
        print("*"*50)
        print('initial_prompt:     ', initial_prompt)
        print('max length:         ', self.params.max_length)
        print('min length:         ', self.params.min_length)
        print('num sequences:      ', self.params.num_sequences)
        print('temperature:        ', self.params.temperature)
        print('top k:              ', self.params.top_k)
        print('top p:              ', self.params.top_p)
        print("*"*50)

        # Create table to store results
        table = wandb.Table(columns=["prompt", "generation"])

        if(self.pretrained):
            print("Selected gpt2 pre-trained model: ", "./models/" + self.artist.replace(" ", "_") + "/")
            model = AutoModelForCausalLM.from_pretrained("./models/" + self.artist.replace(" ", "_") + "/")
        else:
            print("Selected gpt2 raw model")
            model = AutoModelForCausalLM.from_pretrained(self.config["model"])
        output_sequences = model.generate(
                        input_ids=encoded_prompt,
                        attention_mask=attention_mask,
                        max_length=self.params.max_length,
                        min_length=self.params.min_length,
                        temperature=self.params.temperature,
                        top_p=self.params.top_p,
                        top_k=self.params.top_k,
                        do_sample=True,
                        num_return_sequences=self.params.num_sequences)
        self.generated = self.__post_process(output_sequences, condition)
        print("generated: ", self.generated)
        for generation in self.generated:
            # Log performance data
            table.add_data(initial_prompt, generation)

        wandb.log({table_name: table})
        wandb.finish()

    def compute_cosine_similarity(self, X, Y):
        # tokenization
        X_list = word_tokenize(X) 
        Y_list = word_tokenize(Y)
        # sw contains the list of stopwords
        sw = stopwords.words('english') 
        l1 =[];l2 =[]
        # remove stop words from the string
        X_set = {w for w in X_list if not w in sw} 
        Y_set = {w for w in Y_list if not w in sw}
        # form a set containing keywords of both strings 
        rvector = X_set.union(Y_set) 
        for w in rvector:
            if w in X_set: l1.append(1) # create a vector
            else: l1.append(0)
            if w in Y_set: l2.append(1)
            else: l2.append(0)
        c = 0
        # cosine formula 
        for i in range(len(rvector)):
                c+= l1[i]*l2[i]

        cosine = 0
        if float((sum(l1)*sum(l2))**0.5) > 0:
            cosine = c / float((sum(l1)*sum(l2))**0.5)
        return cosine
  
        
        
    def __remove_consecutive_duplicates_using_cosine_similarity(self, arr, threshold):
        """Removes consecutive duplicated words"""
        results = []
        #print("%"*50)
        #print("previous_line before deleting duplicates: ", arr[0])
        previous_line = ' '.join(self.__remove_consecutive_duplicates(arr[0].split(' '),self.params.max_repeat, True))
        #print("previous_line AFTER deleting duplicates: ", previous_line)
        results.append(previous_line)
        for x in range(1, len(arr)):
            #print("current_line before deleting duplicates: ", arr[x])
            current_line = ' '.join(self.__remove_consecutive_duplicates(arr[x].split(' '),self.params.max_repeat, True))
            #print("current_line AFTER deleting duplicates: ", current_line)
            #print("%"*50)
            #if(self.compute_cosine_similarity(previous_line, current_line) > threshold):
            #    print("&"*50)
            #    print("previous_line: ", previous_line)
            #    print("current_line: ", current_line)
            #    print("self.compute_cosine_similarity(previous_line, current_line): ", self.compute_cosine_similarity(previous_line, current_line))
            #    print("&"*50)
            if self.compute_cosine_similarity(previous_line, current_line) < threshold:
                results.append(current_line)
                previous_line = current_line
        #print("results: ", results)
        return results

    def __post_process(self, output_sequences, condition):
        """Decodes lyrics text from tokenizer and cleansup text"""
        predictions = []
        generated_sequences = []

        # decode prediction
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            generated_sequences.append(text.replace(condition + ':', "").strip())
        for i, g in enumerate(generated_sequences):
            res = str(g).replace('\n\n\n', '\n').replace('\n\n', '\n')
            lines = res.split('\n')
            #print(lines)
            lines = self.__remove_consecutive_duplicates_using_cosine_similarity(lines, 0.80)
            predictions.append('\n'.join(lines))
        return predictions
    
    def __remove_consecutive_duplicates(self, arr, max_repeat, words = False):
        """Removes consecutive duplicated words"""
        #print("&"*50)
        #print("initial_arr: ", arr)
        results = []
        if len(arr) >= max_repeat:
            current_word = arr[0]
            results.append(arr[0])
            count = 1
            for x in range(1, len(arr)):
                if arr[x] != current_word:
                    current_word = arr[x]
                    count = 0
                if current_word == arr[x]:
                    count += 1
                if max_repeat >= count:
                    if not words:
                        duplicated_words_removed = ' '.join(self.__remove_consecutive_duplicates(arr[x].split(' '),self.params.max_repeat, True))
                        results.append(duplicated_words_removed)
                    else:
                        results.append(arr[x])
                #else:
                    #print("word removed: ", arr[x], "on line ", arr)
            #print("results: ", results)
            #print("&"*50)
            return results
        else:
            return arr


