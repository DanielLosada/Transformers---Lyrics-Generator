from torch import device, cuda
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load device to use eith GPU or CPU
device = device("cuda") if cuda.is_available() else device("cpu")

@dataclass
class LyricsGeneratorParams:
    num_sequences: int = 3
    min_length: int = 1
    max_length: int = 5
    temperature: float = 1.5
    top_p: float = 0.9
    top_k: int = 30
    repetition_penalty: float = 1.0
    max_repeat: int = 2

class LyricsGenerator():
    def __init__(self, config, artist, params=LyricsGeneratorParams):
        self.config = config
        self.artist = artist
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.params = params
        self.generated=""
    
    def generate_lyrics(self, initial_prompt):
        """Generates lyrics text using trained model from /models/ and specified generator parameters"""
        encoded_prompt = self.tokenizer(initial_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        encoded_prompt.to(device)
        print("encoded_prompt: ", encoded_prompt)
        print("Loading model")
        model = AutoModelForCausalLM.from_pretrained("./models/" + self.artist.replace(" ", "_") + "/")
        output_sequences = model.generate(
                        input_ids=encoded_prompt,
                        max_length=self.params.max_length,
                        min_length=self.params.min_length,
                        temperature=self.params.temperature,
                        top_p=self.params.top_p,
                        top_k=self.params.top_k,
                        do_sample=True,
                        repetition_penalty=self.params.repetition_penalty,
                        num_return_sequences=self.params.num_sequences)
        print("output_sequences: ", output_sequences)
        self.generated = self.__post_process(output_sequences)
        print("&"*50)
        for text in self.generated:
            print(text)
        print("&"*50)
        
    def __post_process(self, output_sequences):
        """Decodes lyrics text from tokenizer and cleansup text"""
        predictions = []
        generated_sequences = []

        # decode prediction
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            generated_sequences.append(text.strip())
        for i, g in enumerate(generated_sequences):
            res = str(g).replace('\n\n\n', '\n').replace('\n\n', '\n')
            lines = res.split('\n')
            #print(lines)
            lines = self.__remove_consecutive_duplicates(lines, self.params.max_repeat)
            predictions.append('\n'.join(lines))

        return predictions
    
    def __remove_consecutive_duplicates(self, arr, max_repeat):
        """Removes consecutive duplicated words"""
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
                    results.append(arr[x])
            return results
        else:
            return arr


