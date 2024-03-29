# Transformers - Lyrics-Generator

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

## Configuration

General configurations can be found within the ```gpt2-model/config.json``` file. This file indicates where the zipped dataset files are located and where to uncompress them among other tranining hyperparameters. Please make sure that your compressed datasets are located and called as the configuration file indicates.

## Run
Train single-artist model genius-lyrics dataset (default):
```
python main.py --trainSingleArtist "50 Cent"
```

Train single-artist model 79-musical-genres dataset:
```
python main.py --trainSingleArtist "50 Cent" --datasetSelection "79-musical-genres"
```

Train multiple-artist conditional model (only genius-dataset supported):
```
python main.py --trainMultipleArtists
```

Train genre model (only 79-musical-genres dataset supported):
```
python main.py --trainGenre "Rock"
```

Generate single-artist model lyrics:
```
python main.py --generateSingleArtist "50 Cent" "You are"
```

Generate multiple-artist conditional model lyrics:
```
python main.py --generateMultipleArtists "Taylor Swift" "You are"
```

Generate genre model lyrics:
```
python main.py --generateGenre "Rock" "You are"
```

Compute single-artist preformance BLEU score (with training enabled and 20 last words from test dataset removal):
```
python main.py --performanceSingleArtist "Taylor Swift" "True" 20
```

Compute single-artist preformance BLEU score (without training and 20 last words from test dataset removal):
```
python main.py --performanceSingleArtist "Taylor Swift" "False" 20
```

Compute single-artist preformance PPL score:
```
python main.py --performanceSingleArtist "Taylor Swift" "True" -1
```

Compute multiple-artist preformance BLEU score (with training enabled and 20 last words from test dataset removal):
```
python main.py --performanceMultipleArtists "Eminem" "True" 20
```

Compute multiple-artist preformance PPL score:
```
python main.py --performanceMultipleArtists "Eminem" "True" -1
```

Compute genre preformance BLEU score (with training enabled and 20 last words from test dataset removal):
```
python main.py --performanceMultipleArtists "Reggae" "True" 20
```

Compute genre preformance PPL score:
```
python main.py --performanceMultipleArtists "Reggae" "True" -1
```


# Lyrics Generator

Final project for the [UPC Postgraduate Course Artificial Intelligence with Deep Learning](https://www.talent.upc.edu/ing/estudis/formacio/curs/310400/postgraduate-course-artificial-intelligence-deep-learning/), edition Spring 2023

Team: Daniel Losada Molina, Pau Rosell Civit, Svetlana Kazakova

Advisor: Daniel Fojo

GitHub repository: [https://github.com/DanielLosada/Transformers---Lyrics-Generator](https://github.com/DanielLosada/Transformers---Lyrics-Generator)

## Table of Contents <a name="toc"></a>
1. [Introduction](#intro)
    1. [Motivation](#motivation)
    2. [Project Goals](#goals)
    3. [Milestones](#milestones)
2. [Data Set](#dataset)
3. [Working Environment](#working_env)
4. [General Architecture and implementation](#architecture)
5. [Preprocessing the data set](#dataset_preprocess)
6. [Postprocessing the generated text](#generation_postprocessing)
7. [Results](#results)
    1. [Experiment 1: Single-artist training and generation with same prompt](#experiment_1)
    2. [Experiment 2: Specific genre training and generation with same prompt](#experiment_2)
    3. [Experiment 3: Conditional lyrics generation with the same prompt](#experiment_3)
    4. [Experiment 4: Conditional lyrics generation based on the prompt](#experiment_4)
    5. [Experiment 5: Performance evaluation single-artist GPT-2 vs fine tuned GPT-2](#experiment_5)
    6. [Experiment 6: Performance evaluation single-artist different datasets](#experiment_6)
    7. [Experiment 7: T5 model](#experiment_7)
8. [Conclusions](#conclusions)
9. [Next Steps](#next_steps)
10. [References](#references)

## 1. Introduction <a name="intro"></a>
Lyrics generation, the task of automatically generating song lyrics using deep learning techniques, has gained significant attention in recent years. With the advancements in natural language processing and deep learning, generating creative and coherent lyrics has become an intriguing but still a challenging task. This project aims to explore and address these challenges by leveraging state-of-the-art deep learning models and fine-tuning them on suitable datasets. 

By analyzing the generated lyrics' quality, we can gain insights into the potential and limitations of deep learning models in the realm of lyrics generation.
<p align="right"><a href="#toc">To top</a></p>

### 1.1 Motivation <a name="motivation"></a>
Our motivation for this project is driven by two main factors: the desire to explore cutting-edge technologies and the fascination with the creative possibilities of lyrics generation. LLMs have shown impressive abilities in understanding and generating human-like text. By working on lyrics generation, we aim to dive deeper into these technologies and understand their potential for creative text generation.
<p align="right"><a href="#toc">To top</a></p>

### 1.2 Project Goals <a name="goals"></a>
* Attempt to generate lyrics with GPT-2 and T5 based models
* Analysis of the results
* Suggestions for further improvement
<p align="right"><a href="#toc">To top</a></p>

### 1.3 Milestones <a name="milestones"></a>
We have established the following key milestones:
* Do a general research on the subject
* Find suitable data sets
* Define the main model architecture
* Preprocess the data and implement the model
* Train the model
* Analyse the obtained results and postprocess
* Try a different model architecture (optional)
* Make suggestions for further improvement
<p align="right"><a href="#toc">To top</a></p>

## 2. Data Set <a name="dataset"></a>
There are three primary methods for acquiring a dataset:

1. Extracting data from websites using tools like BeautifulSoup or Scrapy.
2. Utilizing the Genius API to gather data.
3. Accessing pre-existing datasets from platforms like Kaggle or other similar sources.

After trying all of them, we have opted for using datasets from Kaggle.
We have chosen the following 2 datasets:
* https://www.kaggle.com/datasets/mervedin/genius-lyrics
* https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres

The first dataset (Genius lyrics) contains 81 artist's song lyrics labeled by their filename. There are 100 songs for each artist.

The second dataset, obtained through web scraping, encompasses song lyrics from 79 different musical genres. It includes lyrics from numerous artists along with supplementary information such as the genre and popularity of each artist. The dataset is composed of two CSV files: "artists-data.csv" and "lyrics-data.csv".

The "lyrics-data.csv" file contains data on a substantial number of song lyrics, totaling 379,893 entries, spanning across 4,239 artists. Each entry includes details such as the artist name, song name, lyrics content, and the language in which the lyrics are written.

The "artists-data.csv" file provides information about 4,168 artists featured in the dataset. It includes additional details like the genre associated with each artist, their popularity, and the number of songs attributed to them.

Together, these CSV files form a comprehensive collection of song lyrics, artist information, and relevant metadata from a diverse range of musical genres.

<p align="right"><a href="#toc">To top</a></p>

## 3. Working Environment <a name="working_env"></a>
[PyTorch](https://pytorch.org/) is used as the main framework.
We started out in [Google Colab](https://colab.research.google.com/) as it was fast and easy for us to access. Then the intention was to pass on to training via a VM instance on [Google Cloud](https://cloud.google.com/) but since we had a problem with access to GPUs on Google Cloud, we couldn't complete our training there. In the end we were training only locally, we had a limited access to GPU RTX 1080 Ti which we used for more time consuming experiments (training), and for faster experiments (generation) we proceeded without GPU. To view the results we used [wandb](https://wandb.ai/site). 

<p align="middle">
  <a href="https://drive.google.com/uc?export=view&id=1Bo-HNWlYIK75T5-wHaX_YLHGheVev2E_">
    <img src="https://drive.google.com/uc?export=view&id=1Bo-HNWlYIK75T5-wHaX_YLHGheVev2E_" alt="Image" style="width: auto; max-width: 50%; height: 80px; display: inline-block;" title="Image" />
  </a>
  
  <a href="https://drive.google.com/uc?export=view&id=1N2ui7rYVl6WPUAgzuMgFe7TU2c_MGm56">
    <img src="https://drive.google.com/uc?export=view&id=1N2ui7rYVl6WPUAgzuMgFe7TU2c_MGm56" alt="Image" style="width: auto; max-width: 50%; height: 80px; display: inline-block;" title="Image" />
  </a>
  
  <a href="https://drive.google.com/uc?export=view&id=1gq6dYc2tmIJV2bvIZDrq2TaokTSYYm-j">
    <img src="https://drive.google.com/uc?export=view&id=1gq6dYc2tmIJV2bvIZDrq2TaokTSYYm-j" alt="Image" style="width: auto; max-width: 50%; height: 80px; display: inline-block;" title="Image" />
  </a>
</p>
<p align="right"><a href="#toc">To top</a></p>

## 4. General Architecture and implementation <a name="architecture"></a>
The development of advanced language models has brought significant changes to tasks like lyrics generation in natural language processing. These models, based on transformer architectures, have shown impressive skills in understanding and creating meaningful text that makes sense in different contexts. GPT, one of these models, has received a lot of attention because of its outstanding performance and flexibility. We have chosen to utilize GPT-2, which is the most recent version of the GPT models accessible on the Hugging Face platform.

GPT-2 consists of solely stacked decoder blocks from the transformer architecture. This architecture allows GPT-2 to effectively capture the relationships between words and generate coherent and contextually relevant text.

The GPT-2 model was trained on a large corpus of text data that consisted of approximately 40 gigabytes of text (around 8 million tokens). The model has 1.5 billion parameters.

<p align="left">
  <a href="https://drive.google.com/uc?export=view&id=1ywV_aKn0qnO4IR9Bxmwqxw2Gt7lyehcg">
    <img src="https://drive.google.com/uc?export=view&id=1ywV_aKn0qnO4IR9Bxmwqxw2Gt7lyehcg" alt="Image" style="width: 800px; height: auto; display: inline-block;" title="Image" />
  </a>
</p>

For implementation we used the Hugging Face Transformers library. We initialized the language model using AutoModelForCausalLM. For tokenizing we utilized the AutoTokenizer class from the transformers library, which automatically selects the tokenizer associated with the specific pre-trained model you are using, ensuring compatibility between the model and tokenizer. 

For training we created a data collator specifically for language modeling training using the DataCollatorForLanguageModeling class from the transformers library. The data collator is responsible for batching and preparing the input data during training.
Then we instantiated the Trainer object from the same library with the model, training arguments, and datasets.
The trainer trains the model using the trainer.train() method.
<p align="right"><a href="#toc">To top</a></p>

## 5. Preprocessing the data set <a name="dataset_preprocess"></a>
Overall, the preprocessing steps involve:

* extracting the dataset
* removing non-English authors to ensure language consistency
* cleaning and formatting the lyrics data to eliminate unwanted artifacts
* tokenizing the datasets for further processing, setting a maximum context length (maximum context length for GPT-2 model is 1024 tokens but we use 512 due to limitations in computational resources)
<p align="right"><a href="#toc">To top</a></p>

## 6. Postprocessing the generated text <a name="generation_postprocessing"></a>

  After generating the text with the trained model, we postprocessed the results.
  * Removing unnecessary line breaks: In some cases, the model generated consecutive line breaks ("\n"), which were not required. We removed these redundant line breaks to improve the readability of the generated text.
  * Eliminating word repetition: Occasionally, the model produced consecutive occurrences of the same word. To maintain naturalness while reducing excessive repetition, we allowed only two consecutive equal words. If there were more than two consecutive equal words, the additional repetitions were removed.
  * Addressing sentence repetition: The model sometimes generated two sentences that were identical or very similar. To ensure greater variety and avoid redundancy, we computed the cosine similarity between consecutive sentences. If the cosine similarity between two consecutive sentences exceeded a threshold of 0.8, we retained only one of them.

## 7. Results <a name="results"></a>
    
### 7.1 Experiment 1: Single-artist training and generation with same prompt <a name="experiment_1"></a>


**Training on Genius Lyrics dataset** https://www.kaggle.com/datasets/mervedin/genius-lyrics

Experiment setup: We conducted fine-tuning of GPT-2 using a dataset of 100 lyrics from a single artist. The training took place in a local environment. During the generation phase, we employed the same input prompt for all the tested artists. To avoid excessively conditioning the model, we utilized a concise input prompt. Our objective was to observe the model's behavior when provided with a limited input prompt for each tested artist. The training and testing were performed on six artists: 50 Cent, Taylor Swift, The Beatles, Queen, Justin Bieber, and Imagine Dragons.

Hypothesis: Despite the utilization of only 100 songs, we anticipate that the model will successfully adapt to the vocabulary and style of each artist, allowing it to extend the given input prompt in the artist's specific style. However, in terms of training, overfitting might be expected.

Resuts: 
The full performance results can be seen in the W&B report: https://wandb.ai//upcproject/Lyrics%20Generator/reports/Single-artist-generated-lyrics-prompt-You-are---Vmlldzo0OTEyNTA2?accessToken=afvobacengbowa8wyvxxhnqvju9x60r0egsa52n7qvjpcnvtveo1597f3ncpp1pu
 Here we are going to show some examples of the results obtained.
* Input prompt: "You are"
    * Artist: 50 Cent
     ```
    I'm not a person with money, I don't care if you gave me stuff, I don't care if you went out and stolen
    I'm not a person with feelings, I'm a person that likes to drink
    There's no way you were gonna give me crap
     ```

     ```
    I'm not giving a fuckin' man a fuck, I ain't get no respect
    But I did make my girl want sex, boy
     
     ```
    * Artist: Taylor Swift
    ```
    And I'm still crying now, when I see you in that restaurant
    There's a new tattoo in my yard
    And all of a sudden I'm feeling so bad about the way you look
    Now I think we are at peace
    ```

    ```
    Never think you're alone
    Never feel like you're alone
    You know I love you
    You know I've loved you
    ```
    * Artist: The Beatles

    ```
    Happy Independence Day
    Sing and dance to peace
    Love and happiness are your only hope
    Hear your friend sing, "The World Won't Stop Talking"
    ```

    In these examples, we can observe that for 50 Cent, the model captured the aggressive tone, discussing topics such as drugs, sex, and crime. When generating lyrics for Taylor Swift, the style drastically changes, focusing more on emotions, love, and using delicate language. For The Beatles, the model generates lyrics emphasizing a message of peace.

    Conclusions:
    Based on our findings, we can conclude that our hypothesis was correct. The model not only adapts the tone and vocabulary to match each artist's style but also adjusts the lyrical themes. We observed that the model successfully fits the vocabulary and style of each artist, allowing it to extend the given input prompt in the artist's specific style. This behavior remained consistent across various artists and input prompts.

    To explore further generations with different input prompts, I recommend referring to the W&B report at the following link: W&B Report: Single-artist Generated Lyrics.

   Regarding the training process, we observed that the loss decreases, although not rapidly. This can be attributed to the fact that the pre-trained GPT-2 model has already been trained on a substantial amount of data and possesses a strong capability to generate coherent English text.
  
   Link to W&B training report: https://api.wandb.ai/links/upcproject/xkw1ugdh
### 7.2 Experiment 2: Specific genre training and generation with same prompt <a name="experiment_2"></a>

Experiment setup: We conducted fine-tuning of GPT-2 using the second dataset filtered by genre. The dataset consisted of songs from different genres, and the number of songs available for each genre varied. This experiment aimed to explore and learn the patterns of genres more generally, rather than focusing solely on individual artists. To select the songs for our experiment, we implemented two criteria for dataset filtering:

* Genre matching: Within the dataset, each artist is associated with the genre or genres they perform. We specifically chose songs from artists who exclusively belong to the desired genre. This approach serves two purposes. Firstly, it ensures that we focus on songs that align with the target genre, preventing the inclusion of songs with different styles. Secondly, it allows us to manage the computational requirements of the training process, as including songs from artists with diverse genres would lead to an overwhelming amount of data.
* Artist popularity: We established a popularity threshold of 5 based on the "Popularity" column in the dataset. The popularity value is derived from website visits where the dataset was sourced. By setting this threshold, we ensure that we only include songs from artists who have attained a certain level of popularity. This step is crucial for ensuring the accuracy of the lyrics and genre information. We aim to work with data from more popular artists to increase confidence in the correctness of the provided information.

The number of songs used for each genre are as follows:

* Pop: 375
* Reggae: 223
* Rock: 954
* Hip Hop: 108

Hypothesis: We hypothesize that the model will be capable of adapting to the specific genres and generate new songs that align with the style of each genre.

Results:
The full results can be seen in the W&B report: https://wandb.ai//upcproject/Lyrics%20Generator/reports/Specific-genre-generated-lyrics-prompt-You-are---Vmlldzo0OTEyNjI1?accessToken=al65ot18qofx8shodfimvfybklq0g95q07gsjbgdj21228vq7e6y19wnsmxcy4la 
Here we are going to show some examples of the results obtained.
* Input prompt: "You are"
    * Genre: Reggae
    ```
    Oh will your mercy please, lord!
    But as God forgive us,
    We will be free, and free will be good Come take a walk and take your shoes off.
    ```

    * Pop: 
    ```
    If you try to resist my love then you are the one
    I'll leave you
    If you cannot see your eyes then you are the one
    I'll make you mad and leave you
    ```

    * Rock:

    ```
    You are the only one of your kind who has shown me how it's done
    You're the only one who can be thankful to me, who's saved my soul on the dark side, who still cares about you
    You're the only one who can love me, and you are not alone I'm sure you are
    And this is just my dream for you
    ```

    The results of this experiment align with our previous findings. The generated lyrics demonstrate a strong association with their respective genres, both in terms of topics and expression. For instance, reggae lyrics often revolve around Rastafari, a religion developed in Jamaica during the 1930s. Consequently, reggae songs frequently address themes of religion and the relationship between humanity and God. Additionally, freedom is a common topic in reggae, reflecting the historical context of slavery and the misalignment of society with Rastafarian values. The provided snippet of reggae-generated text exemplifies the combination of these themes. However, the model doesn't incorporate specific Rastafarian slang, possibly due to insufficient training examples to capture those nuances.

    Similarly, the generated lyrics for the "Pop" genre predominantly focus on love and heartbreak, while the "Rock" genre exhibits more complex vocabulary and sentence structures. On the other hand, the "Hip Hop" genre's style and specific vocabulary are not as distinctly represented, likely due to the limited number of songs used for training.

    Conclusion:

    In conclusion, our observations from this experiment reaffirm that the model is capable of fitting the genre, learning its style and tone. However, it appears to struggle with utilizing specific genre-related vocabulary. To explore additional generations with another input go to the W&B report: https://wandb.ai//upcproject/Lyrics%20Generator/reports/Specific-genre-generated-lyrics-prompt-I-will---Vmlldzo0OTEyNTYx?accessToken=9leuj9h6e51qzjqvomv3xgp4tf9tnrf1hsbb06r9mi623kqbavlf29k0bx6gwbxm

    Training resuts were siliar to those obtained with single-artist training.
    
    Link to W&B training report: https://api.wandb.ai/links/upcproject/opb8hkt3

<p align="right"><a href="#toc">To top</a></p>

### 7.3 Experiment 3: Conditional lyrics generation with the same prompt <a name="experiment_3"></a>
Experiment setup: For this experiment, we trained GPT-2 using the songs of ten artists from the first dataset. We ensured that we had 100 songs for each artist, providing a balanced dataset. The objective was to create a conditional model that allows users to choose the artist whose style they want the generated songs to emulate. To achieve this, we modified the preprocessing step by adding the name of the artist at the beginning of each lyric. During generation, we concatenated the desired artist's name before the initial prompt, enabling the model to understand the relationship between the lyrics and the artist.

The ten artists used in this experiment are as follows:
* 50 Cent
* Imagine Dragons
* Justin Bieber
* Taylor Swift
* Queen
* Lil Peep
* Arctic Monkeys
* The Notorious B.I.G.
* Radiohead
* Mac Miller

Hypothesis: We hypothesize that the model trained in this experiment will generate similar results to those obtained in the first experiment.

Results:
The full results can be seen in the W&B report. 
W&B link: https://wandb.ai//upcproject/Lyrics%20Generator/reports/Multiple-artists-generated-lyrics-prompt-You-are---Vmlldzo0OTEyNzQ3?accessToken=g3lev2vf26vg7jsg20q6a2qgmmq8faqczopt89e99f2gp1xprrjaznx87e3llqxk

Here we are going to show some examples of the results obtained.

* Input prompt: "You are"
  * Artist: The Notorious B.I.G.
  ```
  I'm so rich, that's what the world says I am
  Makin' shit, what happens when my money flows?
  Life's too short, the world is too fucked up
  I'm like, 'damn, what's the money?', the motherfuckin time
  You can die, kill yourself, man, you fucking faggot
  ```

  * Artist: Justin Bieber

  ```
  You are a young devil I've never seen before
  Don't you dare believe me
  I think I'm the man to blame
  Shake it up and replace it with a new one
  And put a new face on the story
  ```

  As observed, the model in this experiment demonstrates similar performance to the first experiment. It successfully captures the change in style and topics based on the requested artist, even when using the same input prompt. For example, the model transitions from the 'egotrip' and tough vocabulary of The Notorious B.I.G. to the heartbroken themes commonly associated with Justin Bieber. However, it is important to note that there is still room for improvement in terms of coherence in the generated lyrics. We had expected better results in this aspect, but limitations in computational resources prevented further fine-tuning of the model parameters or expanding the dataset.

  Conclusions:
  
  In conclusion, the model's capacity to accommodate multiple artists simultaneously and understand conditioned prompts is evident. To explore additional generations with different inputs, we recommend referring to the W&B report for more details and examples.

  W&B link: https://wandb.ai//upcproject/Lyrics%20Generator/reports/Multiple-artists-generated-lyrics-prompt-I-will---Vmlldzo0OTEyNjgw?accessToken=lnb3rb377mkatl0g5xk76ral4a4w7wk03wclk4dsvcvmxnmdg718hlqknmohwrs1
   
  Link to W&B training report: https://api.wandb.ai/links/upcproject/zzppte9f
  
  The training is quite stable with gradually decresing training loss and evaluation loss.

  <p align="right"><a href="#toc">To top</a></p>

### 7.4 Experiment 4: Conditional lyrics generation based on the prompt <a name="experiment_4"></a>

Experiment setup:

In this experiment, we trained GPT-2 using lyrics from the first dataset, focusing on the contributions of 50 Cent and Justin Bieber. Each artist provided 100 songs for training. Unlike the previous experiment, we did not explicitly condition the model by specifying the artist during the generation process. Instead, we utilized an initial prompt that was already embedded within the lyrics. This initial prompt served as a guiding input, indicating the desired style and acting as a guiding factor for the model. It is important to note that the initial prompt was integrated naturally into the lyrics and did not serve as explicit instructions for the model.

Hypothesis: 
Our hypothesis is that the model will be capable of discerning the intended style to follow based on the topic or style presented in the initial prompt. However, there is a possibility that the model may occasionally exhibit confusion and blend the two distinct styles, despite their inherent differences.

Results:
The full results can be seen in the W&B report: 
https://wandb.ai//upcproject/Lyrics%20Generator/reports/Conditional-lyrics-generation-based-on-prompt---Vmlldzo0OTEzNTIw?accessToken=92bum66dzakw9sk353xizq7tqy5el21xwxfq1onvgpzz97cxd0n7f782x5885y33

* Input prompt: I'm hustling

  ```
  Oh, man, I'm hustling
  I'm hustling, hustling, hustling my way through this shit
  Feels like you get up, right?
  Come on, yeah, yeahomp, come on yo, come on yo
  ```

* Input prompt: I miss you
  ```
  I miss you
  Lost your little girl
  Lost her heart, she miss me, no, no, no
  Don't know her name
  Lost everything
  I'm stuck, oh-oh
  Oh my niggas on the block and the cops
  Don't wanna talk, I'm stuck
  Oh my niggas on the block and the cops
  Don't wanna talk, I'm stuck
  I miss ya
  ```

  As observed, the model demonstrates the ability to adapt to different styles based on the input prompt provided. However, there are instances where the transition between styles may not be entirely coherent or consistent, as exemplified in the second example where the lyrics start with a Justin Bieber style but switch to a 50 Cent style. This inconsistency detracts from the overall coherence of the generated lyrics.

  Conclusion:
  In conclusion, the model shows the capability to fit both styles, despite their distinct differences, and generally knows when to employ the more appropriate style. However, there is still room for improvement in terms of maintaining consistency and coherence throughout the generated lyrics.

### 7.5 Experiment 5: Performance evaluation single-artist GPT-2 vs fine tuned GPT-2 <a name="experiment_5"></a>

**A. BLEU SCORE**

Experiment setup: In our evaluation process, we conduct a distinct training session. During evaluation, we provide the model with song lyrics as a prompt and remove the final 20 words from each song. The model generates the missing portion, and we utilize the BLEU score to assess the similarity between the generated lyrics and the truncated reference lyrics. By afterwards comparing the BLEU scores obtained from the model fine-tuned on our dataset and the model without fine-tuning, we can evaluate the quality of the generated lyrics.

Hypothesis: BLEU score should be higher in fine-tuned model output.

Results : BLEU score is computed for 5 artists using the genious-lyrics dataset: 50 Cent, Taylor Swift, Imagine Dragons, Queen and The Beatles. Each training set includes 80 songs and each evaluation set 20 songs. As example, the generation details of the highest BLEU score increase (from GPT-2 to fined-tuned GPT-2 model) is presented:



| _Artist_        | _Prompt (Last 26 words)_                                                                                                            | _True Lyrics (20 words)_                                                                                                                   | _GPT-2 model Generated lyrics  (20 words)_                                                     | _Fine-tuned GPT-2-model generated lyrics  (20 words)_                                                                       |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Imagine Dragons | (...) AHHHHH! I'm wakin' up, I feel it in my bones Enough to make my systems blow Welcome to the new age, to the new age Welcome to | the new age, to the new age Whoa, whoa, I'm radioactive, radioactive Whoa, whoa, I'm radioactive, radioactive I'm radioactive, radioactive | the new age, to the new age  What's that gonna take?  It's like getting off of the train to go | the new age, to the new age  Whoa, whoa, I'm radioactive, radioactive  Whoa, whoa, I'm radioactive, radioactive  I raise my |

In this particular example, BLEU scored 0.35 for the GPT-2 model and 0.85 for the fine tuned GPT-2 model. As it can be seen, this metric basically displays the simmilarity word by word between generations. Although is is a way to parammetrize our generations, other features such as context or story telling are not assessed by it. As a summary, mean scores for all conducted tests are presented in the following table:


| _Artist_        | _Total number test datasets_ | _Average prompt words_ | _BLEU score (GPT-2 model)_ | _BLEU score (fine-tuned GPT-2-model)_ |
|-----------------|------------------------------|------------------------|----------------------------|---------------------------------------|
| 50 Cent         | 14*                          | 523.21                 | 0.33                       | 0.40                                  |
| Taylor Swift    | 19*                          | 346.89                 | 0.22                       | 0.32                                  |
| Imagine Dragons | 20                           | 310.05                 | 0.30                       | 0.57                                  |
| Queen           | 20                           | 211.30                 | 0.17                       | 0.18                                  |
| The Beatles     | 20                           | 156.50                 | 0.24                       | 0.30                                  |

_*The amount of test sets is sometimes lower than 20 as some lyrics contained more than 1024 tokens. This is the maximum accepted for a single generation prompt in GPT-2 models. For later executions a timming factor has been added to the tokanizer to avoid bypassing larger prompts._


Conclusions : BLEU score is indeed higher in lyrics generated by the fine-tuned model, it suggests that the fine-tuned model has improved in generating lyrics that are more similar to the reference lyrics.
However, it's important to note that BLEU scores alone do not provide a complete assessment of the quality of the generated lyrics. BLEU primarily measures the overlap in n-grams and may not capture aspects such as creativity, coherence, or semantic understanding. Therefore, it's recommended to complement BLEU scores with other evaluation methods, including human evaluation, to obtain a comprehensive understanding of the fine-tuned model's performance in generating high-quality lyrics.

**B. PERPLEXITY**

Experiment setup: We also measure perplexity (Perplexity = 2^(- (1 / N) * Σ(log2(p(w_i))))). A lower perplexity value indicates that the model is more confident and has a better understanding of the data. Conversely, a higher perplexity value suggests that the model is less certain and has a harder time predicting the data. We compare the perplexity obtained from raw GPT-2 model and the GPT-2-based model fine-tuned on our dataset.

Hypothesis: Perplexity should be lower in fine-tuned model output.

Results : Perplexity score is computed for 5 artists using the genious-lyrics dataset: 50 Cent, Taylor Swift, Imagine Dragons, Queen and The Beatles. Each training set includes 80 songs and each evaluation set 20 songs. If the perplexity score for the previous BLEU computation is presented we obtain in this case very similar values (tiny decay from 19.592 to 19.166). As a summary, mean scores for all conducted tests are presented in the following table:


| _Artist_        | _Total number test datasets_ | _Average Sequence Length_ | _Average PPL score (GPT-2 model)_ | _PPL score (fine-tuned GPT-2-model)_ |
|-----------------|------------------------------|---------------------------|-----------------------------------|--------------------------------------|
| 50 Cent         | 20                           | 881.20                    | 24.33                             | 16.99                                |
| Taylor Swift    | 20                           | 518.75                    | 11.55                             | 7.98                                 |
| Imagine Dragons | 20                           | 466.75                    | 8.51                              | 6.63                                 |
| Queen           | 20                           | 317.15                    | 23.15                             | 15.33                                |
| The Beatles     | 20                           | 210.30                    | 15.59                             | 10.18                                |


Conclusions : Our hypothesis is confirmed and we get lower values of perplexity in fine-tuned model output. However, perplexity alone can not be used to measure the performance of the Lyrics generator. Perplexity measures the model's ability to predict the next word based on the context but it doesn't capture the semantic coherence or meaningfulness of the generated lyrics. A model could have low perplexity but still produce nonsensical or incoherent lyrics. In addition, perplexity can not measure creativity.

Link to the performance report for single-artist BLEU score and perplexity: https://wandb.ai//upcproject/Lyrics%20Generator/reports/Single-artist-performance-report--Vmlldzo0ODkxNDA4?accessToken=4noqi3yx5rreg3syp16ikpary7jzravi875c57fg2k6ku6jvkef2nliu4k7tipu2
<p align="right"><a href="#toc">To top</a></p>

### 7.6 Experiment 6: Performance evaluation single-artist different datasets <a name="experiment_6"></a>

**A. BLEU SCORE**

Experiment setup: Same processing as it has been introduced in the previous example. In this case both genius-lyrics and 79-musical-genres dataset are used and compared for the same artist.

Hypothesis: BLEU score should be higher in the bigger dataset i.e. 79-musical-genres.

Results: BLEU score is computed for three different artists using both datasets. As an example, the same lyric generation is showed for both fine-tuned models compared with raw GPT-2.


***genious-lyrics dataset:*** 80 training lyrics and 20 evaluation lyrics.

| _Artist_ | _Prompt (Last 26 words)_                                                                                                                   | _True Lyrics (20 words)_                                                                                                   | _GPT-2 model Generated lyrics  (20 words)_                                                                      | _Fine-tuned GPT-2-model Generated lyrics  (20 words)_                                                            |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| 50 Cent  | (...) Them earrings is nice, that's what you bought for me? Take that shit off, move, I'll break you off properly I get mine the fast way, | ski mask way Make money, make money-money-money Nigga, if you ask me, this the only way Take money, take money-money-money | Yeah I'm tryna catch me something  A little jux or something Nigga, that watch is nice,  that's what you bought | Yeah I'm tryna catch me something  A little jux or something Nigga, that watch is nice,  that's what you bought  |


In this particular example, BLEU scored 0.10 for both the GPT-2 model and the fine tuned GPT-2 model.


***79-musical-genres dataset:*** 375 training lyrics and 94 evaluation lyrics.

| _Artist_ | _Prompt (Last 26 words)_                                                                                                              | _True Lyrics (20 words)_                                                                                 | _GPT-2 model Generated lyrics  (20 words)_                                                        | _Fine-tuned GPT-2-model Generated lyrics  (20 words)_                                                       |
|----------|---------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| 50 Cent  | (...) That’s what you bought from me? Take that shit off, move I’ll break you off properly I get mine the fast way, ski mask way Make | money Make money, money, money nigga if you ask me It’s the only way Take money Take money, money, money | money Keep it real  Make money, money, money  Ya bitch it looks like you did it on my back.  What | money  Make money, money, money  nigga if you ask me  It’s the only way  Take money  Take money Take money, |

In this particular example, BLEU scored 0.30 for the GPT-2 model and 0.95 the fine tuned GPT-2 model. Altough these results show promissing improvements for the computed scores using the same lyrics in both models, the summarized tables for all artists show otherwise. 

| _Artist_        | _Total number test datasets_ | _Average prompt words_ | _BLEU score (GPT-2 model)_ | _BLEU score (fine-tuned GPT-2-model - small dataset)_ |
|-----------------|------------------------------|------------------------|----------------------------|---------------------------------------|
| 50 Cent         | 14*                          | 523.21                 | 0.33                       | 0.40                                  |
| Taylor Swift    | 19*                          | 346.89                 | 0.22                       | 0.32                                  |

_*The amount of test sets is sometimes lower than 20 as some lyrics contained more than 1024 tokens. This is the maximum accepted for a single generation prompt in GPT-2 models. For later executions a timming factor has been added to the tokanizer to avoid bypassing larger prompts._

| _Artist_        | _Total number test datasets_ | _Average prompt words_ | _BLEU score (GPT-2 model)_ | _BLEU score (fine-tuned GPT-2-model - big dataset)_ |
|-----------------|------------------------------|------------------------|----------------------------|---------------------------------------|
| 50 Cent         | 86*                          | 473.05                 | 0.25                       | 0.33                                  |
| Taylor Swift    | 76*                          | 306.87                 | 0.22                       | 0.30                                  |

_*The amount of test sets is sometimes lower than 94 as some lyrics contained more than 1024 tokens. This is the maximum accepted for a single generation prompt in GPT-2 models. For later executions a timming factor has been added to the tokanizer to avoid bypassing larger prompts._


Conclusions: BLEU score shows to be lower in lyrics generated by the fine-tuned model, which indicates that increasing the amount of training data hasn't increased the n-gram simmilarity between generations. This revokes our initial hypothesis although metrics show to improve when comparing against the raw GPT-2 model.

**B. PERPLEXITY**

Experiment setup: Same processing as it has been introduced in the previous example. In this case both genius-lyrics and 79-musical-genres dataset are used and compared for the same artist.

Hypothesis: perplexity score should be lower in the bigger dataset i.e. 79-musical-genres.

Results: Perplexity score is computed for two different artists using both datasets. Summarized results are presented as it follows

***genious-lyrics dataset:*** 80 training lyrics and 20 evaluation lyrics.


| _Artist_        | _Total number test datasets_ | _Average Sequence Length_ | _Average PPL score (GPT-2 model)_ | _PPL score (fine-tuned GPT-2-model - small dataset)_ |
|-----------------|------------------------------|---------------------------|-----------------------------------|--------------------------------------|
| 50 Cent         | 20                           | 881.20                    | 24.33                             | 16.99                                |
| Taylor Swift    | 20                           | 518.75                    | 11.55                             | 7.98                                 |


***79-musical-genres dataset:*** 80 training lyrics and 20 evaluation lyrics.


| _Artist_        | _Total number test datasets_ | _Average Sequence Length_ | _Average PPL score (GPT-2 model)_ | _PPL score (fine-tuned GPT-2-model - big dataset)_ |
|-----------------|------------------------------|---------------------------|-----------------------------------|--------------------------------------|
| 50 Cent         | 94                           |  712.79                   | 41.94                             | 19.55                                |
| Taylor Swift    | 77                           | 416.86                    | 14.18                             | 9.69                                 |

Conclusions: Similarly to the previous metric PPL score shows to be higher in lyrics generated by the fine-tuned model. This revokes our initial hypothesis although metrics show to improve when comparing against the raw GPT-2 model. As it was exposed in the previous experiment, these metrics can't be considered a reliable generation quality check. They can show general trends but subjective assessment in this case is much optimal.

Link to the performance report for single-artist BLEU score and perplexity: https://wandb.ai//upcproject/Lyrics%20Generator/reports/Single-artist-performance-report--Vmlldzo0ODkxNDA4?accessToken=4noqi3yx5rreg3syp16ikpary7jzravi875c57fg2k6ku6jvkef2nliu4k7tipu2

### 7.7 Experiment 6: T5 model <a name="experiment_7"></a>
T5 (Text-To-Text Transfer Transformer) is a transformer-based language model developed by Google Research. Unlike GPT-2, which is primarily designed for autoregressive language generation, T5 follows a "text-to-text" framework. Instead of generating lyrics word by word like GPT-2, T5 is trained to map an input text prompt to an output text sequence.
T5 was trained on a large and diverse collection of publicly available text data from the internet. The specific details regarding the exact number of data, tokens, and parameters used for training T5 have not been disclosed publicly by Google Research. However, it is known to be a large-scale model with billions of parameters.

<a href="https://drive.google.com/uc?export=view&id=1VdnrzFDsd-yFs99O8ujCDiWPRzJFdoow" align="left">
  <img src="https://drive.google.com/uc?export=view&id=1VdnrzFDsd-yFs99O8ujCDiWPRzJFdoow" alt="Image" style="width: auto; max-width: 70%; height: auto; display: inline-block;" title="Image" />
</a>

Unfortunately, due to time constraints and our prioritization of experimenting with GPT-2, we were unable to conduct the T5 experiment.

<p align="right"><a href="#toc">To top</a></p>

## 8. Conclusions <a name="conclusions"></a>

Model Performance: The performance of the lyrics generator is heavily influenced by the size and quality of the dataset used for fine-tuning. When fine-tuning the model on a small dataset, the generated lyrics may exhibit limitations in terms of coherence, structure, and meaningfulness. In contrast, fine-tuning the model on a larger and more diverse dataset tends to yield lyrics that are more coherent and contextually relevant.

Furthermore, we observed that the model's performance is affected when generating lyrics for artists who extensively utilize slang or have unconventional grammar. This could be attributed to the limited availability of data capturing such patterns, making it challenging for the model to learn them effectively. Additionally, issues with the tokenizer may arise, preventing correct tokenization of new or fragmented words.

Interestingly, during single artist training, we noticed that artists like 50 Cent or Lil Wayne exhibited higher validation loss compared to other artists. This suggests that these artists' lyrics may pose additional challenges for the model, potentially due to their unique linguistic characteristics or unconventional usage of language.

In summary, the performance of the lyrics generator depends on factors such as dataset size, diversity, and the linguistic patterns present in the training data.

Training time and hardware requirements: Fine-tuning the GPT-2 model on a large dataset can be a computationally intensive task. It is essential to have access to sufficient computational power to train the model effectively.

Creativity: While the lyrics generator can produce very good outputs, it is important to note that the model will always lack true creativity and originality that a human songwriter could produce. However, it can serve as a valuable tool for generating initial ideas or providing inspiration to human songwriters.
<p align="right"><a href="#toc">To top</a></p>

## 9. Next Steps <a name="next_steps"></a>
After completing the project, several potential further steps for research and exploration can be proposed:

* Trying a model with different architecture (e.g. T5)
* Multilingual Lyrics Generation: Extend the project to support lyrics generation in multiple languages.
* Generation of lyrics with a proper song structure, including a title, chorus, verses and other sections.
* Multimodal lyrics generation: Extend the project to generate not only textual lyrics but also explore the generation of accompanying music or melodies.
* Deploying an application for lyrics generation
<p align="right"><a href="#toc">To top</a></p>

## 9. References <a name="references"></a>
https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272

https://www.youtube.com/watch?v=cHymMt1SQn8&ab_channel=NicholasRenotte

https://snappishproductions.com/blog/2020/03/01/chapter-9.5-text-generation-with-gpt-2-and-only-pytorch.html.html

https://huggingface.co

https://towardsdatascience.com/how-i-created-a-lyrics-generator-b62bde13badb

https://www.aiweirdness.com/the-ais-carol-19-12-24/

https://blog.ml6.eu/gpt-2-artificial-intelligence-song-generator-lets-get-groovy-3e7c1f55030f

https://github.com/adigoryl/Styled-Lyrics-Generator-GPT2

https://medium.com/mlearning-ai/artist-based-lyrics-generator-using-machine-learning-eb70dc4fb993

https://elischolar.library.yale.edu/cgi/viewcontent.cgi?article=1000&context=yurj
    
