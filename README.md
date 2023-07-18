# Transformers - Lyrics-Generator
## Dataset
The dataset used for this project is the [Lyrics Dataset](https://www.kaggle.com/datasets/mervedin/genius-lyrics) from Kaggle.

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
6. [Results](#results)
    1. [Experiment 1: Single-artist training](#experiment_1)
    2. [Experiment 2: Specific genre training](#experiment_2)
    3. [Experiment 3: Conditional lyrics generation](#experiment_3)
    4. [Experiment 4: Performance evaluation](#experiment_4)
    5. [Experiment 5: T5 model](#experiment_5)
7. [Conclusions](#conclusions)
10. [Next Steps](#next_steps)
11. [References](#references)

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
<p align="right"><a href="#toc">To top</a></p>

## 3. Working Environment <a name="working_env"></a>
[PyTorch](https://pytorch.org/) is used as the main framework.
We started out in [Google Colab](https://colab.research.google.com/) as it was fast and easy for us to access. Then we passed on to training locally and via a VM instance on [Google Cloud](https://cloud.google.com/) but we had a problem with access to GPUs on Google Cloud therefore we couldn't complete our training there. To view the results we used [wandb](https://wandb.ai/site). 

<p align="middle">
  <a href="https://drive.google.com/uc?export=view&id=1jgmyVjKrc69KLUzmZw7j2BYIghZrDnZL">
    <img src="https://drive.google.com/uc?export=view&id=1jgmyVjKrc69KLUzmZw7j2BYIghZrDnZL" alt="Image" style="width: auto; max-width: 50%; height: 80px; display: inline-block;" title="Image" />
  </a>
  
  <a href="https://drive.google.com/uc?export=view&id=1N2ui7rYVl6WPUAgzuMgFe7TU2c_MGm56">
    <img src="https://drive.google.com/uc?export=view&id=1N2ui7rYVl6WPUAgzuMgFe7TU2c_MGm56" alt="Image" style="width: auto; max-width: 50%; height: 80px; display: inline-block;" title="Image" />
  </a>
  
  <a href="https://drive.google.com/uc?export=view&id=1LClGQxV6tDbLHU4dEowivvrehZPXnkHB">
    <img src="https://drive.google.com/uc?export=view&id=1LClGQxV6tDbLHU4dEowivvrehZPXnkHB" alt="Image" style="width: auto; max-width: 40%; height: 80px; display: inline-block;" title="Image" />
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

## 6. Results <a name="results"></a>
    
### 6.1 Experiment 1: Single-artist training and generation with same prompt <a name="experiment_1"></a>


**A. Training on Genius Lyrics dataset** https://www.kaggle.com/datasets/mervedin/genius-lyrics

Experiment setup: We fine-tuned GPT-2 on 100 lyrics by a single artist. We used local environment for training. For generatiion we use the same input prompt for all the artist tested. We use a small input prompt to not condition the model too much with it. We want to see how it behaves with a small input prompt for each artist tested. We trained and tested on 6 artists: 50 Cent, Taylor Swift, The Beatles, Queen, Justin Bieber, and Imagine Dragons. 

Hypothesis: Even though we only use 100 songs, the model should be able to fit the vocabulary and style of each one of them, and continue the input prompt with the artist style.

Resuts: 
The full results can be seen in the W&B report. #TODO: add link to the "You are" report. Here we are going to show some examples of the results obtained.
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

    On this examples of the results obtained, we can see how on 50 Cent, the model understood that his lyrics have more agressive tone, talks about drugs, sex, crime, etc. When going to Taylor Swift, the style changes completely. Now talks way more about feelings and love, and it's really delicate with the words used. When generating The Beatles, we can see some claim pro peace.

    Conclusions:
    We can conclude, that the Hypothesis was correct, not only changes the tone and vocabulary used to one that the artist would use, also change the topic of the lyrics. We can see that the model is able to fit the vocabulary and style of each artist, and continue the input prompt with the artist style. We can see the same behaviour with the other artists tested and with other inputs. To see more generations with another input go to the W&B report. #TODO: add link to the "I will" report.

### 6.2 Experiment 2: Specific genre training and generation with same prompt <a name="experiment_2"></a>

Experiment setup: We fine-tuned GPT-2 over the second dataset filtered by genre. Depending on the genre we found more or less songs. It's a similar situation to the experiment 1, but we want to try to learn the patterns of something more general than an artist. The amount of songs used for each genre are:

* Pop: 375
* Reggae: 223
* Rock: 954
* Hip Hop: 108

Hypothesis: The model is going to be able to fit the genre and produce new songs following the style of the genre.

Results:
The full results can be seen in the W&B report. #TODO: add link to the genre "You are" report. Here we are going to show some examples of the results obtained.
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

    As we can see, the results are similar to the previous experiments. The topics and the way they are expressed match the genre. For example, reagge it's really related to Rastafari, a religion developed in Jamaica during the 1930s. That's why in many reggae songs, the main topic it's religion and the relationship between human and god. Also, another common topic is freedom, related to the slavery past of black people in Jamaica and to a society that doesn't fit with the rastafari way of seeing live. As we can see in the snipped of reggae generated text above, these two topics are convined. On the other hand, the model doesn't really use rastafari slang. Maybe there are just not enough examples on the training for the model to learn how to and when use them. Also we observe how the lyrics generated for "Pop" are mainly talking about love or heartbreak, and on "Rock" tend to use more complex vocabulary and sentences. On "Hip Hop" we find the same problem than on reggae, it doesn't really use specific vocabulary and the "Hip Hop" style is not really visible, but make sense seeing the amount of songs used for the training.

    Conclusion:

    As we observed in the prvious experiment, the model was also able to fit the genre and learn the style and tone of it, even though it looks like it's harder for it to use specific vocabulary. We can see the same behaviour with other inputs. To see more generations with another input go to the W&B report. #TODO: add link to the "I will" report.



<p align="right"><a href="#toc">To top</a></p>

### 6.3 Experiment 3: Conditional lyrics generation with the same prompt <a name="experiment_3"></a>
Experiment setup: For this experiment, we trained GPT-2 with the songs of ten artists from the first dataset. That way we ensure that we have 100 songs for each one. The objective is to build a conditional model that let you choose on whose style you want to generate songs. To do that, after the preprocessing of the lyrics, we added the artist name ar the beginning of them. Then, at generation time, we concatenate the name of the artist we want to generate before the initial prompt. That way we make the model understand the relationship between the lyric and the artist. 
The ten artist used are the next ones:
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

Hypothesis: The model should produce similar results to the ones obtained on the first experiment.

Results:
The full results can be seen in the W&B report. 
W&B link: https://wandb.ai//upcproject/Lyrics%20Generator/reports/Multiple-artists-generated-lyrics-prompt-You-are---Vmlldzo0OTEyNzQ3?accessToken=g3lev2vf26vg7jsg20q6a2qgmmq8faqczopt89e99f2gp1xprrjaznx87e3llqxk. 

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

  As we can see, the model performs similar to the fist experiment. We can easyly see the change of style and topics depending on the artist requested even though the input prompt it's the same. The model went from the 'egotrip' and tough vocabulary from The Notorious B.I.G., to a heartbroken Justin Beaver.However, we note that there is still quite a bit of room for improvement in terms of coherence of the lyrics. We expected the results to be better in that sense. The main issue is the lack of computational resources to increase the dataset and/or further fine-tune the parameters.

  Conclusions:
  
  The model is big enough to fit multiple artist at the same time and is able to understand the conditioned prompt. To see more generations with another input go to the W&B report. 
  W&B link: https://wandb.ai//upcproject/Lyrics%20Generator/reports/Multiple-artists-generated-lyrics-prompt-I-will---Vmlldzo0OTEyNjgw?accessToken=lnb3rb377mkatl0g5xk76ral4a4w7wk03wclk4dsvcvmxnmdg718hlqknmohwrs1
   
  Link to W&B training report: https://api.wandb.ai/links/upcproject/zzppte9f 
  The training is quite stable with gradually decresing training loss and evaluation loss.

<p align="right"><a href="#toc">To top</a></p>

### 6.4 Experiment 4: Conditional lyrics generation based on the prompt <a name="experiment_4"></a>

Experiment setup:
For this experiment, we trained GPT-2 with lyrics from the first dataset with lyrics of 50 Cent and Justin Bieber. Both of them have 100 songs. The difference with the previous experiment is, that instead of conditioning the generation specifying the artist. We use an initial prompt that will lead the model to follow one style or the other.

Hypothesis: The model will be able to understand, based on the topic or style of the initial prompt, which style should follow. Sometimes might get confused and mix both styles even though they are really different.

Results:
The full results can be seen in the W&B report. 
# TODO add report of runs 
generate_multiple_2_artists_genius-lyrics_I'm_hustling-20230718-194558
generate_multiple_2_artists_genius-lyrics_I_miss_you-20230718-194748
generate_multiple_2_artists_genius-lyrics_I'm_sad-20230718-201241
generate_multiple_2_artists_genius-lyrics_The_street_is_tough-20230718-201202

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

  As we can see, depending on the input prompt the model will use the style that it's more apropiate for it. But as we can see on the second example, the lyrics start with a Justin Bieber style, and at some point switch to a 50 Cent style. That should not happen as its not coherent with itself.

  Conclusion:
  The model is able to fit both styles, even though they are rally different, and in general, know when to use the more appropiate one.

### 6.4 Experiment 4: Performance evaluation single-artist GPT-2 vs fine tuned GPT-2 <a name="experiment_4"></a>

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

_*The amount of test sets is sometimes lower than 20 as some lyrics contained more than 1024 tokens. This is the maximum accepted for a single generation prompt in GPT-2 models._


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

### 6.5 Experiment 5: Performance evaluation single-artist different datasets <a name="experiment_5"></a>

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

_*The amount of test sets is sometimes lower than 20 as some lyrics contained more than 1024 tokens. This is the maximum accepted for a single generation prompt in GPT-2 models._

| _Artist_        | _Total number test datasets_ | _Average prompt words_ | _BLEU score (GPT-2 model)_ | _BLEU score (fine-tuned GPT-2-model - big dataset)_ |
|-----------------|------------------------------|------------------------|----------------------------|---------------------------------------|
| 50 Cent         | 86*                          | 473.05                 | 0.25                       | 0.33                                  |
| Taylor Swift    | 76*                          | 306.87                 | 0.22                       | 0.30                                  |

_*The amount of test sets is sometimes lower than 94 as some lyrics contained more than 1024 tokens. This is the maximum accepted for a single generation prompt in GPT-2 models._

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

### 6.6 Experiment 6: T5 model <a name="experiment_6"></a>
T5 (Text-To-Text Transfer Transformer) is a transformer-based language model developed by Google Research. Unlike GPT-2, which is primarily designed for autoregressive language generation, T5 follows a "text-to-text" framework. Instead of generating lyrics word by word like GPT-2, T5 is trained to map an input text prompt to an output text sequence.
T5 was trained on a large and diverse collection of publicly available text data from the internet. The specific details regarding the exact number of data, tokens, and parameters used for training T5 have not been disclosed publicly by Google Research. However, it is known to be a large-scale model with billions of parameters.

<a href="https://drive.google.com/uc?export=view&id=1VdnrzFDsd-yFs99O8ujCDiWPRzJFdoow" align="left">
  <img src="https://drive.google.com/uc?export=view&id=1VdnrzFDsd-yFs99O8ujCDiWPRzJFdoow" alt="Image" style="width: auto; max-width: 70%; height: auto; display: inline-block;" title="Image" />
</a>

Unfortunately due to the lack of time we could not conduct the T5 experiment.

<p align="right"><a href="#toc">To top</a></p>

## 7. Conclusions <a name="conclusions"></a>
Model Performance: The performance of the lyrics generator heavily depends on the size and quality of the dataset used for fine-tuning. When fine-tuning the model on a small dataset, the generated lyrics may lack coherence, structure, and meaningfulness. On the other hand, fine-tuning the model on a larger and more diverse dataset tends to produce more coherent and contextually relevant lyrics.

Training time and hardware requirements: Fine-tuning the GPT-2 model on a large dataset can be a computationally intensive task. It is essential to have access to sufficient computational power to train the model effectively.

Creativity: While the lyrics generator can produce very good outputs, it is important to note that the model will always lack true creativity and originality that a human songwriter could produce. However, it can serve as a valuable tool for generating initial ideas or providing inspiration to human songwriters.
<p align="right"><a href="#toc">To top</a></p>

## 8. Next Steps <a name="next_steps"></a>
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
    
