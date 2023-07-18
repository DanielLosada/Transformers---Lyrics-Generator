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
4. [General Architecture](#architecture)
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

## 4. General Architecture <a name="architecture"></a>
The development of advanced language models has brought significant changes to tasks like lyrics generation in natural language processing. These models, based on transformer architectures, have shown impressive skills in understanding and creating meaningful text that makes sense in different contexts. GPT, one of these models, has received a lot of attention because of its outstanding performance and flexibility. We have chosen to utilize GPT-2, which is the most recent version of the GPT models accessible on the Hugging Face platform.

GPT-2 consists of solely stacked decoder blocks from the transformer architecture. This architecture allows GPT-2 to effectively capture the relationships between words and generate coherent and contextually relevant text.

The GPT-2 model was trained on a large corpus of text data that consisted of approximately 40 gigabytes of text (around 8 million tokens). The model has 1.5 billion parameters.

<p align="left">
  <a href="https://drive.google.com/uc?export=view&id=1phbpF208uCZM_w4tmRwuD0BA9IdGb68G">
    <img src="https://drive.google.com/uc?export=view&id=1phbpF208uCZM_w4tmRwuD0BA9IdGb68G" alt="Image" style="width: 500px; height: 500px; display: inline-block;" title="Image" />
  </a>
</p>
<p align="right"><a href="#toc">To top</a></p>

## 5. Preprocessing the data set <a name="dataset_preprocess"></a>
Overall, the preprocessing steps involve:

* extracting the dataset
* removing non-English authors to ensure language consistency
* cleaning and formatting the lyrics data to eliminate unwanted artifacts
* tokenizing the datasets for further processing, setting a maximum context length (maximum context length for GPT-2 model is 1024 tokens but we use 512 due to limitations in computational resources)
<p align="right"><a href="#toc">To top</a></p>

## 6. Results <a name="results"></a>
    
### 6.1 Experiment 1: Single-artist training <a name="experiment_1"></a> 

**A. Training on Genius Lyrics dataset** https://www.kaggle.com/datasets/mervedin/genius-lyrics

Experiment setup: We trained on 100 lyrics by a single artist. We used Google Colab or local environment for training.

Hypothesis: Training data size is very limited, we expect overfitting.

Resuts and conclusions: We are not even sure if the model is training properly, in 4 out of 5 runs the training loss is not even shown (We used the Trainer class from Hugging face and the Linear learning rate scheduler with the initial lr=5e-6). With bigger learning rates we observe overfitting. 

The problems that we encountered in the generated lyrics were also mostly due to the small size of the dataset - predisposition to word repetition and to generating truncated lines or lines consisting of one word. We tried to address this issue in post processing by introducing a __post_process function that cleans up the generated sequences of lyrics by removing redundant line breaks, and removes consecutive duplicated words using the __remove_consecutive_duplicates helper function.

Link to W&B training report: https://api.wandb.ai/links/upcproject/uxoj59gw

**B. Training on Lyrics from 79 musical genres dataset** https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres

Experiment setup: The number of lyrics for one artist in this dataset is higher - 300-500 songs (depending on the artist we choose)

Hypothesis: We expect better performance.

Resuts and conclusions: Now we see progress in training and a certain decrease in training and evaluation loss. The highest evaluation loss we get is for the model trained on 50 Cent, which we believe might be due to unique vocabulary and language style.

Link to W&B training report: https://api.wandb.ai/links/upcproject/5ao7yfw0
### 6.2 Experiment 2: Specific genre training <a name="experiment_2"></a>
Experiment setup: Now we are training on even bigger amounts of data - a set of lyrics of a certain genre (determined by an argument specified in argparse) containing of up to a 1000 of songs (Lyrics from 79 musical genres dataset) . Training is done in a local environment or via a Google Cloud VM instance (CPU only, we didn't have GPUs available). 
We only choose artists with popularity >5 since we believe that with more popular artists the chances of getting better quality lyrics are higher since their lyrics have been checked and validated by many users. Some artists´songs also belong to several genres, we only take into account those that have songs of only one genre to avoid genre mixup in our generated lyrics.

Hypothesis: We expect the training to be more productive and a significant improvement in the quality of generated lyrics.

Results and conclusions: We observe a decrease in overfitting issues, indicating a better generalization capability of the model. The generated lyrics showed reasonable quality and coherence, making more sense in the context of the chosen genre.
At this stage is became more difficult to complete training with the computational resources we had. Training was taking a longer time.

Link to W&B training report: https://api.wandb.ai/links/upcproject/icp5ie11
<p align="right"><a href="#toc">To top</a></p>

### 6.3 Experiment 3: Conditional lyrics generation <a name="experiment_3"></a>
Experiment setup: Training with a full dataset to generate song lyrics similar to those of a specific artist. The dataset (one of the two available) and the artist are determined by arguments specified in argparse. Training locally or via a Google Cloud VM instance (CPU only)

Hypothesis: The model should produce good results but it will be really time costly to train without GPUs, and we can not really afford to tune the hyperparameters.

Results and conclusions : Lyrics of enhanced quality and coherence (though there is still quite a bit of room for improvement).
The main issue was lack of computational resources.

Link to W&B training report: https://api.wandb.ai/links/upcproject/zzppte9f 
<p align="right"><a href="#toc">To top</a></p>

### 6.4 Experiment 4: Performance evaluation <a name="experiment_4"></a>

**A. BLEU SCORE**

Experiment setup: In our evaluation process, we conduct a distinct training session. During evaluation, we provide the model with song lyrics as a prompt and remove the final 20 words from each song. The model generates the missing portion, and we utilize the BLEU score to assess the similarity between the generated lyrics and the truncated reference lyrics. By afterwards comparing the BLEU scores obtained from the model fine-tuned on our dataset and the model without fine-tuning, we can evaluate the quality of the generated lyrics.

Hypothesis: BLEU score should be higher in fine-tuned model output.

Results and conclusions : BLEU score in indeed higher in lyrics generated by the fine-tuned model, it suggests that the fine-tuned model has improved in generating lyrics that are more similar to the reference lyrics.
However, it's important to note that BLEU scores alone do not provide a complete assessment of the quality of the generated lyrics. BLEU primarily measures the overlap in n-grams and may not capture aspects such as creativity, coherence, or semantic understanding. Therefore, it's recommended to complement BLEU scores with other evaluation methods, including human evaluation, to obtain a comprehensive understanding of the fine-tuned model's performance in generating high-quality lyrics.

**A. PERPLEXITY**

Experiment setup: We also measure perplexity (Perplexity = 2^(- (1 / N) * Σ(log2(p(w_i))))). A lower perplexity value indicates that the model is more confident and has a better understanding of the data. Conversely, a higher perplexity value suggests that the model is less certain and has a harder time predicting the data. We compare the perplexity obtained from raw GPT-2 model and the GPT-2-based model fine-tuned on our dataset.

Hypothesis: Perplexity should be lower in fine-tuned model output.

Results and conclusions : Our hypothesis is confirmed and we get lower values of perplexity in fine-tuned model output. However, perplexity alone can not be used to measure the performance of the Lyrics generator. Perplexity measures the model's ability to predict the next word based on the context but it doesn't capture the semantic coherence or meaningfulness of the generated lyrics. A model could have low perplexity but still produce nonsensical or incoherent lyrics. In addition, perplexity can not measure creativity.

Link to the performance report for BLEU score and perplexity: https://wandb.ai//upcproject/Lyrics%20Generator/reports/Single-artist-performance-report--Vmlldzo0ODkxNDA4?accessToken=4noqi3yx5rreg3syp16ikpary7jzravi875c57fg2k6ku6jvkef2nliu4k7tipu2
<p align="right"><a href="#toc">To top</a></p>

### 6.5 Experiment 5: T5 model <a name="experiment_5"></a>
T5 (Text-To-Text Transfer Transformer) is a transformer-based language model developed by Google Research. Unlike GPT-2, which is primarily designed for autoregressive language generation, T5 follows a "text-to-text" framework. Instead of generating lyrics word by word like GPT-2, T5 is trained to map an input text prompt to an output text sequence.
T5 was trained on a large and diverse collection of publicly available text data from the internet. The specific details regarding the exact number of data, tokens, and parameters used for training T5 have not been disclosed publicly by Google Research. However, it is known to be a large-scale model with billions of parameters.

<a href="https://drive.google.com/uc?export=view&id=1VdnrzFDsd-yFs99O8ujCDiWPRzJFdoow" align="left">
  <img src="https://drive.google.com/uc?export=view&id=1VdnrzFDsd-yFs99O8ujCDiWPRzJFdoow" alt="Image" style="width: auto; max-width: 70%; height: auto; display: inline-block;" title="Image" />
</a>


Experiment setup: In our experiment due to the lack of time and computing power we chose to train the T5 model only on single-artist data.

Hypothesis: The results will probably be close to the ones that we got on a single-artist GPT-2-based model , we want to check if we can note any difference.

Results and conclusions : The results observed were mainly similar to those we obtained with GPT-2 - overfitting and rather low quality of generated lyrics. At this stage there can be no noticeable differences tracked compared to the GPT-2 model. We would need to train on a bigger dataset to really perceive the difference.

TODO: links to report/charts/screenshots of obtained resuts???? (weights and biases or other) 
<p align="right"><a href="#toc">To top</a></p>

## 7. Conclusions <a name="conclusions"></a>
Model Performance: The performance of the lyrics generator heavily depends on the size and quality of the dataset used for fine-tuning. When fine-tuning the model on a small dataset, the generated lyrics may lack coherence, structure, and meaningfulness. On the other hand, fine-tuning the model on a larger and more diverse dataset tends to produce more coherent and contextually relevant lyrics.

Training time and hardware requirements: Fine-tuning the GPT-2 or T5 model on a large dataset can be a computationally intensive task. It is essential to have access to sufficient computational power to train the model effectively.

Creativity: While the lyrics generator can produce very good outputs, it is important to note that the model will always lack true creativity and originality that a human songwriter could produce. However, it can serve as a valuable tool for generating initial ideas or providing inspiration to human songwriters.
<p align="right"><a href="#toc">To top</a></p>

## 8. Next Steps <a name="next_steps"></a>
After completing the project, several potential further steps for research and exploration can be proposed:

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
    
