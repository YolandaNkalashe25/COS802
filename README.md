# COMPARATIVE ANALYSIS ON BERT, LTSM AND NAIVE BAYES IN PREDICTING FAKE NEWS:

Author: Yolanda Nkalashe 
<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project

Fake news detection algorithms have gained immense pop- ularity due to the rate at which fake news is being spread through the various online communities. Deep learning mod- els have historically performed better in classifications task especially with respect to natural language tasks (NLP), however since the introduction of the BERT transformers as state of the models they have emerged to be the better counterpart when solving complex issues such as text classi- fication. In this paper we compare which model will produce the best fake news detection system. The models compared are Naıve Bayes, Long short term memory recurrent neural network and the BERT. We further use a less complex inter- pretable machine learning model, decision tree as our base model for comparison. Lastly, we determine how well the models perform against unseen data which in was trained on and a set of unseen South African fake news data set.

Keywords—LTSM, BERT, Fake News, Machine Learning, Transformers.

As stated ee run four experiments using google colab note four experiments 
* Decision Tree
* Nave Bayes
* Long short term memory RNN and
* BERT transformer


<!-- GETTING STARTED -->
## Getting Started

To run the code and and notes books. Google colab is required, alternatively to have the ability to run on local machine using one needs to first open the file using google colab then opt for the option for to download the file as .py file.

There are two google notebooks:
1. Which builds and trains the Machine Learning models and LTSM RNN.
2. Which builds and test the BERT transformer model.

### Prerequisites

1. Change run type to GPU. We run BERT model on GPU.

2. These are the packages required to install:
* numpy
```sh
!pip install numpy
```

* panda
```sh
!pip install panda
```

* nltk
```sh
!pip install nltk
```

* tensorflow
```sh
!pip install tensorflow
```

* hugging faces transformer library
```sh
!pip install -qq transformers
```

### Libraries and connecting to data set

1. Connect to nd mount google drive  drive.mount('/content/drive')
2. import numpy and panda library
```sh
import numpy as np
import pandas as pd
```
3. import Bert and torch libraries
```sh
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
```
4. import natural language toolkit and genism library for word2vec
```sh
import nltk
import gensim
```

<!-- USAGE EXAMPLES -->
## Usage

To run the notebook,one could run all cells under runtime or run each cell seperatly. When training the BERT transformer it will take around 3hours to train the model and 30min to train the LTSM RNN.

* The machine learning models are ran first and trained on kaggle data set.
* Secondly, the LTSM is trained using word2vec word embeddings.
* Lastly the BERT model is trained.
* For each model we test its performance on kaggle test set and on the South african news data set.
* To extract the images produced snipping tool function can be used.

<!-- CONTACT -->
## Contact

Your Name - Yolanda Nkalashe - u13193016@tuks.co.za

Project Link: git@github.com:YolandaNkalashe25/COS802-Project.git

