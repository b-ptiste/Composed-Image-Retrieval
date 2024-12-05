# Project

We carried out this project as part of the Object recognition and computer vision 2023 course at ENS Ulm during our semester in the MVA master's programme.

You can read our report : [here](https://drive.google.com/file/d/1TSedD1iCNDuQ77YY0pboVanj45ZuREN6/view).

The paper ”Learning Composed Video Retrieval from Web Video Captions” published at AAAI'24 introduces the Composed Video Retrieval (CoVR) task, an advancement of Composed Image Retrieval (CoIR). As illustrated in the Figure bellow, given an image and a modification text, the objective is to perform video retreival in large database. 

![example (1)](https://github.com/user-attachments/assets/c6552754-b051-4ebe-838e-e6a281315171)

Our aim is to provide a comprehensive analysis of the solutions proposed in the paper from a theoretical and practical point of view.
* We reproduced their experiments.
* We proposed to go further by studying explainability using attention mechanisms to understand model predictions.
* We studied the sampling process with three new approaches
* We replaced the original BLIP architecture with the more advanced BLIP-2.

As a result, we have obtained a slight improvement compared with the existing method with the sampling study.

## Reproce the experiments 

Our different experiments could be fined of 3 differents branches on this repo : 
* sampler_exp
* attention_exp
* blip2-exp

Experiments were conducted on GCP virtual machines.

## Installation

Details for dependency and data can be found in the original repo : https://github.com/lucas-ventura/CoVR/

## Some nice results 

### Attention Experiments

We can see that the model uses more the multimodal features rather than image or text features. In addition, we also observe better results when the model uses more the image features than text features. This corroborates results from the original papers.

<img src="https://github.com/b-ptiste/Composed-Image-Retrieval/assets/75781257/8b3b0d36-a586-4d23-955e-2a38655807c8" width="300" alt="table_sampler">

### Sampler Experiments

Our first strategy is Hard Negative Sampling ($\textit{HNS}$). The idea is to have all the images belonging to the same member in the same batch. A member is made up of images that are semantically very similar and whose differences can be described using simple modification texts. On the other hand, we implement a Filtering Sampling ($\textit{FS}$). We would like to see the influence of learning when the images of the same member are not in the same batch. 

We propose $\beta$-Hard/Filtering Sampling ( $\beta$ HN-FS) which allows to control the part $\beta$ of the batches with which the $\textit{HNS}$ strategy is used $\beta .\textit{FS} + (1- \beta) . \textit{HNS}$. We have noticed that the larger $\beta$ is, the better the model is in general (ie for $Recall$@k for high K). On the contrary, when $\beta$ decreases slightly then the $Recall$@k for small K are better to the detriment of the large K. When $\beta$ tends towards 0 then the performances degrade. 

<img src="https://github.com/b-ptiste/Composed-Image-Retrieval/assets/75781257/e83c10c8-1746-4190-b157-132c92dbfbb6" width="300" alt="table_sampler">

# Authors
* CALLARD Baptiste (MVA)
* ZHENG Steven (MVA)

# Acknowledgement and credit
We would like to thank Lucas Ventura for his help with this project. In addition, our github is a clone of its project (see. [web page](https://imagine.enpc.fr/~ventural/covr/)).

