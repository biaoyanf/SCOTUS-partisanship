# More than Votes? Voting and Language based Partisanship in the US Supreme Court

## Introduction

This repository contains code introduced in the following paper:

- More than Votes? Voting and Language based Partisanship in the US Supreme Court 

- Biaoyan Fang, Trevor Cohn, Timothy Baldwin, and Lea Frermann 

- In EMNLP2023 findings 

## Dataset 

- This dataset is a subset of [the Super-SCOTUS dataset](https://github.com/biaoyanf/Super-SCOTUS), a multi-sourced dataset for the Supreme Court of the US. We make this subset available under [Harvard Database](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SHFQYU). 
- Download the related file from abpve link, which we used for the experiment in this paper. 
- Change the dataset path in `experiments.conf` to the downloaded path 



## Training Instructions
- Run with python 3
- Run `python train.py <experiment>`, where experiment can be found from `experiments.conf` 
- Run `python train_chronological.py <experiment>` for chronological setting 


## Evaluation
- Evaluation: `python evaluate.py <experiment>`, where the experiment listed above 
- Run `python evaluate_chronological.py <experiment>` for chronological setting 


