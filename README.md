# More than Votes? Voting and Language based Partisanship in the US Supreme Court

## Introduction

This repository contains code introduced in the following paper:

- [More than Votes? Voting and Language based Partisanship in the US Supreme Court](https://aclanthology.org/2023.findings-emnlp.306/)

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


## Related work 
- Biaoyan Fang, Trevor Cohn, Timothy Baldwin, and Lea Frermann. 2023. Super-SCOTUS: A multi-sourced dataset for the Supreme Court of the US. In Proceedings of the Natural Legal Language Processing Workshop 2023, Singapore. Association for Computational Linguistics. [Github](https://github.com/biaoyanf/Super-SCOTUS)


- Biaoyan Fang, Trevor Cohn, Timothy Baldwin, and Lea Frermann. 2023. It’s not only What You Say, It’s also Who It’s Said to: Counterfactual Analysis of Interactive Behavior in the Courtroom. In Proceedings of The 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, Nusa Dua, Bali. Association for Computational Linguistics. [Github](https://github.com/biaoyanf/SCOTUS-counterfactual)
