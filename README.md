# visum-competition2021
[VISUM 2021](http://visum.inesctec.pt) Summer School Digital Edition - The Competition Repository 

Here you will find all information regarding the competition, including:

1. General information about the challenge, including the evaluated metrics.
2. Detailed technical instructions about the competition.
3. Auxiliary source code, including the source code of the baseline solution.

We wish you all good luck and much success in your work :)

**Presentation** - [here](INSERT_LINK)

**FAQs** - [here](https://github.com/visum-summerschool/visum-competition2021/blob/main/VISUM2021_FAQs.pdf)

**Submission Platform** - [here](https://visum.inesctec.pt/submissions)

## How to run the baseline code (from the /home/visum folder)
1. Access the baseline source code by either:
   - Accessing the files already present in your machine **OR**
   - Cloning this repo into your machine by doing and afterwards moving the contents of the created directory into ```/home/visum```
      ``` 
      git clone https://github.com/visum-summerschool/visum-competition2021.git
      cp -r visum-competition2021/ .
      rm -r visum-competition2021 
      ```
2. run ```python3 split_data_nondisjoint.py``` to create the training and validation splits in ```/home/visum/processed_data```
3. run ```python3 generate_community_prods.py``` to generate the louvain communities
4. Either:
    - train the baseline using ```python3 train.py``` (model weights and checkpoints are saved in ```/home/visum/results/<timestamp>```) **OR**
    - use the baseline weights which are already in ```/home/visum/results/```.
5. **IF** you trained a new model you need to copy the weights (and the tokenizer) into ```/home/visum``` by doing:
    ```
       cp -r results/<timestamp>/tokenizer results/.
       cp results/<timestamp>/best_model_weights.pth results/.
    ```
    where \<timestamp\> corresponds to the name of the folder where the trained model is located. Usually this will be a timestamp in the format YYYY-MM-DD_hh-mm-ss.
6. generate predictions for your model with ```python3 test.py```
7. check the predictions are in the correct format by running ```python3 evaluate.py preds.csv /home/master/dataset/test/solutions.csv```
8. submit your results by accessing [here](https://visum.inesctec.pt/submissions)


### [Optional] How to test your model on the validation split
The steps described in the previous section allow you to train your model and test it on some dummy test queries (generated from the train data) located in ```home/master/dataset/test```. However, you might want to test the model on the validation split created in step 2. To do so, consider the following steps:

1. run ```python3 generate_test_queries.py```to generate queries from your validation split
2. test your model with ```python3 test.py -t processed_data/valid```
3. evaluate your model by running ```python3 evaluate.py preds.csv processed_data/valid/solutions.csv```

 
