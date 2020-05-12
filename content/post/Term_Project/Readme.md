### CSE-5334 Project
### Data Mininig 

### BoardGameGeek Reviews: This report is written in Jupiter notebook,
##### By: Amnah Abdelrahman
##### Date: May 11, 2020



The goal of this project is to predict the rating of a board-game given a review, using text analysis tools and machine learning methods. The data of size 1 GB is taken form Kaggle: https://www.kaggle.com/jvanelteren/boardgamegeek-reviews#2019-05-02.csv. The data spread over 3 CSV files:
* games_detailed_info.csv (with 56 columns and 17063) : This file has a many detials about all type of board games, exmaples: are the game rank, number of plyers, number of user rated, time requried for the game, desiner and some instructions.
***
* bgg-13m-reviews.csv (with 6 columns and 13170073): provids the user names with there reviews and rating for each game.
***
* 2019-05-02.csv (9 columns and 17065) : This file contain more precisis information for each game the game name ,year, rank and number of users rated.
***


## Getting Started

These instruction will guid to deploy the project in you local machine 

### Prerequisites

From your terminla you to install the follwing:

   * pip 3 installer [https://pip.pypa.io/en/stable/installing/]
   * Python 3 [https://www.python.org/downloads/]
   * SKlearn [https://scikit-learn.org/stable/install.html]
   
You can use this link for an easy steps [https://evansdianga.com/install-pip-osx/]


### Installing
Frist you will need to downlaod the data from https://www.kaggle.com/jvanelteren/boardgamegeek-reviews#2019-05-02.csv in you local machine, write the right path for your files in you machine. 
Now from you terminal run jupyter notbook, 
it should open a page in your browser. 

## Running the tests

Downlaod the file in your computer then open it from the jupyter notebook tab.
Then from Kernal you can run the codes.


## Deployment


This jupyter Notebook has 5 main sections:
1. Load the file contents, Data description and Data Processig,
    this section has alot of infoamtion about most of the data features, training and testing set slpiting 
2.  Text Analysis and Feature extarction Vector (CountVectorizer, TfidfVectorizer) 
3.  Machine Learning Model Selection 
      
      *  Linear and Logistic Regression <u>[linear_model.LogisticRegression]</u>
      3. Linear Classifiers  <u>[svm.LinearSVC]</u>
      4. Support Vector Machine <u>[SGDClassifier]</u>
      5. Linear Support Vector Machine <u>[svm.LinearSVC]</u>
      6. Naive Bayes Classifier <u>[MultinomialNB]</u>
      7. Decision Tree Classifier <u>[DecisionTreeClassifier]</u>
      8. Ensemble Model 
            - Boosting Models <u>[ensemble.AdaBoostClassifier]</u>
            - Random Forest Classifier <u>[ensemble.RandomForestClassifier]</u>          
            
4.  Model Evaluation and Optimazation (<u>using GridSearch, RandomizedSearchCV</u>)

The end of the file is result descution, cahhlenges and refrences 


## Refrences 

* https://www.kaggle.com/ngrq94/boardgamegeek-reviews-data-preparation
 
* https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
 
* https://medium.com/@galen.ballew/board-games-meet-machine-learning-34026870f8d5
 
* https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

* https://towardsdatascience.com/columntransformer-meets-natural-language-processing-da1f116dd69f   

* https://www.mooc-list.com/course/build-board-game-predictor-using-machine-learning-eduonix
 
* https://monkeylearn.com/text-classification/

* https://www.kaggle.com/emanueleamcappella/random-forest-hyperparameters-tuning

* https://guneetkohli.github.io/machine-learning/board-game-reviews/#.XqK_ky-z1QI

* https://docs.google.com/viewer?a=v&pid=sites&srcid=YWltcy5hYy50enxhaW1zLXRhbnphbmlhLWFyY2hpdmV8Z3g6YmI1ODkxYjEwOGQ4NGVi



```python

```
