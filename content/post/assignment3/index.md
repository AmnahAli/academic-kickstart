---
title: "DATA MINING : Assigment 03"
subtitle: "Implementing Naive Bayes Classifier "
summary: " Sentimental Analysis of moives reviews"

authors:
- admin
tags: []
categories: []
date: "2020-04-02"
lastMod: ""
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ""
  focal_point: ""
  
# Custom links (optional).
#  Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_code: "https://github.com/AmnahAli/academic-kickstart/blob/master/content/post/assignment3/index.ipynb"
url_pdf: ""
url_slides: ""
url_video: ""


# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).


#  Assignment #3  :
 The goal of this assignment is to learn about the Naive Bayes Classifier (NBC).  Implementation of NBC Algorithm from scratch using Python. The targeted problem is text classification. 

### Using text dataset about the movie review. The goal is predicting the sentiment.  
http://ai.stanford.edu/~amaas/data/sentiment/


## (a) Divide the dataset as train, development and test sets. 
we are going to split them after making the matrix

###  Load Librarys 


```python
from sklearn.datasets import load_files
import numpy as np
import pandas as pd
import re 
import string
import random
import nltk
```

### Load traning data  


```python
! wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
```



```python
!tar --gunzip --extract --verbose --file=aclImdb_v1.tar.gz
```


```python
review_train = load_files('aclImdb/train', categories= ['neg', 'pos'])
X,y = review_train.data , review_train.target     # compound reviews 

review_train_postive = load_files('aclImdb/train', categories= ['pos'])
review_train_negative = load_files('aclImdb/train', categories= ['neg'])
X1,y1 = review_train_postive.data , review_train_postive.target      # only postive reviews
X2,y2 = review_train_negative.data , review_train_negative.target    # only negative reviews

print(type(review_train))
print(review_train.keys())
```

    <class 'sklearn.utils.Bunch'>
    dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])



```python
print(type(X), type(y))
train_set=pd.DataFrame(X,columns = ["Reviews"])
train_set["Sentiment_Label"] = y
train_set.head()
```

    <class 'list'> <class 'numpy.ndarray'>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reviews</th>
      <th>Sentiment_Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>b"Zero Day leads you to think, even re-think w...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>b'Words can\'t describe how bad this movie is....</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>b'Everyone plays their part pretty well in thi...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>b'There are a lot of highly talented filmmaker...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>b'I\'ve just had the evidence that confirmed m...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_set.count()
```




    Reviews            25000
    Sentiment_Label    25000
    dtype: int64



### Load test data 


```python
review_test = load_files('aclImdb/test/', categories= ['neg', 'pos'])
X_test,y_test = review_train.data , review_train.target                 # compound reviews   

review_test_postive = load_files('aclImdb/test/', categories= ['pos'])
review_test_negative = load_files('aclImdb/test/', categories= ['neg'])
X1_test,y1_test = review_test_postive.data , review_test_postive.target      # only postive reviews
X2_test,y2_test = review_test_negative.data , review_test_negative.target     # only negative reviews


```


```python
test_set=pd.DataFrame(X_test,columns = ["Reviews"])
test_set["Sentiment_Label"] = y_test
test_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reviews</th>
      <th>Sentiment_Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>b"Zero Day leads you to think, even re-think w...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>b'Words can\'t describe how bad this movie is....</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>b'Everyone plays their part pretty well in thi...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>b'There are a lot of highly talented filmmaker...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>b'I\'ve just had the evidence that confirmed m...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



##  b. Build a vocabulary as list
Bulid the features for training 


```python
def remove_digit_text(data):                        # Remove digit from data and transfrom data from bytelike to string
    data = [txt.lower() for txt in data]
    data = [txt.replace(b'1',b'') for txt in data] 
    data = [txt.replace(b'2',b'') for txt in data]
    data = [txt.replace(b'3',b'') for txt in data]
    data = [txt.replace(b'4',b'') for txt in data]
    data = [txt.replace(b'5',b'') for txt in data]
    data = [txt.replace(b'6',b'') for txt in data]
    data = [txt.replace(b'7',b'') for txt in data]
    data = [txt.replace(b'8',b'') for txt in data]
    data = [txt.replace(b'9',b'') for txt in data]
    data = [txt.replace(b'0',b'') for txt in data] 
    data = [txt.decode() for txt in data]           # create a string using the decode() method of bytes
    
    return data


def tokenize_remove_punctuations(X):                # Tokenize the text into tokens and count their frequencies
    wordfreq = {}
    for txt in X:
        tokens = nltk.RegexpTokenizer(r"\w+").tokenize(txt)     # we could use .lower().split()
        for token in tokens:                                    # and remove the punctuations
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
                
    return wordfreq  


def sort_words_freq(wordfreq_X):                    # Sort words frequency in reverse order    
    wordfreq_sorted = dict(sorted(wordfreq_X.items(), key=lambda x: x[1], reverse=True))
    return wordfreq_sorted
 
    
    
def delete_threshold(wordfreq_sorted):               # ignore rare words if the occurrence is less than five times
    delete = []                                      # delete keys with value less than 5
    for key, val in wordfreq_sorted.items(): 
        if val < 5 : 
            delete.append(key) 

    for i in delete: 
        del wordfreq_sorted[i] 
    return   wordfreq_sorted
    
```


```python
X = remove_digit_text(X)
X_test = remove_digit_text(X_test)

X1 = remove_digit_text(X1)
X2 = remove_digit_text(X2)

X1_test = remove_digit_text(X1_test)
X2_test = remove_digit_text(X2_test)

#print(X1[0])   # check output 
```


```python
wordfreq_X = tokenize_remove_punctuations(X)
wordfreq_X_test = tokenize_remove_punctuations(X_test)

wordfreq_X1 = tokenize_remove_punctuations(X1)
wordfreq_X2 = tokenize_remove_punctuations(X2)

wordfreq_X1_test = tokenize_remove_punctuations(X1_test)
wordfreq_X2_test = tokenize_remove_punctuations(X2_test)

#print(wordfreq_X)   # check output 
```

### A reverse index as the key value might be handy


```python
wordfreq_sorted_X = sort_words_freq(wordfreq_X)
wordfreq_sorted_X_test = sort_words_freq(wordfreq_X_test)

wordfreq_sorted_X1 = sort_words_freq(wordfreq_X1)
wordfreq_sorted_X2 = sort_words_freq(wordfreq_X2)

wordfreq_sorted_X1_test = sort_words_freq(wordfreq_X1_test)
wordfreq_sorted_X2_test = sort_words_freq(wordfreq_X2_test)

#print(wordfreq_sorted_X1)    #check ouput
```

###  You may omit rare words for example if the occurrence is less than five times


```python
wordfreq_sorted_X = delete_threshold(wordfreq_sorted_X)
wordfreq_sorted_X_test = delete_threshold(wordfreq_sorted_X_test)

wordfreq_sorted_X1 = delete_threshold(wordfreq_sorted_X1)
wordfreq_sorted_X2 = delete_threshold(wordfreq_sorted_X2)

wordfreq_sorted_X1_test = delete_threshold(wordfreq_sorted_X1_test)
wordfreq_sorted_X2_test = delete_threshold(wordfreq_sorted_X2_test)

print(wordfreq_sorted_X)
```

    {'the': 336749, 'and': 164141, 'a': 163136, 'of': 145866, 'to': 135724, 'is': 107333, 'br': 101871, 'it': 96467, 'in': 93978, 'i': 87691, 'this': 76007, 'that': 73286, 's': 65710, 'was': 48209, 'as': 46936, 'for': 44345, 'with': 44130, 'movie': 44047, 'but': 42623, 'film': 40161, 't': 34390, 'you': 34267, 'on': 34202, 'not': 30632, 'he': 30155, 'are': 29438, 'his': 29376, 'have': 27731, 'be': 26957, 'one': 26795, 'all': 23985, 'at': 23516, 'they': 22915, 'by': 22548, 'an': 21564, 'who': 21442, 'so': 20615, 'from': 20499, 'like': 20281, 'there': 18865, 'her': 18424, 'or': 18008, 'just': 17774, 

```python
len(wordfreq_sorted_X)  # number of features (words in the dictionary for our model)
```




    28764




```python
print(wordfreq_sorted_X)  #for all the reviews
```

    {'the': 336749, 'and': 164141, 'a': 163136, 'of': 145866, 'to': 135724, 'is': 107333, 'br': 101871, 'it': 96467, 'in': 93978, 'i': 87691, 'this': 76007, 'that': 73286, 's': 65710, 'was': 48209, 'as': 46936, 'for': 44345, 'with': 44130, 'movie': 44047, 'but': 42623, 'film': 40161, 't': 34390, 'you': 34267, 'on': 34202, 'not': 30632, 'he': 30155, 'are': 29438, 'his': 29376, 'have': 27731, 'be': 26957, 'one': 26795, 'all': 23985, 'at': 23516, 'they': 22915, 'by': 22548, 'an': 21564, 'who': 21442, 'so': 20615, 'from': 20499, 'like': 20281, 'there': 18865, 'her': 18424, 'or': 18008, 'just': 17774,  'sex':  
```python
vocab_list= [k  for  k in  wordfreq_sorted_X.keys()]
print(vocab_list[:15])    
```

    ['the', 'and', 'a', 'of', 'to', 'is', 'br', 'it', 'in', 'i', 'this', 'that', 's', 'was', 'as']



```python
vocab_freq= [k  for  k in  wordfreq_sorted_X.values()]    # the frequence for each word
print(vocab_freq[:15])
```

    [336749, 164141, 163136, 145866, 135724, 107333, 101871, 96467, 93978, 87691, 76007, 73286, 65710, 48209, 46936]



```python
bag=list()                                   # the same dictionary just in a form of list
for i ,j in zip(vocab_list,vocab_freq):
    bag.append([i,j])
print(bag[:10])    
```

    [['the', 336749], ['and', 164141], ['a', 163136], ['of', 145866], ['to', 135724], ['is', 107333], ['br', 101871], ['it', 96467], ['in', 93978], ['i', 87691]]


## Martix representaion (Features vector)
note: running this section takes a very long time


```python
def vector_matrix_review(X):
    sentence_vectors = []
    for txt,i  in zip(X,y):   #txt,i in train_set.iterrows():
        sentence_tokens = nltk.RegexpTokenizer(r"\w+").tokenize(txt)        #    data = data.lower().split()
        sent_vec = []
        for token in wordfreq_sorted_X:   
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append([sent_vec,i])
    return sentence_vectors
```


```python
len(sentence_vectors_review)
```




    25000




```python
# Bag of Words matrix represnation 
sentence_vectors_review = np.asarray(vector_matrix_review(X))  
print(sentence_vectors_review[:1])                              # we can see the matrix is very sparse
```

    [[list([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 


```python
# Bag of Words matrix represnation (Postive)
sentence_vectors_postive_x = np.asarray(vector_matrix_review(X1))  
#sentence_vectors_pos       = np.asarray(vector_matrix_review(X1)) 
```


```python
# Bag of Words matrix represnation (Negative)
sentence_vectors_negative_x = np.asarray(vector_matrix_review(X2))  
#sentence_vectors_negative   = np.asarray(vector_matrix_review(X2))
```

    

## Divide the dataset as train, development and test sets. (a)


```python
# All data
Traning_set = sentence_vectors_review[:17500]         # 70% traning.  30% developing
developing_set = sentence_vectors_review[17500:]
```


```python
# postive
train_set_pos = sentence_vectors_postive_x[:8750]
dev_set_pos = sentence_vectors_postive_x[8750:]
```


```python
# negative
train_set_neg = sentence_vectors_negative_x[:8750]
dev_set_neg = sentence_vectors_negative_x[8750:]
```


```python
sentence_vectors_review[:,1]
```




    array([1, 0, 1, ..., 0, 0, 0], dtype=object)



# (c) Calculate the following probability
* Probability of the occurrence
    * P[“the”] = num of documents containing ‘the’ / num of all documens



```python
count=0

for txt in X:
    tokens = nltk.RegexpTokenizer(r"\w+").tokenize(txt)    #txt.lower().split()
    if 'the' in tokens:
        count+=1
                
print("number of reviews containing ‘the’", count)   
print ("P[The]= ",count/ len(X))
```

    number of reviews containing ‘the’ 24793
    P[The]=  0.99172


### Conditional probability based on the sentiment
P[“the” | Positive]  = # of positive documents containing “the” / num of all positive review documents


```python
count1=0
for txt in X1:
    tokens = nltk.RegexpTokenizer(r"\w+").tokenize(txt)    #txt.split()
    if 'the' in tokens:
        count1+=1
              
print("number of postive reviews containing ‘the’", count1)   
print ("P[The| Postive]= ", count1/ len(X1))

      
```

    number of postive reviews containing ‘the’ 12381
    P[The| Postive]=  0.99048



```python
len(sentence_vectors_review[0][0])  # Number of features (the words)
```




    28764



### Occurnce of each words  (Train the model using Trianing_set)


```python
# All set
#word_sum_all = [sum(x) for x in zip(*sentence_vectors_review[:,0])]  # occurnce for each word in all the postive reviews
word_sum_all = [sum(x) for x in zip(*Traning_set[:,0])]  # occurnce for each word in all the postive reviews

print(word_sum_all[:10])
```

    [17335, 16920, 16955, 16579, 16435, 15704, 10281, 15600, 15413, 13978]


#### Conditional probablity (Postive)


```python
# Postive set
#word_sum_pos = [sum(x) for x in zip(*sentence_vectors_postive_x[:,0])]  # occurnce for each word in all the postive reviews
word_sum_pos = [sum(x) for x in zip(*train_set_pos[:,0])]  # occurnce for each word in all the postive reviews

# Caculate Porbablity for each term 
probablity_pos = [float(x)/float(len(X1)) for x in word_sum_pos]  # conditional probablity for each words in postive reviews
print(vocab_list1[:10])
print(probablity_pos[:10])
```

    ['the', 'and', 'a', 'of', 'to', 'is', 'in', 'br', 'it', 'i']
    [0.69368, 0.67936, 0.67752, 0.66376, 0.6508, 0.63312, 0.40088, 0.62096, 0.62016, 0.53616]



```python
Pos_Prob=list()                                   # the same dictionary just in a form of list
for i ,j in zip(vocab_list1,probablity_pos):
    Pos_Prob.append([i,j])
print(Pos_Prob[:10]) 
```

    [['the', 0.69368], ['and', 0.67936], ['a', 0.67752], ['of', 0.66376], ['to', 0.6508], ['is', 0.63312], ['in', 0.40088], ['br', 0.62096], ['it', 0.62016], ['i', 0.53616]]



```python
# The final P(Postive | Wi) = P(Wi | postive) * P(postive) ; prior prbablity (len(X1)/len(X))

probablity_pos_class = np.log(sum(probablity_pos))* float(len(X1)/len(X))    # use log and sum them up
print(probablity_pos_class)
```

    2.2856623163721492



```python
#probablity_pos_class = np.prod(probablity_pos)    # Because number are so hug it will convrege to zero 
#print(probablity_pos_class)                       # we use the logarithm 
```

#### Conditional probablity (Negative)


```python
word_sum_neg = [sum(x) for x in zip(*train_set_neg[:,0])]  # occurnce for each word in all the negative reviews
probablity_neg = [float(x)/float(len(X1)) for x in word_sum_neg]  # conditional probablity for each words in postive reviews
print(vocab_list2[:10])
print(probablity_neg[:10])
```

    ['the', 'a', 'and', 'of', 'to', 'br', 'is', 'it', 'i', 'in']
    [0.69544, 0.67392, 0.67752, 0.66408, 0.66496, 0.61952, 0.42016, 0.6232, 0.61208, 0.57744]



```python
Neg_Prob=list()                                   # the same dictionary just in a form of list
for i ,j in zip(vocab_list2,probablity_neg):
    Neg_Prob.append([i,j])
print(Neg_Prob[:10])  
```

    [['the', 0.69544], ['a', 0.67392], ['and', 0.67752], ['of', 0.66408], ['to', 0.66496], ['br', 0.61952], ['is', 0.42016], ['it', 0.6232], ['i', 0.61208], ['in', 0.57744]]



```python

# The final P(negative | Wi) = P(Wi | negative) * P(negative)  ; ; prior prbablity (len(X2)/len(X))

probablity_neg_class = np.log(sum(probablity_neg))* float(len(X2)/len(X))
print(probablity_pos_class)
```

    2.2856623163721492



```python
if (probablity_pos_class>=probablity_neg_class):
    print (1)
else:
    print (0)
```

    1


# (e)  Do following experiments
### Compare the effect of Smoothing


#### Conditional probablity (Postive) (Laplace)


```python
#Smoothing  Postive
probablity_pos_Laplace = [float(x+1)/float(len(X1)+2) for x in word_sum_pos]  # conditional probablity for each words in postive reviews
probablity_pos_class_Laplace = np.log(sum(probablity_pos_Laplace))* float(len(X1)/len(X))
print(probablity_pos_class_Laplace)
```

    2.4725414017202834


#### Conditional probablity (Negative) (Laplace)


```python
#Smoothing Negative
probablity_neg_Laplace = [float(x+1)/float(len(X1)+2) for x in word_sum_neg]  # conditional probablity for each words in postive reviews
probablity_neg_class_Laplace = np.log(sum(probablity_neg_Laplace))
print(probablity_neg_class_Laplace)
```

    4.939400269203754



```python
if (probablity_pos_class_Laplace>=probablity_neg_class_Laplace):
    print (1)
else:
    print (0)
```

    0


### Derive Top 10 words that predicts positive and negative class    
// I did not remove the stop words
P[Positive| word] 



```python
# Remove stop words from dict
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')) 

```

    [nltk_data] Downloading package stopwords to /Users/mamo/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


   


```python
d = dict(zip(vocab_list1,probablity_pos)) 
tokens_without_sw = [word for word in d.keys() if not word in stopwords.words()]
```


```python
d2 = dict(zip(vocab_list2,probablity_neg)) 
tokens_without_sw_negative = [word for word in d2.keys() if not word in stopwords.words()]
```

### Top 10 words that predicts positive


```python
print(tokens_without_sw[:10])   # br is not considered it is from the HTML tag
```

    ['br', 'film', 'movie', 'like', 'good', 'story', 'time', 'great', 'well', 'see']



```python
print(tokens_without_sw[1:11])
```

    ['film', 'movie', 'like', 'good', 'story', 'time', 'great', 'well', 'see', 'really']


### Top 10 words that predicts negative 


```python
print(tokens_without_sw_negative[:10]) # br is not considered it is from the HTML tag
```

    ['br', 'movie', 'film', 'like', 'even', 'good', 'bad', 'would', 'really', 'time']



```python
print(tokens_without_sw_negative[1:11])
```

    ['movie', 'film', 'like', 'even', 'good', 'bad', 'would', 'really', 'time', 'see']


// I did not remove the stop words
P[Negative| word] 

# d. Calculate accuracy using dev dataset 


```python
def classify(review):
    pos_p = 1
    neg_p = 1
    for word in review:
        if word in Pos_Prob:
            pos_p *=Pos_Prob[word] * float(len(X1)/len(X))
        if word in Neg_Prob:
            neg_p = neg_p[word] * float(len(X2)/len(X))
    if(pos_p>=neg_p):
        return 1
    if(pos_p<neg_p):
        return 0

print(classify('text'))
```

    1



```python
# based on the above example   classify('text') is 1 (postive)
predict_label = 0
for row in developing_set[:,1]:
    if row == int(classify('text')):
        predict_label+=1
        
accuracy = float(predict_label/len(developing_set))
print(accuracy)    

# we should remove stop words
```

    0.5048


###  Conduct five fold cross validation



```python
#https://machinelearningmastery.com/implement-resampling-methods-scratch-python/  refrence 

from random import seed
from random import randrange

# Split a dataset into k k
def fold_cross_validation(dataset, k):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / k)     
    for i in range(k):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


seed(1)
dataset = developing_set
k = fold_cross_validation(dataset, 5)
print(k)
```


```python
"""
accuracy_=[]        #error 
predict_label_ = 0
for fold in k:
    for row in fold[:,1]:
        if row == int(classify('text')):
            predict_label_+=1
    accuracy_.append(float(predict_label_/len(fold)))
    print(accuracy_) 
    """
```


    ---------------------------------------------------------------------------



# (f) Using the test dataset
Use the optimal hyperparameters you found in the step e, and use it to calculate the final accuracy.  


###  Test Set


```python
sentence_vectors_review_test = np.asarray(vector_matrix_review(X_test))     # feature matrix represnation ALL
sentence_vectors_postive_x_test = np.asarray(vector_matrix_review(X1_test))  # feature matrix represnation POSTIVE
sentence_vectors_negative_x_test = np.asarray(vector_matrix_review(X2_test))  # featureWords matrix represnation NEGATIVE
```

   

```python
predict_label__ = 0
for row in sentence_vectors_review_test[:,1]:
    if row == int(classify('text')):
        predict_label__+=1
        
accuracy__ = float(predict_label__/len(sentence_vectors_review_test))
print(accuracy__)    
```

    0.5


## References
* scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
* Potts, Christopher. 2011. On the negativity of negation. In Nan Li and David Lutz, eds., Proceedings of Semantics and Linguistic Theory 20, 636-659.
* github.com/weihua77/mlove_mlearning/blob/master/IMDB_movie_review/Sentiment_analysis_tutorial_youtube.ipynb
* https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
* https://www.curiousily.com/posts/movie-review-sentiment-analysis-with-naive-bayes/
