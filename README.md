# CS7641 Project Fake News Detection Algorithm
#### Kai Sun, Xiyang Wu, Xuesong Pan, Ruize Yang

## 1. Introduction 

Fake news on social media has experienced a resurgence of interest due to the recent political climate and the growing concern around its negative effect. Not only does it provide a source of spam in our lives, but fake news also has the potential to manipulate public perception and awareness in a major way. The amount of disseminated information and the rapidity of its diffusion make it practically impossible to assess reliability in a timely manner, highlighting the need for automatic fake news detection systems. In that spirit, our project centers around building a fake news detect system for text content based on data collected from news articles.

 

We start with a brief description of the data set used in the development of this system and the metric we used to measure performance, followed by each of the various approaches and algorithms we implemented.

## 2.Data set and Basic idea

The data set used in training and testing the detection systems comes from Kaggle fake news. Kaggle is an online community of data scientists and machine learning practitioners and offering public datasets for algorithm testing. Kaggle fake news dataset is a set of 20799 news article with fake (or not) label. Each data has 5 attributes: 


1. id: unique id for a news article;
2. title: the title of a news article; 
3. author: author of the news article; 
4. text: the text of the article (could be incomplete); 
5. label: a label that marks the article as potentially unreliable, 1 for unreliable and 0 for reliable.

### Vectorize the sentences

The method we use to process the raw sentences is the Doc2Vec method. The Doc2Vec method [5] is a modified algorithm based on the Word2Vec [6] method. The basic idea for the Word2Vec method is to map each word in the sentence into the vector and use the words in the rest part of the article to predict the next word. The prediction could be taken as a multi-class classification problem, while the generated vector contains the probability that each word in the dictionary exists in the next position. The Doc2Vec method introduces an extra unit on the top of each sentence that represents the ‘topic’ of the given sentence. In this case, each sentence could be mapped into the vector space and processed together, instead of being processed separately. In our project, the Doc2Vec method is used to process the dataset for all the machine learning method except the LSTM methods.

 In this project, our work mainly surrounding text analysis of fake news. We focused on mining particular linguistic cues, like, by finding anomalous patterns of pronouns, conjunctions, and words associated with negative emotional word usage. For example, that fake news often contains an inflated number of swear words and personal pronouns. The raw news be represented using the Bag-of-Words model, which then be modified to incorporate the relativity between the words. We used both supervised and unsupervised learning here.  For unsupervised learning, we used GMM and Kmeans; for supervised learning, we take a try at LSTM and a two layer neural network.

## 3.Unsupervised Method



## 4.Supervised Method



### Naive Bayes Classifier 

Naive Bayes is a relatively simple algorithm, but it is often found to have very good results in the field of NLP content analysis. Here we use the naive Bayes class library in scikit-learn to classify the data. Compared to decision trees, algorithms like KNN, Naive Bayes requires fewer parameters. In scikit-learn, there are 3 naive Bayes classification algorithm classes. They are GaussianNB, MultinomialNB and BernoulliNB. Among them, GaussianNB is naive Bayes with priori Gaussian distribution, MultinomialNB is naive Bayes with priori polynomial distribution, and BernoulliNB is naive Bayes with priori Bernoulli distribution.

## 5. Deep Learning Method

Deep learning based method is an effective tool in the document classification field, especially for the tasks with complicated features. Generally, deep learning based method is a subclass of the supervised learning, while the neural network is trained by presenting the expected label for each data point. In this project, we intend to present two kinds of Deep learning based methods: the traditional backward-propagation neural network and the LSTM network.

### 5.1 Backward-propagation Neural Network

The traditional backward-propagation neural network [7] uses the fully-connected layer to store the learnt parameters. The forward propagation process generates the output with the current parameter, and the backward propagation process uses the error between the current output and the ideal one to update the weight for each neural unit in each iteration. 

The neural network for the fake news classification task has three layers in general. The structure of the neural network is shown in Fig. 1. The first layer, which is the layer to read the primitive training data, has 300 input channels and 256 output channels. The hidden layer in the middle has 256 input channels and 80 output channels. The third layer, which the last one, has 80 input channels and 2 output channels. The activation function used between each layer in the middle is the ReLU function, while the activation function for the output layer is the sigmoid function, in order to fix the scope for the output values. To avoid the potential overfitting issue, which is common in the NLP tasks, two dropout layers are introduced between the first and second layer, and the second and third. These layers could randomly drop the neural nodes in the previous layer during the training process. To discuss effect of the dropout layers on the network, the dropout rate is changeable during the implementation.

### 5.2 LSTM Network

Long Short-Term Memory (LSTM) Network [4] is a neural network with recurrent structure. This deep learning model reveals impressive capability in processing the continuous sequences, since this model could find out and utilize the relationship between different samples in the same sequence. In this project, we will design two kinds of LSTM network for fake news detection, the LSTM network with single propagation direction and the bi-directional LSTM network.
The structure of the neural network is shown in Fig. 2. Unlike other method, the vectorize method for the LSTM network is the word embedding method.  In this case, the sentence read by the network is an array with fixed length, and the content in the index of each word in the dictionary that contains the words with highest frequency. The vectorized sentences are sent into the embedding layer, so that each word in the sentence will be presented as a vector that presents its similarity with other words. The vector length for each word is 10. The LSTM network used for this project includes 300 input channels, which corresponds to the length of each sentence, and 150 hidden states. The activation function for the LSTM layer is ReLU. The sum of the elements in the vector that represents each word is calculated after the LSTM layer, and the result is sent into the fully connected layer with 150 input channels and 2 output channels. Two dropout layers locate after the embedding layer and the LSTM layer. The activation function for the output layer is the sigmoid function.

As a modified version of the original LSTM network, bidirectional LSTM network (Bi-LSTM) could encode the input sentences in both directions, which makes it capable to process more complex sentences. The general structure of the Bi-LSTM network is similar to the original LSTM network. The main difference between these network is that the Bi-LSTM network has LSTM units for backward propagation, so that the output size for the Bi-LSTM layer is twice as the original LSTM layer.

## 6. Evaluation and results 

![](https://github.com/ksun86/ML-Project/blob/master/fig1.png)  

![](https://github.com/ksun86/ML-Project/blob/master/percent_capital.png)  

![](https://github.com/ksun86/ML-Project/blob/master/similar_words.png)  

![](https://github.com/ksun86/ML-Project/blob/master/title_length.png)   Example Readmes



### Results for Naive Bayes Classifier 

We tried all three naive Bayesian models of sklearn. Among them, the polynomial model and the Bernoulli model perform relatively well, and can reach 0.902 and 0.908 after parameter adjustment(show in below), respectively. but the Gaussian model performs poorly, reaching only 0.7. The reason why the Gaussian model is less effective may be because it is mainly used in continuous random variables, but text analysis belongs to discrete variable analysis. The difference between the polynomial model and the Bernoulli model have different calculation granularities. The polynomial model uses words as the granularity, and the Bernoulli model uses files as the granularity. Therefore, the calculation methods of the a priori probability and the class conditional probability are different. When calculating the posterior probability, for a document B, in the polynomial model, only the words that have appeared in B will participate in the posterior probability calculation. While in the Bernoulli model, if a word does not appear in B but appeared in the global word list, those words will also participate in the calculation, but only as the "counter party". Therefore, the judgment criterion of the Bernoulli model is more comprehensive, which may be the reason why it is slightly better than the polynomial model. 


<img src="/MultinomialNB.png" width = "450" height = "450" alt="MultinomialNB.png"  /> <img src="/Bernoulli.png" width = "450" height = "450" alt="Bernoulli.png"  />
 


However, the Naive Bayes model only classifies from a priori probability point of view, its classification effect will be worse than the deep learning model which have more complicate structure and much more parameters.



## 7.Conclusions



## References

1. Ward A, Ross L, Reed E, et al. Naive realism in everyday life: Implications for social conflict and misunderstanding[J]. Values and knowledge, 1997: 103-135.
2. Nickerson R S. Confirmation bias: A ubiquitous phenomenon in many guises[J]. Review of general psychology, 1998, 2(2): 175-220.
3. Shu K, Sliva A, Wang S, et al. Fake news detection on social media: A data mining perspective[J]. ACM SIGKDD Explorations Newsletter, 2017, 19(1): 22-36.
4. Thota, Aswini, et al. “Fake News Detection: A Deep Learning Approach.” SMU Data Science Review, vol. 
5. Le, Quoc, and Tomas Mikolov. "Distributed representations of sentences and documents." International conference on machine learning. 2014.
6. Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.
7. Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. Learning internal representations by error propagation. No. ICS-8506. California Univ San Diego La Jolla Inst for Cognitive Science, 1985.
8. Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.

