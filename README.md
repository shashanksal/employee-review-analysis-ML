# employee-review-analysis-ML
A machine learning project that uses a Glassdoor dataset for employee review analysis

In this report, we try to create machine learning models that can predict the employee sentiments based on the textual reviews on popular professional website called Glassdoor. By formulating a model, we can try to predict the sentiments of the users by accurately predicting the numerical ratings based on the text provided. Studies have shown that the employee sentiments are directly proportional to their work output and hence the company itself.

## Introduction
  Sentiment analysis is a branch of study that involves analysing individual’s opinions, sentiments, attitudes and emotions in form of text or writings. Due to the importance of this field of study in business and society as a whole, study of sentiment analysis has grown beyond the scope of just Computer Science to be included in management sciences as well as social sciences. The significance of sentiment analysis is increasing simultaneously with the growth of social media in form of reviews, blogs, discussions, tweets, etc. Sentiment analysis is employed in quite a large number of business and social domains as options are crucial to practically entire human activities and they(opinions) impact our behaviour. [1]

  For the current project, I have used a dataset retrieved from Glassdoor where tech employees from top five reputed corporations have written their opinions and ratings about their respective companies. These are all reviews in text format are qualitative in nature. The problem with qualitative analysis is that they can be inaccurate sometimes and it’s difficult to arrive at a conclusion with just review analysis. If qualitative analysis is supplemented with quantitative research, then this problem can be resolved. Luckily for us, Glassdoor allows users to rate their companies in ratings (0-5). So, the idea of the project is to predict the numerical ratings based on the textual reviews. Another aspect considered in the project is to predict the positive and negative words that constitutes for the pros and cons in the overall rating. The meaning of pros and cons is discussed in the dataset section.
  
## Data Description
  The data that I used for my study was scrapped from Glassdoor website. I used a readily available data source from Kaggle.com [2] It’s an extensive dataset containing reviews of over 67k employees. The reviews are segregated based on the employee’s present or previous organization they worked with. The companies are majorly the top tech companies like Google, Apple, Microsoft, Amazon and Netflix. The dataset is divided into 67k rows and 17 columns or variables.
  
## Methods and Results
* ### Data Preparation
  The data that was directly scrapped from Glassdoor and downloaded from Kaggle, was a noisy dataset. This was because the reviews contained use of special characters, incorrect spellings which needed to be corrected in order to use the dataset for our analysis. Thus, the original dataset was subjected to extensive number of pre- processing steps in order to standardize the data and reduce its size as well. For my project, I have concentrated only on pros and cons columns for the qualitative data (text). For the quantitative data, I have concentrated on overall-ratings column.
  Some of the tasks that were involved in pre-processing were
  * Converting the reviews to lower case.
  * Stripping off extra spaces and quotes from the reviews.
  * Replacing more than one spaces with a single space.
  
  I have used ‘nltk’ (Natural Language Toolkit) package available in python to tokenize the reviews, convert the reviews to lower case.
  
 * ### Preliminary data analysis
    I was able to find average length of the pros and cons reviews along with their min and max length.
    
    I was able to perform frequency analysis of the length of each review against the frequency of occurrence.
    
    The preparatory data analysis of the data made it possible to create graphical visualisation of the pros and cons in the form of a Wordcloud. A Wordcloud is a visual representation of words used in particular piece of text. The words are of different sizes according to how often they are repeated in the text. [8] The more number of times a particular word was repeated in either pros or cons columns, the more prominent was its size.
    
 * ### Naïve Bayes Classification algorithm
    Naïve Bayes is a simple classifier model that can be used for text classification. 
    
    After pre-processing the data, I was able to divide the data into training size of 80% and test size 20%. Multinomial Naïve Bayes classifier was used from python’s sklearn library. Training set and Testing set classification report was created using the split dataset. The classification report shows a representation of main classification metrics on a per class basis. Correspondingly, a training and testing accuracy of the model was obtained using accuracy score. The model gave out accuracy of little over 84% in both the test and train dataset.
    
 * ### Logistic Regression
    Logistic Regression is a classification model that is easy to implement and performs well on linearly separable classes. It is one of the most widely used algorithm in classification industry.
    
    An input is defined for the algorithm. Each sample will act as input for the dataset under consideration. Each sample consists of several features. For algorithm to learn, we need to define variables that we can adjust accordingly.
    
    One of the drawbacks of the model is if we train too many epochs, we risk problem of overfitting. If we train too little, the model will fail to find pattern and the prediction accuracy is low.
    
    For training the model we evaluated Bag of Words and TF-IDF features In bag of words model, a text is represented as a bag or multiset of words disregarding the grammar as well as word order but maintain the multiplicity.
    
  * ### Support Vector Machine and Random Forest Classifier
     SVM or Support Vector machines is a supervised learning model. It is a non- probabilistic binary linear classifier. For a set of training points (x,y), we need to find maximum margin hyperplane that divides the points with y=1 and y=-1.
     
  * ### Deep Learning
     Deep learning models like CNN can be employed to predict the ratings to a better extent. The deep learning models are inspired by how a human brain works. Deep learning models give out more accuracy as long as more data is fed to the system.
    CNN is a class of deep feed forward artificial neural network wherein the connections between nodes do not form a cycle. CNN uses a variation of multilayer perceptron which is uniquely designed to achieve minimal pre-processing.
    Unfortunately, I failed to perform a satisfactory CNN on my dataset and hence I had to conclude my study.
    
## Conclusion and Further Study
While my study was abruptly ended, I did manage to create a few traditional machine learning models that accurately predicted the ratings based on the text. While naïve Bayes did provide me the best accuracy rates, naïve Bayes classifier has a drawback of assuming that the data is distributed independent of the output class. Among the other traditional models, SVM performed best with Bag of Words features introduced. Further study can be made to improve the bag of words features as well as tokenisation. This will give out better results in form of accuracy ratings for each model. A Deep learning model can be re-created for best predictions as Deep Learning models perform better than traditional machine leaning models.

