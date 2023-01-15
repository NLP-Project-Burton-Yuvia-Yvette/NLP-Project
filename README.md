# CodeUp-DS-NLP-Project
 
### Project Goals 
* The goal of this classification project is to first identify key words of the programming language and create a machine learning model that can effectly predict the programming language used.
### The Plan
* Aquire ReadMe data from GitHub repositories via webscraping.
* Prepare data for exploration by:
    * Convert text to all lower case for normalcy.
    * Remove any accented characters, non-ASCII characters.
    * Remove special characters.
    * Lemmatize the words.
    * Remove stopwords.
    * Store the clean text and the original text for use in future notebooks.
#### Explore data in search of key features with the basic following questions:
* What is the average word count?
* Which language is the most used?
* What are the top words that are most popular?
* Is there a difference in sentiment by programming language?
* What are the top 20 most frequent bigrams?
* What are the top 10 unique words to Python?
#### Develop a Model to predict happiness score
* Use key words identified to build predictive models of different types
* Evaluate models on train and validate data samples
* Select the best model based on accuracy
* Evaluate the best model on test data samples
#### Draw conclusions

### Steps to Reproduce
* Clone this repo.
* Acquire the data from GitHub
* Put the data in the file containing the cloned repo.
* Run notebook
### Conclusions
* Decision Tree model Accuracy scores:
    
        * 0.704762 on training data samples
        * 0.637363 on validate data samples
        * 0.671052 on test data samples
        
#### Key TakeAway:
    Decision Tree model was successful on all train, validate and test data sets. 
### Recommendations

   * Consider aquiring larger "text" datasets
   * Consider hyperparameter tunning
   * Consider gradient boosting algorithims