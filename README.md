# CSE-151A-Project-
Team members: Emily, Joey, Risab, Christine (Qingtong), Viraj, Sebastian, Justin, Armaan

3/10/24 Note: Joey Kaminsky improved Model 1 (from Checkpoint 3) by 5% (up to 65% testing accuracy) by using class weights, early stopping, and utilizing softmax in both the hidden and output layers. This is due to the imbalance present in our data, where getting the class weights can help the model understand the imbalance data better. Additionally, the addition of early stopping and softmax allow the model to achieve high accuracies by avoiding overfitting issues and explore different activation functions that best fit the problem at hand, respectively. While this is a slight improvement to our model, we are still dealing with the issue of the model underfitting due to trying to predict the first class too often–due to its higher frequency in the data. We try to fix this issue in the SVM model (model 2). The updates can be found in the notebook `Milestone_3_JK_Updates.ipynb` or at this link: https://colab.research.google.com/drive/10ver3YLrlP45MuUj6kbIVuZaNE3fF7jx#scrollTo=L7tX2TnysAvE.

# Milestone 4

#### 3. Evaluate training versus testing error.
Our SVM Classifier model from Milestone 4 has a training error of 0.63 and a test error of 0.60.

For context, our Milestone 3 Model 1 had a training error of about 65%, with testing error around 59.5%.

Overall, the training error decreased while the test error increased trivially.

#### 4. Where does your model fit in the fitting graph?

This SVM model is less overfitting. In the fitting graph, it would be more left (closer to ideally fitted) compared to the first model.
In contrast, the Model 1 usually only predicts $$ with only a few rare outlier predictions of $$$. It therefore had a high training and test error.
This second model (SVM) predicts more evenly (possibly thanks to the class weights)

#### 5. Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?

We performed:
Hyperparameter grid search: This took about 5+ minutes to run and iterated through C, gamma, and kernel options. This increased training error and overall accuracy but decreased test error. Eventually, we deemed this change to not be helpful in increasing model performance (not implemented in final model)
Class rebalancing: We used `imblearn.over_sampling` to rebalance the dataset. While this increased accuracy by about 5% (compared to the unbalanced dataset), we determined that changing class weights instead of resampling actually increased performance more. Therefore we switched to class weights (not implemented in final model)
Class weights: Fed in class weights to the SVM model to help it increase accuracy. It increased performance (overall accuracy) by about 10%.

#### 5. What is the plan for the next model you are thinking of and why?

The next model we are going to try is XGBoost (a boosted decision tree model). It is known for being a powerful model and performing well when synthesizing complex tabular data. Hopefully it will capture the complex nature of our data and class imbalances. Another positive is that there is less preprocessing needed for the decision tree to understand/leverage features.

#### 6. Colab Link: 

https://colab.research.google.com/drive/1fVuWdEJfC3t6FoNlaJaenktjRcf1XFpx?usp=sharing

#### 7. Conclusion

The conclusion of our 2nd model is that it performed well/better than the first model. We can see from decreasing train/test error as well as looking at the raw predictions that it is predicting a more diverse range of outputs. (I.e., it’s not just predicting the same value for any input).
To possibly improve model performance we could:
Continue to work with the class rebalancing/class weights to see if changing/iterating those would improve performance
Feature expansion on the existing features (more context/data for the model to work with)
Vectorize the text inputs that our proc.csv provided ⇔ gives the model more data about reviews, restaurant names, etc. Possibly perform more complex word vectorization like sentiment analysis, positive/negative sentiment as well.
Increase dataset size (go back into the data processing stage, use more samples) ⇔ leverage unreasonable effectiveness of data to improve model performance

# Milestone 3

#### 3. Evaluate training versus testing error.

Our training error is about 65%. Our testing error is 59.5%. While these numbers are high, we suspect that this is a result of an imbalanced dataset, and in reality our model is underfitting to the prediction problem. More information on that in section 4.

#### 4. Where does your model fit in the fitting graph?

We believe our model is underfitting the classification problem. The only reason we’re seeing the difference between training and testing error is because it’s consistently predicting the first class (`$$`) and the training data has a lot of that class. Therefore the model predicts with high accuracy (about 65%) in training. However, in testing, where the test data has a different class distribution, the accuracy is lower (see the confusion matrix).
This suggests that the model is underfitting.

Our data consists of unbalanced distribution of restaurant price categories, where the overwhelming majority have two dollar signs. This led our first model to underfit our data, where it predicts almost the same class (a price category of two dollar signs, `$$`) every time regardless of the data. This can be seen by looking at the confusion matrix: 

<img width="149" alt="Screenshot 2024-02-25 at 1 27 28 PM" src="https://github.com/jkaminsky2/CSE-151A-Project-/assets/8013994/b3231a51-185b-4198-b376-d2bd8507c9c5">

Each column represents the predicted class. We can see an overwhelming majority of predictions lie in the first column for class 1 (`$$`) with only 11 predictions for class 2 (`$$$`) and no predictions for class 3 (`$$$$`).
Each row represents the true class value. These rows show that in reality there are 105 predictions in class `$$` (see first row summed), 100 predictions for class 2 (`$$$`), etc.
Basically, we can see that the model consistently predicts the first class (`$$`) and most test cases are `$$`. 

This suggests the model predicts the most common class most of the time, but doesn't understand how to differentiate classes. This suggests **underfitting.**

#### 5. Which are the next 2 models you plan to add and why?

In our future work, we plan to implement an XGBoost model and an SVM to determine whether different model architectures can better understand and predict this classification problem.

XGBoost implements an ensemble of models trained to overcome the weaknesses of the others overlaid on a base decision tree model. The nature of the model supports the finetuning of various hyperparameters (max depth, learning rate, lambda, alpha, etc.) to counteract overfitting. These additional parameters could be particularly helpful as we attempt to overcome issues regarding underfitting and overfitting.

SVM was described in class and we think it would be interesting to apply to a multiclass classification problem, especially using the kernel trick to add dimensionality and possibly create better decision boundaries.

#### 6. Colab Link: 

https://colab.research.google.com/drive/1rsKRfCgZOV-R0u1Pj4pfU-P5DVu6IIv6?usp=sharing

#### 7. Conclusion

For our first model, we decided to use a DNN with hyperparameter tuning to predict price category. We selected this model as it can take in complex data and apply functions to predict multiclass labels. To implement this model, we used a similar method to the one used in homework 2, where we test different combinations of activation functions, nodes in hidden layers, learning rate, and number of epochs to see which combination can produce the best accuracy. We hard coded the following: 4 layers to make the neural network complex but not too complex, the loss function binary_focal_crossentropy (due to the class imbalance present in our data), and the output layer activation function to be softmax–so that probabilities for each prediction summed up to 1.

We chose not to use image data for our first model, despite the dataset providing this information. This is because we reasoned that images attached to Google restaurant reviews could be anything: food, the outside of the restaurant, a bug, something completely random. Therefore it might not be very indicative of the price of the restaurant. Moreover, classifying these images as “what they are” and then deciding the impact on price might take too much resources for a Google Colab notebook. Therefore we decided to not use the image data provided.

As described in section 4, we believe our model is underfitting on the prediction problem. This could be a result of data imbalance (there are only 3 test cases which are $$$$, so the model might learn to never predict $$$$ to maximize accuracy). This could also be a result of a small dataset; a test size of 200 and a training size of about 1700 is not usually large enough for an ANN to properly grasp data distributions.

There are multiple ways we can improve this model. For one, we could add more layers to the neural network to better understand and “break down”/understand the data inputs. Another huge way is to understand the different activations functions and which one works the best given our model along with the optimiser which would improve with hyperparameter tuning. We could also try to balance the dataset to get better overall accuracy for each class.

As mentioned previously, one solution could be to add more data to the model. The limited size of our dataset (due to the computational complexity of our preprocessing steps and the RAM limitations of Colab) may have contributed to the underfitting our initial model experienced. However, since we are using Colab with limited RAM, we do not konw if this is technically feasible. The data preprocessing pipeline (including translation) is very computationally taxing. So increasing the amount of data may or may not be possible given our current resources.

# Milestone 2
Colab Notebook link: https://colab.research.google.com/drive/1bjhNxm6oj0MIGyFm1_xxBMnk-qcns9mw?usp=sharing#scrollTo=xy1WmqThhDQh

You can also find the downloaded Jupyter Notebook in the Github directory.

### Data Preprocessing
Our data preprocessing pipeline takes several steps.

#### 1\. Load our data

1A. We downloaded two Google datasets: "places.clean.json" and "reviews.clean.json". These datasets can be found here: https://datarepo.eng.ucsd.edu/mcauley_group/data/googlelocal/
- "places" stores Google locations
- "review" stores the corresponding Google reviews.

For more information on these 2 datasets, see ["Table descriptions"](#table-descriptions)

When loading the two datasets:
- Loaded only the first 5,000 reviews for this EDA
- Dropped unnecessary columns when loading datasets
  - Removed `phone`, `hours` from places dataset
  - Removed `reviewerName`, `phone`, `hours` from review dataset 
- Used `urllib` and `gzip` to access these files straight from the CSE webpage URL.

1B. Merged the two datasets together using the primary key "gPlusPlaceID" (which represents the Id associated with the business).
- Used a left join on the reviews dataset

#### 2\. Clean existing columns
- Converted price (originally a *string* ranging from "$" to "$$$$$") into an ordinal variable
- Converted reviewTime (originally a *string*) into a datetime object
- Converted `gps` into a geographical object so we can compare distance, geography, etc. between restaurants as a feature

#### 3\. Create new columns
- `language`: *String* - Language of the original review. Used a Google Translate API on reviewText
- `translated`: *String* - English text of the review, translated into English if not already in English. Used a Google Translate API on reviewText
- `top_category`: *String* - Top category associated with the business. Used TF-IDF
- `relevant_cat`: *bool* - True if top_category is related to a restaurant, False if otherwise. Used a Word2Vec machine learning model to identify relevance of the words in `top_category` compared to the word "restaurant" and used a threshhold to categorize True/False relevance.
- `cat_cluster`: *float64* - Cluster restaurants into one of 5 categories based on top_category. Used a Kmeans clustering model to group restaurants based on their `top_category`. For more information about the groups generated, look at the EDA code
- `sentiment`: *object* - Sentiment analysis of the review. Includes negative, neutral, and positive sentiment as well as the compounded sentiment (overall sentiment based on negative, neutral, and positive sentiment). Used the vaderSentiment package to get the negative, neutral, and positive sentiment from `translated`; from this, we also got the composite sentiment to see whether the rating was positive, negative, or neutral overall.
- `charged_words`: *object* - List of words that influenced the sentiment analysis of the review. Generated from vaderSentiment (see `sentiment`)
- `price_words`: *object* - List of words relating from the review that relate to their opinion of the pricing of the business. (EX: "overpriced", "great deal", "not worth the cost"). Generated from vaderSentiment (see `sentiment`)
- `longitude`, `latitude`: *String*, *String* - We convert the GPS coordinates provided in string format ([latitude, longitude]) into two separate columns, namely latitude and longitude.
- `geo_Kmeans`: *Int* - To identify patterns and group restaurants with similar geographical locations, we implement KMeans clustering. Currently, the number of clusters is set to 200, which can be changed later.
- `location_obj`: *object* - We leverage the GPS pairs to generate a detailed location object for each restaurant. This object includes essential geographical information such as the road, city, postal code, and country.
- `class_$$`, `class_$$$`: *Int* - In addition to geographical details, we hot-encode the price bracket of each restaurant. If the restaurant has a price = $$, there is a 1 in the `class_$$` column and a 0 in the `class_$$$` column. Vice-versa if the price = $$$.

#### 4\. Imputation
- `reviewTime`, `unixReviewTime`: `None` is imputed with mean time
- `name`: `None` is imputed with 'Unknown'
- `closed`: `None` is imputed with 'False' (assumes the place/location is still open)

#### 4\. Normalization
- `price`: normalized all symbols to USD (from other currency characters like yen (¥), euro (€), etc.)

### Dataset Descriptions
#### ***Reviews Dataset***:

- **Row** - Corresponds to one review
- **Columns (6)**
    - **rating**: *float64* - A value from [1, 2, 3, 4, 5]
    - **reviewText**: *object* - review text of the specific business
    - **categories**: *object* - List of Categories associated with the business
    - **gPlusPlaceId**: *object* - Id associated with the business
    - **unixReviewTime**: *float64* - Represents the time the review was made in Unix Formatted time
    - **reviewTime**: *object* - Represents the time the review was made in human-readable format

#### ***Places Dataset***:

- **Row** - Corresponds to one place
- **Columns (6)**
    - **name**: *object* - Name of the place
    - **price**: *object* - categorical price-range of the business, ranges from $ to $$$$
    - **address**: *object* - Address of the place
    - **closed**: *bool* - True if place is closed False otherwise
    - **gPlusPlaceId**: *object* - Id associated with the place
    - **gps**: *object* - Latitude and Longitude of location of the place

#### ***Added Columns****
- **Row** - Corresponds to one review
- **Columns (8)**
    - **language**: *String* - Language of the review
    - **translated**: *String* - English text of the review, translated into English if not anlready in English
    - **top_category**: *String* - Top category associated with the business
    - **relevant_cat**: *bool* - True if top_category is related to a restaurant, False if otherwise
    - **cat_cluster**: *float64* - Cluster restaurants into one of 5 categories based on top_category
    - **sentiment**: *object* - Sentiment analysis of the review. Includes negative, neutral, and positive sentiment as well as the compounded sentiment (overall sentiment based on negative, neutral, and positive sentiment)
    - **charged_words**: *object* - List of words that influenced the sentiment analysis of the review
    - **price_words**: *object* - List of words relating from the review that relate to their opinion of the pricing of the business
