# CSE-151A-Project-
Team members: Emily, Joey, Risab, Christine (Qingtong), Viraj, Sebastian, Justin, Armaan

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
