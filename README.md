# bt5153_gp
AirBnB Dataset Recommendation

<strong> File Name: </strong> sentiment_analysis.ipynb <br>
<strong> Packages used:</strong> VADER, texthero, nltk, pandarallel, pandas, numpy <br>
<strong> Description: </strong><br>
This notebook does sentiment analysis of the listing reviews files.
The input is the reviews which were pre-processed with VADER to do lemmatization without cleaning.
The sentiment analysis done was polarity scores of the reviews, which are then grouped into either Positive, Negative or Neutral sentiments. Reviews which were not in English were dropped.<br>

<strong> File Name: </strong>london_nearest_tube.ipynb <br>
<strong> Packages used:</strong> pandas, numpy, geopy <br>
<strong> Description: </strong><br>
Merge listings dataset with London stations dataset so that for each listing, there are 3 new features included, nearest stations to listing, listing’s distance to nearest station and listing’s walking distance to station.<br><br>

<strong> File Name: </strong>content_based_recommendation.ipynb <br>
<strong> Packages used:</strong> pandas, numpy, sklearn, preprocessing (see file description below) <br>
<strong> Description: </strong><br>
This file returns top 10 most similar listings to a particular listing. Before running the recommendation algorithm, a few key steps was done: 
(1) Merge Data & Feature Engineering: listing data was merged with host-related and property-related engineered features, sentiment and location data. 
(2) Feature Selection: Relevant features were selected.
(3) Preprocessing & Normalisation: Features were preprocessed and normalised so that cosine similarity calculation can be done
(4) Recommendation Results: The normalized features were done used to compute similarity matrix and top 10 listings which had the highest similarity to a listing will be returned. <br><br>

<strong> File Name: </strong>preprocessing.py <br>
<strong> Packages used:</strong> pandas, numpy, re, collections, ast <br>
<strong> Description: </strong><br>
This file contains functions to preprocess dataframes, which included drop unused features, remove inactive listings, expand features, imputation, log-transform skewed columns, Winsorization, one hot key encoding. It also contains a function which expanded amenities features. This preprocessing file was imported and used in the content_based_recommendation.ipynb and in the models.ipynb. <br><br>

<strong> File Name: </strong>eda.ipynb <br>
<strong> Packages used:</strong> pandas, seaborn, numpy, matplotlib <br>
<strong> Description: </strong><br>
This file contains some of the graphs used in the report. <br><br>
