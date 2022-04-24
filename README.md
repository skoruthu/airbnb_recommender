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
