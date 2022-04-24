import re
import pandas as pd
import numpy as np
import collections
import ast


def preprocess_df(data):
    ## Preprocess a standard Airbnb dataset
    df = data.copy()

    ########################################################################
    ## Datatypes
    if df["price"].dtypes == "O":
        df["price"] = df["price"].apply(lambda x: re.sub("[\$,]", "", x))
        df["price"] = df["price"].astype(float)

    df["last_review"] = pd.to_datetime(df["last_review"], yearfirst=True)
    df["host_since"] = pd.to_datetime(df["last_review"], yearfirst=True)
    df["first_review"] = pd.to_datetime(df["last_review"], yearfirst=True)
    df["host_since"] = pd.to_datetime(df["host_since"], yearfirst=True)

    ########################################################################

    ## Drop features not used
    not_relevant = [
        "listing_url",
        "scrape_id",
        "last_scraped",
        "picture_url",
        "host_about",
        "host_thumbnail_url",
        "host_verifications",
        "name",
        "description",
        "neighborhood_overview",
        "host_name",
        "host_picture_url",
        "calendar_last_scraped",
        "license",
        "host_url",
    ]
    empty = ["calendar_updated", "bathrooms"]
    dup = ["host_neighbourhood", "neighbourhood", "host_listings_count"]
    not_useful = [
        "minimum_minimum_nights",
        "maximum_minimum_nights",
        "minimum_maximum_nights",
        "maximum_maximum_nights",
        "neighbourhood_cleansed",
        "beds",
        "host_location",
        "host_response_time",
        "host_response_rate",
        "host_acceptance_rate",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms",
        "calculated_host_listings_count",
        "calculated_host_listings_count_entire_homes",
    ]

    df = df.drop(not_relevant + empty + dup + not_useful, axis=1)

    ########################################################################

    ## Remove inactive listings
    df["active"] = np.nan

    # Find those who joined before 5/12/20 (i.e. 1 year before the data scraped date)
    filter_date = pd.to_datetime("2020-12-05")
    df_old = df[df["host_since"] < filter_date]
    df_recent = df[df["host_since"] >= filter_date]

    # Old hosts defined as 25> percentile. Drop inactive.
    df_old["active"] = [1 if i >= 0.09 else 0 for i in df_old["reviews_per_month"]]
    df_recent.loc[:, "active"] = 1
    # Get the active ones from df_old
    df_old = df_old[df_old["active"] == 1]
    # combine & get back original df
    df_combined = pd.concat([df_old, df_recent], axis=0)
    df = df_combined.drop("active", axis=1)

    # Drop rows where price is 0
    df = df[df["price"] != 0]

    ########################################################################

    ## Expand features
    df["bathrooms_text"] = df["bathrooms_text"].str.lower()
    df["bathroom_num"] = df["bathrooms_text"].str.extract(
        pat=r"(^[0-9]*[\.0-9]*)"
    )  # Capture the numbers at the beginning, including decimals
    df["bathroom_num"] = np.where(
        df["bathrooms_text"].str.contains(r"half-bath", regex=True),
        0.5,
        df["bathroom_num"],
    )  # extract half-bath as 0.5
    df["bathroom_num"] = df["bathroom_num"].astype("float")
    df["bathroom_sharing"] = df["bathrooms_text"].str.extract(pat=r"(shared|private)")

    col_to_drop = ["bathrooms_text"]
    df = df.drop(col_to_drop, axis=1)

    ########################################################################
    ## Imputation
    for col in df.columns:

        # categorical dtypes
        if df[col].dtype == object:
            df[col] = df[col].fillna("0")

        # date dtypes
        if df[col].dtype == "<M8[ns]":
            df[col] = df[col].fillna(pd.to_datetime("2200-01-01"))

        # numeric types
        if df[col].dtype in ("int", "float"):
            df[col] = df[col].fillna(0)

    ########################################################################

    ## Log-transform skewed non-0 variables. Others transformed in pipeline
    df["price_log"] = df["price"].apply(np.log)

    ########################################################################

    ## Winsorization
    win_dict = {
        "bedrooms": df["bedrooms"].quantile(0.97),
        "minimum_nights": df["minimum_nights"].quantile(0.95),
        "maximum_nights": df["maximum_nights"].quantile(0.95),
        "minimum_nights_avg_ntm": df["minimum_nights_avg_ntm"].quantile(0.95),
    }

    for column, value_cutoff in win_dict.items():
        df[column] = df[column].clip(upper=value_cutoff)  # cap values above the cutoff

    ########################################################################

    ## Encoding
    df = df[~(df["host_identity_verified"] == "0")]
    encode = {"t": 1, "f": 0}
    df = df.replace(
        {
            "host_is_superhost": encode,
            "host_has_profile_pic": encode,
            "host_identity_verified": encode,
            "has_availability": encode,
            "instant_bookable": encode,
        }
    )

    return df


def clean_amenities(data, num_amenities=20):

    df = data.copy()
    df["amenities"] = df["amenities"].str.lower()
    amenities = df["amenities"].apply(lambda x: ast.literal_eval(x))

    amenities_combined = []
    for sublist in amenities.tolist():
        for item in sublist:
            amenities_combined.append(item)

    counter = collections.Counter(amenities_combined)
    most_common = counter.most_common(num_amenities)

    # Convert dictionary to dataframe
    counter_df = pd.DataFrame.from_dict(counter, orient="index").reset_index()
    counter_df.columns = ["amenity", "num_listings"]
    counter_df = counter_df.sort_values("num_listings", ascending=False)

    amenities_list = counter_df["amenity"][0:num_amenities]
    for value in amenities_list:
        df["amenities_{}".format(value)] = df["amenities"].str.contains(
            value, regex=True
        )
        df["amenities_{}".format(value)] = df["amenities_{}".format(value)].astype(int)

    ## Drop columns
    preprocessing_cols = ["amenities"]

    df = df.drop(preprocessing_cols, axis=1)

    return df
