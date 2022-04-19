import re
import pandas as pd
import numpy as np
import collections
import ast

def preprocess_df(data, verbose = True):
    
    df = data.copy()
    
    ####### FIX DATATYPES #######
    if df['price'].dtypes=='O':
        df['price'] = df['price'].apply(lambda x: re.sub('[\$,]','',x))
        df['price'] = df['price'].astype(float)
    
    #Convert to datetime
    df['host_since']=pd.to_datetime(df['host_since'],yearfirst=True)
    df['first_review']=pd.to_datetime(df['last_review'],yearfirst=True)
    df['last_review']=pd.to_datetime(df['last_review'],yearfirst=True)
    df['host_since']=pd.to_datetime(df['last_review'],yearfirst=True)

    ####### REMOVE IRRELEVANT COLUMNS #######
    not_relevant = ['listing_url', 'scrape_id', 'last_scraped', 'picture_url', 'host_about', 'host_picture_url', 'calendar_last_scraped', \
                    'license','host_url','host_thumbnail_url','host_verifications', 'name', 'description', 'neighborhood_overview', 'host_name']
    all_0 = ['calendar_updated', 'bathrooms']
    duplicated = ['host_neighbourhood', 'neighbourhood','host_listings_count']
    not_useful = ['beds','host_location','host_response_time','host_response_rate','host_acceptance_rate','calculated_host_listings_count',\
                'calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms',\
                'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', \
                'neighbourhood_cleansed'] # ADD ALL YOUR NOT USEFUL FEATURES HERE

    df = df.drop(not_relevant+all_0+duplicated+not_useful, axis = 1)
    
    if verbose:
        print('Columns dropped {}'.format(not_relevant+all_0+duplicated+not_useful))

    ####### REMOVE INACTIVE LISTINGS #######

    if verbose:
        print('Before removing inactive listings, length = {}'.format(len(df)))

    #create temp column to store active status
    df['active'] = np.nan 

    #Find those who joined before 5/12/20 (i.e. 1 year before the data scraped date)
    filter_date = pd.to_datetime('2020-12-05')
    df_old = df[df['host_since']<filter_date]
    df_new = df[df['host_since']>=filter_date]

    #For old hosts: check if their reviews per month >0.15(~ 25th percentile);else deem them inactive
    df_old['active']=[1 if i>=0.15 else 0 for i in df_old['reviews_per_month']]
    df_new.loc[:,'active'] = 1
    #Get the active ones from df_old
    df_old = df_old[df_old['active']==1]
    #combine & get back original df
    df_combined = pd.concat([df_old,df_new],axis=0)
    df=df_combined.drop('active',axis=1)

    if verbose:
        print('After removing inactive listings, length = {}'.format(len(df)))
    
    # Drop rows where price is 0
    df = df[df['price'] != 0]
    if verbose:
        print('After removing price = 0 listings, length = {}'.format(len(df)))
    
    ####### EXTRACT BATHROOM FEATURES #######
    df['bathrooms_text'] = df['bathrooms_text'].str.lower()
    df['bathroom_num'] = df['bathrooms_text'].str.extract(pat = r'(^[0-9]*[\.0-9]*)') # Capture the numbers at the beginning, including decimals
    df['bathroom_num'] = np.where(df['bathrooms_text'].str.contains(r'half-bath', regex = True), 0.5, df['bathroom_num']) # extract half-bath as 0.5
    df['bathroom_num'] = df['bathroom_num'].astype('float')
    df['bathroom_sharing'] = df['bathrooms_text'].str.extract(pat = r'(shared|private)')
    if verbose:
        print('Features bathroom_num and bathroom_sharing extracted.')
    
    ####### FILL MISSING VALUES #######
    # Chosen approach = fill with 0
    for col in df.columns:
        # select only integer or float dtypes
        if df[col].dtype in ("int", "float"):
            df[col] = df[col].fillna(0)
            
        # select only object dtypes
        if df[col].dtype == object:
            df[col] = df[col].fillna('0')
        
        # select date dtypes
        if df[col].dtype == '<M8[ns]':
            df[col] = df[col].fillna(pd.to_datetime('2200-01-01'))
    
    if verbose:
        print('Int and float mv filled with 0, object mv filled with "0" and date mv filled with 2200-01-01.')

    ####### TRANSFORMATION #######
    if verbose:
        print('Before Log Transform to price, min = {}, max = {}'.format(min(df['price']), max(df['price'])))
    
    df['price_log'] = df['price'].apply(np.log)

    if verbose:
        print('After Log Transform to price, min = {}, max = {}'.format(min(df['price']), max(df['price'])))
    
    ####### WINSORIZE OUTLIERS #######
    cols_and_limit_dict = {'bedrooms': df['bedrooms'].quantile(0.995), 
                        'minimum_nights':  df['minimum_nights'].quantile(0.995), 
                        'maximum_nights': df['maximum_nights'].quantile(0.99), 
                        'minimum_nights_avg_ntm':  df['minimum_nights_avg_ntm'].quantile(0.9975)}

    for column, value_cutoff in cols_and_limit_dict.items():
        if verbose:
            print('{}: {} values above cutoff, value_cutoff = {}'.format(column, 
                                                                        len(df[df[column]>value_cutoff]), 
                                                                        value_cutoff))
        df[column] = df[column].clip(upper = value_cutoff) # cap values above the cutoff

    ####### SIMPLE ENCODING #######
    # Encode T/F columns under host as 1 and 0
    df = df[~(df['host_identity_verified'] == '0')] # remove 5 records where there is no information about all 3 of this
    encode = {"t" : 1, "f" : 0}
    df = df.replace({"host_is_superhost": encode,
                    'host_has_profile_pic': encode,
                    'host_identity_verified': encode,
                    'has_availability': encode,
                    'instant_bookable': encode})
    
    ####### DROP COLUMNS USED FOR PREPROCESSING #######
    preprocessing_cols = ['bathrooms_text']
    if verbose:
        print('Preprocessing columns dropped {}'.format(preprocessing_cols))

    df = df.drop(preprocessing_cols, axis = 1)

    return df


def get_amenities(data, num_amenities = 15, verbose = True):
    
    df = data.copy()
    df['amenities'] = df['amenities'].str.lower()
    amenities_list = df['amenities'].apply(lambda x: ast.literal_eval(x)) # create a new list by using ast to parse

    amenities_combined = []
    for sublist in amenities_list.tolist():
        for item in sublist:
            amenities_combined.append(item)

    counter=collections.Counter(amenities_combined)
    most_common=counter.most_common(num_amenities)
    if verbose:
        print(most_common)

    # Convert dictionary to dataframe
    counter_df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    counter_df.columns = ['amenity', 'num_listings']
    counter_df = counter_df.sort_values('num_listings', ascending=False)

    list_of_amenities = counter_df['amenity'][0:num_amenities]
    for value in list_of_amenities:
        df['amenities_{}'.format(value)] = df['amenities'].str.contains(value, regex = True)
        df['amenities_{}'.format(value)] = df['amenities_{}'.format(value)].astype(int) # cast to integer
    
    ####### DROP COLUMNS USED FOR PREPROCESSING #######
    preprocessing_cols = ['amenities']
    if verbose:
        print('Preprocessing columns dropped {}'.format(preprocessing_cols))

    df = df.drop(preprocessing_cols, axis = 1)

    return df
