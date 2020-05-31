#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install geopy


# In[4]:


import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
from pandas.io.json import json_normalize  # tranform JSON file into a pandas dataframe

import folium # map rendering library

# import k-means from clustering stage
from sklearn.cluster import KMeans

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[5]:


import requests

from bs4 import BeautifulSoup


# In[63]:


a = pd.read_csv(r'C:\Users\Fujitsu\Desktop\Bangalore.csv')
a.head()


# In[64]:


a.shape


# In[65]:


import pandas as pd

b = pd.read_csv(r'C:\Users\Fujitsu\Desktop\Zone.csv')
b.head()


# In[66]:


c = pd.merge(a,b, on= 'Pin')
c.head(10)


# In[67]:


c.dropna(subset=["Place","Pin","Zone","Latitude","Longitude"], how='any',inplace= True)
c.head(10)


# In[69]:


c.describe


# In[68]:


c.shape


# In[8]:


get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim


# In[9]:


get_ipython().system('pip install geocoder')


# In[15]:


CLIENT_ID = 'XUAYS5LROHLIGN0LTDNAPHS015AYOFWTTLX45U05EUFZ4ZJK' # your Foursquare ID
CLIENT_SECRET = 'NEC2M2AKOEHNOS11MFKJEQHNNLCYRFMZSC521A40JGIRBJIY' # your Foursquare Secret
VERSION = '20200531' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[70]:


c.loc[0, 'Place']


# In[72]:


neighborhood_latitude = c.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = c.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = c.loc[0, 'Place'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# In[73]:


LIMIT = 100

radius = 500
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url


# In[74]:


results = requests.get(url).json()
results


# In[75]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[76]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[77]:



print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# In[78]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
        
         # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Place', 
                  'Place Latitude', 
                  'Place Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[80]:



B_venues = getNearbyVenues(names=c['Place'],
                                   latitudes=c['Latitude'],
                                   longitudes=c['Longitude']
                                  )


# In[81]:


print(B_venues.shape)
B_venues.head()


# In[83]:


B_venues.groupby('Place').count()


# In[89]:


# one hot encoding
B_onehot = pd.get_dummies(B_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
B_onehot['Neighborhood'] = B_venues['Place'] 

# move neighborhood column to the first column
fixed_columns = [B_onehot.columns[-1]] + list(B_onehot.columns[:-1])
B_onehot = B_onehot[fixed_columns]

B_onehot.head()


# In[90]:


B_onehot.shape


# In[91]:


B_grouped = B_onehot.groupby('Neighborhood').mean().reset_index()
B_grouped


# In[92]:


B_grouped.shape


# In[94]:


#Sort Data by Most Popular Venues
num_top_venues = 5

for hood in B_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = B_grouped[B_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[95]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[96]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = B_grouped['Neighborhood']

for ind in np.arange(B_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(B_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[97]:


# import k-means from clustering stage
from sklearn.cluster import KMeans


# In[130]:


#Group The Data Into Clusters
kclusters = 8

B_grouped_clustering = B_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(B_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[132]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labelsv3', kmeans.labels_)

B_merged = c

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
B_merged = B_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Place')

B_merged.head() # check the last columns!


# In[133]:


address = "Bangalore, BLR"

geolocator = Nominatim(user_agent="bangalore_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Bangalore city are {}, {}.'.format(latitude, longitude))


# In[134]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)


# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(B_merged['Latitude'], B_merged['Longitude'], B_merged['Place'], B_merged['Cluster Labelsv3']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[149]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(
        B_merged['Latitude'], 
        B_merged['Longitude'], 
        B_merged['Place'], 
        B_merged['Cluster Labelsv3']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow,
        fill=True,
        fill_color=rainbow,
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:





# In[136]:


# Examining Clusters
B_merged.loc[B_merged['Cluster Labelsv3'] == 0, B_merged.columns[[1] + list(range(5, B_merged.shape[1]))]]


# In[137]:


# Examining Clusters
B_merged.loc[B_merged['Cluster Labelsv3'] == 1, B_merged.columns[[1] + list(range(5, B_merged.shape[1]))]]


# In[138]:


# Examining Clusters
B_merged.loc[B_merged['Cluster Labelsv3'] == 2, B_merged.columns[[1] + list(range(5, B_merged.shape[1]))]]


# In[139]:


# Examining Clusters
B_merged.loc[B_merged['Cluster Labelsv3'] == 3, B_merged.columns[[1] + list(range(5, B_merged.shape[1]))]]


# In[140]:


# Examining Clusters
B_merged.loc[B_merged['Cluster Labelsv3'] == 4, B_merged.columns[[1] + list(range(5, B_merged.shape[1]))]]


# In[141]:


# Examining Clusters
B_merged.loc[B_merged['Cluster Labelsv3'] == 5, B_merged.columns[[1] + list(range(5, B_merged.shape[1]))]]


# In[143]:


# Examining Clusters
B_merged.loc[B_merged['Cluster Labelsv3'] == 6, B_merged.columns[[1] + list(range(5, B_merged.shape[1]))]]


# In[144]:


# Examining Clusters
B_merged.loc[B_merged['Cluster Labelsv3'] == 7, B_merged.columns[[1] + list(range(5, B_merged.shape[1]))]]


# In[ ]:





# In[ ]:




