#data retrieved from https://insideairbnb.com/get-the-data/
## imports
import pandas as pd
import numpy as np
import reverse_geocoder as geo
from datetime import date
from dateutil.relativedelta import relativedelta
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

## useful tip

pd.set_option('display.max_rows', 100)



## Part 1: Cleaning Dataframes

# Loading datasets
calendar = pd.read_csv('calendar.csv')
listings = pd.read_csv('listings.csv')

#Columns for calendar:
#['listing_id', 'date', 'available', 'price', 'adjusted_price',
#       'minimum_nights', 'maximum_nights']
#
#Columns for listings:
#['id', 'listing_url', 'scrape_id', 'last_scraped', 'source', 'name',
#       'description', 'neighborhood_overview', 'picture_url', 'host_id',
#       'host_url', 'host_name', 'host_since', 'host_location', 'host_about',
#       'host_response_time', 'host_response_rate', 'host_acceptance_rate',
#       'host_is_superhost', 'host_thumbnail_url', 'host_picture_url',
#       'host_neighbourhood', 'host_listings_count',
#       'host_total_listings_count', 'host_verifications',
#       'host_has_profile_pic', 'host_identity_verified', 'neighbourhood',
#       'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude',
#       'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
#       'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price',
#       'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
#       'maximum_minimum_nights', 'minimum_maximum_nights',
#       'maximum_maximum_nights', 'minimum_nights_avg_ntm',
#       'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability',
#       'availability_30', 'availability_60', 'availability_90',
#       'availability_365', 'calendar_last_scraped', 'number_of_reviews',
#       'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review',
#       'last_review', 'review_scores_rating', 'review_scores_accuracy',
#       'review_scores_cleanliness', 'review_scores_checkin',
#       'review_scores_communication', 'review_scores_location',
#       'review_scores_value', 'license', 'instant_bookable',
#       'calculated_host_listings_count',
#       'calculated_host_listings_count_entire_homes',
#       'calculated_host_listings_count_private_rooms',
#       'calculated_host_listings_count_shared_rooms', 'reviews_per_month']

# Making sure the 'listing_id' feature names and types match in both data frames
calendar['listing_id'] = calendar['listing_id'].astype(str)
listings.rename({'id':'listing_id'},axis=1,inplace=True)
listings['listing_id'] = listings['listing_id'].astype(str)
# Changing 'adjusted_price' values to floats
calendar['adjusted_price'] = calendar['adjusted_price'].str.replace('$', '')
calendar['adjusted_price'] = calendar['adjusted_price'].str.replace(',', '')
calendar['adjusted_price'] = calendar['adjusted_price'].astype(float)
calendar = calendar.drop('price',axis=1)
## Here we are looking to find the total days booked and total sales in $ for each listing_id
# Isolating the booked listings from the listings not booked
all_bookings = calendar[calendar['available']=='f']
all_bookings = all_bookings.drop('available',axis=1)
# Deleting some columns for speed
shorter = all_bookings[['listing_id','adjusted_price']]
# Counting the days booked for each listing using groupby with listing_id
bookings = shorter.groupby(['listing_id']).count()
# Renaming the counted 'adjusted_price' column to 'bookings'
bookings.rename({'adjusted_price':'bookings'}, axis=1, inplace=True)
# Summing the price of every day booked for each listing using groupby with listing_id
sales = shorter.groupby(['listing_id']).sum()
# Renaming the summed 'adjusted_price' column to 'sales' #
sales.rename({'adjusted_price':'sales'}, axis=1, inplace=True)
## Using the latitude and longitude to find an accurate location and fill in nans
# Combining the latitude and longitude into strs to be used by reverse_geocoder
details = listings
details['latitude'] = details['latitude'].astype(str)
details['longitude'] = details['longitude'].astype(str)
# Using reverse_geocoder to find the district names from the coordinates
coords = [(details['latitude'][i],details['longitude'][i]) for i in range(0,len(details['latitude']))]
locations = geo.search(coords)
regions = [x['name'] for x in locations]
details['location'] = regions
# Merging the sales df with the details df on listing_id to add annual sales total as a feature of the listing
annual_details = pd.merge(details,sales, how='outer', on='listing_id')
# Merging the annual_details df with the bookings df to add total annual bookings as a feature of the listing
annual_details = pd.merge(annual_details,bookings, how='outer', on='listing_id')
annual_details.dropna(subset='bookings', inplace=True)
# Cleaning some characters off numerical features to be used later
annual_details['price'] = annual_details['price'].str.replace('$', '')
annual_details['price'] = annual_details['price'].str.replace(',', '')
annual_details['host_response_rate'] = annual_details['host_response_rate'].str.replace('%', '')
annual_details['host_acceptance_rate'] = annual_details['host_acceptance_rate'].str.replace('%', '')
# Turning the host_since feature into number of days as a host using apply
annual_details['host_since'] = pd.to_datetime(annual_details['host_since'])
current_date = pd.to_datetime(calendar.date.max())
annual_details['host_since'] = annual_details['host_since'].apply(lambda x: current_date.date() - x.date())
annual_details['host_since'] = annual_details['host_since'].dt.days
# Was plotting the listings with the geojson and appparently some of these listings aren't in Bangkok
not_bkk = ['Ban Khlong Bang Sao Thong', 'Bang Kruai','Lam Luk Ka', 'Phra Pradaeng', 'Salaya']
for x in not_bkk:
    annual_details = annual_details[annual_details['location']!=x]

# Also reversee geocoder named some Phra Nakhon listings as 'Bangkok'
annual_details['location'] = annual_details['location'].replace({'Bangkok':'Phra Nakhon'})
## Finding the ten highest selling listings ##
ten_highest_sales = annual_details.sort_values(by='sales',ascending=False)[:10].set_index('listing_url', drop=True)
ten_highest_sales['sales'] = ten_highest_sales['sales']/1000000
ten_highest_sales.rename({'sales':'Sales ฿mm'}, axis=1, inplace=True)
print(ten_highest_sales)
# Saving the csv
ten_highest_sales.to_csv('ten_highest_sales.csv')
# Converting the table to html for website
#ten_highest_sales[['Sales ฿mm']].to_html().replace('\n','')
## Finding the ten most booked listings ##
ten_highest_booked = annual_details.sort_values(by=['bookings','sales'],ascending=False)[:10].set_index('listing_url', drop=True)
print(ten_highest_booked)
# Saving the csv
ten_highest_booked.to_csv('ten_highest_booked.csv')
# Converting the table to html for website #
#ten_highest_booked[['bookings']].to_html().replace('\n','')
## Finding the districts with the highest sales average and the districts with the bookings average
districts = annual_details[['location','sales','bookings']]
districts = districts.groupby('location').sum()
districts['sales'] = districts['sales']/1000000
districts.rename({'sales':'sales ฿mm'}, axis=1, inplace=True)
# Saving the csv
districts.to_csv('districts.csv')
# Highest Average Sales Total
print(districts.sort_values('sales ฿mm',ascending=False)[:10])
# Converting the table to html for website
#districts.sort_values('sales ฿mm',ascending=False)[:10].to_html().replace('\n','')
# Highest Average Bookings Total
print(districts[['bookings','sales ฿mm']].sort_values('bookings',ascending=False)[:10])
# Converting the table to html for website
#districts[['bookings','sales ฿mm']].sort_values('bookings',ascending=False)[:10].to_html().replace('\n','')


## Finding daily sales and booking trends
days_sales = calendar[['date','adjusted_price']].groupby('date').mean()
days_sales.rename({'adjusted_price':'Sales ฿','date':'Date'},axis=1,inplace=True)
days_sales.index = days_sales.index.astype('datetime64[ns]')
days_bookings = calendar.copy()
days_bookings['available'] = days_bookings['available'].replace({'t':0,'f':1})
days_bookings = days_bookings[['date','available']].groupby('date').mean()
days_bookings['available'] = days_bookings['available']*100
days_bookings.rename({'available':'Booking Rate','date':'Date'},axis=1,inplace=True)
days_bookings.index = days_bookings.index.astype('datetime64[ns]')
# Saving to csv for later
days_sales.to_csv('days_sales.csv')
days_bookings.to_csv('days_bookings.csv')
# Plotting with sns
from matplotlib import dates
sns.set_theme(palette='inferno')
days_sales_fig = plt.figure()
days_sales_plot = sns.lineplot(days_sales)
days_sales_plot.xaxis.set_major_formatter(dates.DateFormatter("%b-%y"))
days_sales_plot.set_title('Average Daily Sales Across Bangkok')
days_sales_plot.set(xlabel=None)
days_sales_fig.savefig('average_sales_daily.png')
#plt.show()

days_bookings_fig = plt.figure()
days_bookings_plot = sns.lineplot(days_bookings)
days_bookings_plot.xaxis.set_major_formatter(dates.DateFormatter("%b-%y"))
days_bookings_plot.set_title('Listings Booked Daily Across Bangkok')
days_bookings_plot.set(xlabel=None)
days_bookings_plot.set(ylabel='% of Listings Booked')
days_bookings_fig.savefig('bookings_daily.png')
#plt.show()

# Looking at strange shape of daily sales
days_sales['day'] = np.array(days_sales.index)
days_sales['day'] = days_sales['day'].dt.day_name()
days_sales.sort_values(by='Sales ฿',ascending=False)[:100]
# So those spikes in daily sales are fridays and saturdays or holidays



