#data retrieved from https://insideairbnb.com/get-the-data/
## imports
import pandas as pd
import numpy as np
import geopandas as gp
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib as mpl
from matplotlib import dates
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly
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
details = listings.copy()
# Making a column with both longitude and latitude
details['lonlat'] = details[['longitude', 'latitude']].values.tolist()
# Function to make a shapely Point from longitude and latitude
def find_point(x):
    point = Point(x[0],x[1])
    return point

# Making a column of all the points
details['point'] = details['lonlat'].apply(find_point)
# Making a geopandas df from geojson file and isolating the Central region
# Geojson from chingchai https://github.com/chingchai/OpenGISData-Thailand/blob/master/districts.geojson
gdf = gp.read_file("districts.geojson")
bkk = gdf[gdf['reg_nesdb'] == 'Central']
# Making a function to check which of the 130 districts the shapely Point lies in
def find_district(x):
    for y in range(0,len(bkk)):
        district = bkk.iloc[y]['geometry']
        if x.within(district):
            return bkk.iloc[y]['amp_en']


details['location'] = details['point'].apply(find_district)
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
## Finding the ten highest selling listings
ten_highest_sales = annual_details.sort_values(by='sales',ascending=False)[:10].set_index('listing_url', drop=True)
ten_highest_sales['sales'] = ten_highest_sales['sales']/1000000
ten_highest_sales.rename({'sales':'Sales ฿MM'}, axis=1, inplace=True)
print(ten_highest_sales)
# Saving the csv
ten_highest_sales.to_csv('ten_highest_sales.csv')
# Here we're going to make a txt file to save all of the html we want to put in the website
f = open('airbnb_html.txt','w')
# Converting the table to html for website
f.write('ten_highest_sales:\n')
f.write(ten_highest_sales[['Sales ฿MM']].to_html().replace('\n',''))
f.write('\n\n')
## Finding the ten most booked listings
ten_highest_booked = annual_details.sort_values(by=['bookings','sales'],ascending=False)[:10].set_index('listing_url', drop=True)
print(ten_highest_booked)
# Saving the csv
ten_highest_booked.to_csv('ten_highest_booked.csv')
# Converting the table to html for website #
f.write('ten_highest_booked:\n')
f.write(ten_highest_booked[['bookings']].to_html().replace('\n',''))
f.write('\n\n')
## Finding the districts with the highest sales average and the districts with the bookings average
districts = annual_details[['location','sales','bookings']]
districts = districts.groupby('location').sum()
districts['sales'] = districts['sales']/1000000
districts.rename({'sales':'Sales ฿MM'}, axis=1, inplace=True)
# Saving the csv
districts.to_csv('districts.csv')
# Highest Average Sales Total
print(districts.sort_values('Sales ฿MM',ascending=False)[:10])
# Converting the table to html for website
f.write('districts sales:\n')
f.write(districts.sort_values('sales ฿MM',ascending=False)[:10].to_html().replace('\n',''))
f.write('\n\n')
# Highest Average Bookings Total
print(districts[['bookings','Sales ฿MM']].sort_values('bookings',ascending=False)[:10])
# Converting the table to html for website
f.write('districts bookings:\n')
f.write(districts[['bookings','sales ฿MM']].sort_values('bookings',ascending=False)[:10].to_html().replace('\n',''))
f.write('\n\n')

## Plotting Districts with Plotly
bkk['location'] = bkk['amp_en']
choro = bkk.merge(districts, on='location')
choro = choro[['geometry', 'location', 'sales ฿MM', 'bookings']]
choro = choro.set_index('location')
# Choropleth mapbox for sales
fig = px.choropleth_mapbox(choro,
                           geojson=choro['geometry'],
                           locations=choro.index,
                           color='sales ฿MM',
                           color_continuous_scale="Viridis",
                           center={"lat": 13.68597, "lon": 100.6038},
                           mapbox_style="open-street-map",
                           zoom=9.3,
                           opacity=.71,
                           title='Total Sales in ฿MM by District')

fig.update_geos(fitbounds="locations", visible=False)
# saving the plotly figure as an html div
f.write('choropleth sales:\n')
f.write(plotly.offline.plot(fig, include_plotlyjs=False, output_type='div'))
f.write('\n\n')
# don't forget to add the following into your html later:
#<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
#fig.show()

# chloropleth mapbox for bookings
fig = px.choropleth_mapbox(choro,
                           geojson=choro['geometry'],
                           locations=choro.index,
                           color='bookings',
                           color_continuous_scale="Viridis",
                           center={"lat": 13.68597, "lon": 100.6038},
                           mapbox_style="open-street-map",
                           zoom=9.3,
                           opacity=.71,
                           title='Total Bookings by District')

fig.update_geos(fitbounds="locations", visible=True)
# saving the plotly figure as an html div
f.write('choropleth bookings:\n')
f.write(plotly.offline.plot(fig, include_plotlyjs=False, output_type='div'))
f.close()
# don't forget to add the following into your html later:
#<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
#fig.show()

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




