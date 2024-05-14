#https://www.kaggle.com/datasets/sinderpreet/analyze-the-marketing-spending

import numpy as np
import pandas as pd


## Function to add key performance indicators to dataframes
def add_kpis(df):
    # Return on Marketing Investment
    df['ROMI'] = ((df['revenue'] - df['mark_spent'])/df['mark_spent'])*100
    # Click Through Rate (people that clicked on the ad after seeing it)
    df['ClckThru'] = (df['clicks'] / df['impressions']) * 100
    # Leads Conversion (from visitors to leads - gave personal info)
    df['LeadConv'] = (df['leads'] / df['clicks']) * 100
    # Sales Convsersion (from leads to sales/orders)
    df['SalesConv'] = (df['orders'] / df['leads']) * 100
    # Average Order Value - average revenue from orders/sales
    df['AOV'] = df['revenue'] / df['orders']
    # Cost per click
    df['CPC'] = df['mark_spent'] / df['clicks']
    # Cost per lead
    df['CPL'] = df['mark_spent'] / df['leads']
    # Customer Acquisition Cost
    df['CAC'] = df['mark_spent'] / df['orders']
    # Gross Profit
    df['GP'] = df['revenue'] - df['mark_spent']



### Groupby Totals for Overall Performance

## Groupby 'category', 'cats' is short for categories
df = pd.read_csv('Marketing.csv')
cats_total = df.drop(['id', 'c_date', 'campaign_name', 'campaign_id'],axis=1).groupby('category').sum()
# Add metrics for 'category'
add_kpis(cats_total)
# Turn cats_total into a .csv file to be uploaded into Tableau
cats_total.to_csv('cats_total.csv')

print(cats_total['ROMI'])

# ROMI results: influencer = 154.29, media = 22.41, search = 7.07, social = -13.68
# As a category, 'influencer' seems the most lucrative and 'social' the least.

## Groupby 'campaign_name', 'camps' is short for campaign
camps_total = df.drop(['id', 'c_date', 'campaign_id'],axis=1).groupby(['campaign_name','category']).sum()
# Add metrics for 'campaigns'
add_kpis(camps_total)
camps_total.sort_values(by=['category','ROMI'],ascending=[True,False], inplace=True)
# Save as a csv file
camps_total.to_csv('camps_total.csv')

print(camps_total['ROMI'])

# Analysis:
#   Social looks bad because they had 6 campaigns across 2 websites - most of which yielded losses except two which seemed quite lucrative. They actually were successful in getting the most leads.
#     Additionally, it seems 'tier1' is much more effective than 'tier2' and 'retargeting' significantly more effective than 'lal'.

#   Influencer only had 2 campaigns, which both yielded profits, one yielding 7.5x more value than the other, proving you just have to find the right influencer.

#   Search consisted of two seo campaigns on one website; they broke even but 'hot' seo is much more lucrative than 'wide' seo

#   Media has only one campaign, which seemed profitable, but perhaps more campaigns need to be added to assess the effectiveness of the Media category.

# Recommendations: Moving forward it would seem most productive for...
#   Social: Cut funding to tier2 and lal campaigns, and continue with tier1 and retargeting campaigns.
#   Influencer: Add more campaigns, continue funding to the instagram_blogger and increase funds to the youtube_blogger.
#   Search: Cut funding to wide seo campaign and increase funding for hot campaign.
#   Media: Continue funding for the banner campaign and add more campaigns.


# With that in mind let's organize data by dates and see how the metrics for each campaign changed throughout the month.


## Organized by 'c_date' (dates)

# Groupby dates for each category
cats = df.drop(['id','campaign_name','campaign_id'],axis=1)
cats = cats.groupby(['c_date','category']).sum()
# Add metrics
add_kpis(cats)
# Save csv file
cats.to_csv('cats_daily.csv')

# Dates for each campaign - no groupby necessary
camps = df.drop(['id','campaign_id'],axis=1)
# Add metrics
add_kpis(camps)
# Save csv file
camps.to_csv('camps_daily.csv')



