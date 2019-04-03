"""These functions create the visualizations for the notebook and slides."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wordcloud import WordCloud

matplotlib.rcParams['figure.dpi'] = 300


def plot_mean_stars_by_category():
    # Load the training stars and categories
    train = pd.read_csv(
        'data/train.csv',
        usecols=lambda x: x == 'stars' or x.startswith('category'))

    # Category names
    categories = [c for c in train.columns if c.startswith('category')]

    # Reverse one hot encoding
    stars = train[['stars']].copy()
    stars['category'] = np.nan
    for c in categories:
        stars.loc[train[c] == 1, 'category'] = c

    # Merge some categories
    map = {
        'category.Restaurants': 'Restaurants',
        'category.Food': 'Other',
        'category.Nightlife': 'Bars & Nightlife',
        'category.Bars': 'Bars & Nightlife',
        'category.American(Traditional)': 'Restaurants',
        'category.American(New)': 'Restaurants',
        'category.Breakfast&Brunch': 'Restaurants',
        'category.EventPlanning&Services': 'Event Planning & Services',
        'category.Shopping': 'Shopping',
        'category.Sandwiches': 'Restaurants',
        'category.Beauty&Spas': 'Beauty & Spas',
        'category.Arts&Entertainment': 'Other',
        'category.Mexican': 'Restaurants',
        'category.Burgers': 'Restaurants',
        'category.Pizza': 'Restaurants',
        'category.Italian': 'Restaurants',
        'category.Hotels&Travel': 'Hotels & Travel',
        'category.Seafood': 'Restaurants',
        'category.Coffee&Tea': 'Restaurants',
        'category.Japanese': 'Restaurants',
        'category.HomeServices': 'Home Services',
        'category.Desserts': 'Restaurants',
        'category.Automotive': 'Automotive',
        'category.Chinese': 'Restaurants',
        'category.SushiBars': 'Restaurants',
        'category.Other': 'Other'
    }
    stars['category'] = stars['category'].apply(lambda x: map[x])

    # Make plot
    sns.barplot(x="stars", y="category", data=stars)


def plot_staff_training_cloud():
    positive = [
        'nice',
        'friendly',
        'helpful',
        'professional',
        'courteous',
        'great service',
        'friendly service',
        'excellent service',
        'outstanding service',
        'excellent customer service',
        'awesome staff',
        'helpful staff'
    ]
    negative = [
        'rude',
        'unprofessional',
        'bad service',
        'poor service',
        'poor customer service',
        'bad customer service',
        'bad attitude',
        'rude staff',
        'super rude'
    ]
    neutral = [
        'management',
        'staff',
        'customer service',
        'attitude',
    ]
    text = {x: 1 for x in positive + negative + neutral}

    def get_color(*args, **kwargs):
        if args[0] in positive:
            return 'green'
        if args[0] in negative:
            return 'red'
        return 'black'

    cloud = WordCloud(background_color='white',
                      color_func=get_color,
                      min_font_size=36,
                      max_font_size=72,
                      width=1200,
                      height=800)
    cloud.generate_from_frequencies(text)

    plt.imshow(cloud)
    plt.axis("off")
    plt.show()


def plot_cleanliness_cloud():
    positive = [
        'clean',
        'fresh',
    ]
    negative = [
        'dirty',
        'stain',
        'filthy',
        'disgust',
        'gross',
        'bug',
        'nasty',
        'dirty room',
        'previous guest',
        'bed bug',
        'bed bug bites',
        'black mold'
    ]
    neutral = []
    text = {x: 1 for x in positive + negative + neutral}

    def get_color(*args, **kwargs):
        if args[0] in positive:
            return 'green'
        if args[0] in negative:
            return 'red'
        return 'black'

    cloud = WordCloud(background_color='white',
                      color_func=get_color,
                      min_font_size=48,
                      max_font_size=96,
                      width=1200,
                      height=800)
    cloud.generate_from_frequencies(text)

    plt.imshow(cloud)
    plt.axis("off")
    plt.show()


def plot_food_cloud():
    positive = [
        'delicious',
        'fresh',
        'good for groups',
        'reservations',
        'full bar',
        'delivery',
        'take out',
        'great food',
        'amazing food',
        'great breakfast',
        'great drinks',
        'food delicious',
        'great meal',
        'great restaurant',
        'nice restaurant',
        'wonderful food'
    ]
    negative = [
        'bad food',
        'disgusting',
        'bad meal'
    ]
    neutral = [
        'restaurant',
        'food',
        'breakfast'
    ]
    text = {x: 1 for x in positive + negative + neutral}

    def get_color(*args, **kwargs):
        if args[0] in positive:
            return 'green'
        if args[0] in negative:
            return 'red'
        return 'black'

    cloud = WordCloud(background_color='white',
                      color_func=get_color,
                      min_font_size=48,
                      max_font_size=96,
                      width=1200,
                      height=800)
    cloud.generate_from_frequencies(text)

    plt.imshow(cloud)
    plt.axis("off")
    plt.show()

#make bar plot to the customer sevice
p = pd.read_csv('./data/inportance_phrases_hotel.csv',index_col=0)
neg_service = p.loc[['horrible experience','bad service','poor service',
                     'poor customer service','bad customer service','horrible customer service',
                     'dont care','didnt care','bad attitude','rude unhelpful','rude staff',
                     'super rude','rude people','long line','front line','huge line','eye contact',
                     'poor management','wrong information']]
pos_service = p.loc[['great service','friendly service','excellent service','outstanding service',
                     'wonderful service','beautiful service','great room service','incredible service',
                     'amazing service','excellent customer service','friendly helpful','nice helpful',
                     'super helpful','super friendly helpful','professional helpful','friendly informative',
                     'courteous helpful','great staff','amazing staff','wonderful staff','super friendly staff',
                     'fantastic staff','friendly staff','friendly helpful staff','perfect staff','awesome staff',
                     'great people','beautiful people','great guy','friendly people','amazing people','wonderful people']]
# set width of bar
barWidth = 0.4

# set height of bar
bars1 = neg_service.sum(axis=0)[0:5]
bars2 = pos_service.sum(axis=0)[0:5]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Make the plot
plt.bar(r2, bars2, color='seagreen', width=barWidth, edgecolor='white', label='positive words/phrases')
plt.bar(r1, bars1, color='crimson', width=barWidth, edgecolor='white', label='negative words/phrases')

# Add xticks on the middle of the group bars
plt.xlabel('group')
plt.xticks([r + barWidth for r in range(len(bars1))], ['1', '2', '3', '4', '5'])
plt.ylabel('frenquency rate')
plt.title('customer service', fontweight='bold',fontsize=20)
 
# Create legend & Show graphic
plt.legend()
plt.show()
