# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:59:21 2022

@author: tonyd
"""
### imports ###
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier

from tqdm import tqdm

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
from string import punctuation
from collections import Counter
from heapq import nlargest
from PIL import Image
import torch
import os

import sentence_transformers
from sentence_transformers import SentenceTransformer, util

from bs4 import BeautifulSoup
import requests
import lxml

@st.cache(allow_output_mutation=True)
def load_file(filename):
    with open(filename, "rb") as file_:
      lf = pkl.load(file_)
    return lf
stats = load_file("stats_df.pkl")
df = load_file("comic_df.pkl")
corpus_embeddings = load_file("comic_corpus_embeddings.pkl")
corpus = load_file("comic_corpus.pkl")

marvel_banner = Image.open('mb.jpg')
st.image(marvel_banner, use_column_width=True)


# with open("stats_df.pkl" , "rb") as file_1:
#     stats = pkl.load(file_1)
#
# with open("comic_df.pkl" , "rb") as file_2:
#     df = pkl.load(file_2)
#
# with open("comic_corpus_embeddings.pkl" , "rb") as file_3:
#     corpus_embeddings = pkl.load(file_3)
#
# with open("comic_corpus.pkl" , "rb") as file_4:
#     corpus = pkl.load(file_4)

# embedder = SentenceTransformer('all-MiniLM-L6-v2')
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedder = load_model()

### page title ###
st.title('Marvel Hero Matcher')

st.markdown("""
This app will let you define a hero by picking hero stats, then match you with
the closest Marvel hero we can find (or villain, if you so choose). Next, it will search Marvel comics for
the best comic titles to read about your hero. Then, change that search as you
want!
""")
st.markdown("""---""")

### sidebar menu - year selection ###
# st.sidebar.header('User Input Features')
# s_int = st.sidebar.slider('Intelligence', min_value=1, max_value=25)
# s_str = st.sidebar.slider('Strength', min_value=1, max_value=25)
# s_spd = st.sidebar.slider('Speed', min_value=1, max_value=25)
# s_dur = st.sidebar.slider('Durability', min_value=1, max_value=25)
# s_pow = st.sidebar.slider('Power', min_value=1, max_value=25)
# s_com = st.sidebar.slider('Combat', min_value=1, max_value=25)

scale = 25
scl = (120/scale)

st.sidebar.header('Pick some stats to get matched to a Marvel Hero!')

h_stats = stats[stats.Alignment=='good']
v_stats = stats[stats.Alignment=='bad']

matched_hero =''
matched_villain = ''

def h_match_maker(hero_stats):
    stat_cols = h_stats.columns[2:8]
    x = h_stats[stat_cols].values
    y = h_stats[['Name']].values
    hero = np.array([hero_stats])
    hero_knn = KNeighborsClassifier(n_neighbors=3)
    hero_knn.fit(x, y)
    ydf = pd.DataFrame(y).rename(columns={0:"name"})
    mvs = hero_knn.kneighbors(hero, n_neighbors=1, return_distance=False)
    mvs = mvs.tolist()[0]
    match_list = ydf['name'][ydf.index.isin(mvs)].values.tolist()
    match_df = h_stats[h_stats["Name"].isin(match_list)]
    return match_df

def v_match_maker(hero_stats):
    stat_cols = v_stats.columns[2:8]
    x = v_stats[stat_cols].values
    y = v_stats[['Name']].values
    hero = np.array([hero_stats])
    hero_knn = KNeighborsClassifier(n_neighbors=3)
    hero_knn.fit(x, y)
    ydf = pd.DataFrame(y).rename(columns={0:"name"})
    mvs = hero_knn.kneighbors(hero, n_neighbors=1, return_distance=False)
    mvs = mvs.tolist()[0]
    match_list = ydf['name'][ydf.index.isin(mvs)].values.tolist()
    match_df = v_stats[v_stats["Name"].isin(match_list)]
    return match_df

def stat_display(stuff, align):
    if align == 'h':
        bar_color = 'red'
    if align == 'v':
        bar_color = 'purple'

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(15,15))

    stat_names = ['Intelligence', 'Strength', 'Speed', 'Durability', 'Power', 'Combat']
    x_pos = np.arange(0,26,5)
    y_pos = np.arange(len(stat_names))
    bar_stats = stuff
    stats_bar_text_color = 'white'

    ax.barh(y_pos, bar_stats, align='center', color=bar_color, height=0.5, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([0,5,10,15,20,25], fontsize=25)
    ax.set_yticklabels(stat_names, fontsize=30)
    ax.set_xlim([0,25])
    ax.invert_yaxis()  # labels read top-to-bottom
    #ax.set_xlabel('\n\nWeak  -------------------------------- Super ------------------------------ Godly\n(but still better than you!)                                                                  ')
    #ax.set_title('Your Hero Stats', color=stats_bar_text_color)

    ax.set_facecolor('none')
    fig.set_alpha(0.5)
    fig.set_facecolor('none')

    ax.xaxis.label.set_color(stats_bar_text_color)        #setting up X-axis label color to yellow
    ax.yaxis.label.set_color(stats_bar_text_color)          #setting up Y-axis label color to blue

    ax.tick_params(axis='x', colors=stats_bar_text_color)    #setting up X-axis tick color to red
    ax.tick_params(axis='y', colors=stats_bar_text_color)  #setting up Y-axis tick color to black

    ax.spines['left'].set_color('none')        # setting up Y-axis tick color to red
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.grid()

    return st.pyplot(fig)
   # Every form must have a submit button.

def char_img(q):
    #word = 'agent bob marvel fandom comic portrait'
    word = q
    url = 'https://www.google.com/search?q={0}&tbm=isch'.format(word)
    content = requests.get(url).content
    soup = BeautifulSoup(content,'lxml')
    images = soup.findAll('img', limit=2)
    image_list=[]
    for image in images:
        image_list.append(image.get('src'))
    # image_url = image_list[1]
    # st.write(image_url)
    display_image = Image.open(requests.get(image_list[1], stream=True).raw)
    return st.image(display_image, width=150)

s_int = scl*st.sidebar.slider('Intelligence', min_value=1, max_value=scale)
s_str = scl*st.sidebar.slider('Strength', min_value=1, max_value=scale)
s_spd = scl*st.sidebar.slider('Speed', min_value=1, max_value=scale)
s_dur = scl*st.sidebar.slider('Durability', min_value=1, max_value=scale)
s_pow = scl*st.sidebar.slider('Power', min_value=1, max_value=scale)
s_com = scl*st.sidebar.slider('Combat', min_value=1, max_value=scale)
hero_stats = [s_int, s_str, s_spd, s_dur, s_pow, s_com]
stats_total = sum(hero_stats)/scl
# st.write("Your current stat total is: ", int(stats_total),"out of 150")
st.markdown("""Select your stats on the left!""")
st.markdown("""Your current stat total is:""")
st.markdown(int(stats_total) , unsafe_allow_html=True)
st.markdown(""" out of a possible 150""")

if stats_total < 50:
    st.markdown("You may want to kick it up a notch, yeah?")
if stats_total >= 50 and stats_total < 75:
    st.markdown("This is fine, but still a bit weak?")
if stats_total >= 75 and stats_total < 125:
    st.markdown("That's more like it! ")
if stats_total >= 125:
    st.markdown("Fine, but we're going to have you fight Thanos when he has all that Infinity Stuff")

st.markdown("""---""")



with st.form("Hero Form"):
   submitted = st.form_submit_button("Find me a Hero!")
   if submitted:
       hero_stats = [s_int, s_str, s_spd, s_dur, s_pow, s_com]
       ###### match hero_stats to character from dataframe
       match_df=h_match_maker(hero_stats)
       match_stats = match_df.iloc[0,2:8].values.tolist()
       matched_hero = match_df.iloc[0,0]
       match_stats = list(np.array(match_stats)/scl)
       st.write(matched_hero)
       #st.write(match_df)
       char_query = str(matched_hero) + ' marvel fandom comic'
       ####### Display stuff about the matched hero
       col1, col2, = st.columns([1,2])
       with col1:
           char_img(char_query)
       with col2:
           stat_display(match_stats, 'h')

with st.form("Villain Form"):
   submitted = st.form_submit_button("Find me a Villain!")
   if submitted:
       hero_stats = [s_int, s_str, s_spd, s_dur, s_pow, s_com]
       ###### match hero_stats to character from dataframe
       match_df=v_match_maker(hero_stats)
       match_stats = match_df.iloc[0,2:8].values.tolist()
       matched_villain = match_df.iloc[0,0]
       match_stats = list(np.array(match_stats)/scl)
       st.write(matched_villain)
       #st.write(match_df)
       ####### Display stuff about the matched hero
       char_query = str(matched_villain) + ' marvel fandom comic'
       ####### Display stuff about the matched villain
       col1, col2, = st.columns([1,2])
       with col1:
           char_img(char_query)
       with col2:
           stat_display(match_stats, 'v')








st.markdown("""---""")

# match_stats = match_df.iloc[0,2:8].values.tolist()
# #if st.button('Click Here to Keep This Party Going!'):
# match_stats = list(np.array(match_stats)/scl)

### function for horizontal bar chart of stats
initial_search=''
comic_title_list=[]

if matched_hero != '':
    initial_search = str(matched_hero) + ' does some sweet hero stuff!'

if matched_villain != '':
    initial_search = str(matched_villain) + ' is turning the evil up to 11!'

st.subheader("""**Comic Search:**""")

query = ''
query = st.text_input("", value=initial_search)
# queries = list([queries])

# Find the closest 5 sentences of the corpus for query sentence based on cosine similarity
if query == '':
    st.markdown("""What kind of action are you looking for?
    Type something like Spiderman fights The Green Goblin and loses! or
    Wolverine teams up with Black Widow!""")
else:
    top_k = min(5, len(corpus))
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    st.markdown("""---""")
    #st.write("You searched for:   ", query, "\n")
    st.subheader("""**You can read about it in these exciting titles:**""")

    for score, idx in zip(top_results[0], top_results[1]):
        # st.write("(Score: {:.4f})".format(score))
        # st.write(corpus[idx], "(Score: {:.4f})".format(score))
        comic=df['comic_name'][df['all_review']==corpus[idx]]
        row_dict = df.loc[df['all_review']== corpus[idx]]
        # row2_dict = sum_df.loc[sum_df['all_review']== corpus[idx]]
        # row3_dict = df1.loc[df1['comic_name']==row_dict['comic_name'].values[0]]
        st.write(row_dict['comic_name'].values[0])
        comic_title_list.append(row_dict['comic_name'].values[0])
        #st.write("Hotel Review Summary: " , row2_dict['summary'].values[0])
        #st.write("Tripadvisor Link: [here](%s)" %row3_dict['url'].values[0], "\n")

st.markdown("""---""")







#### previous hero display code

#
# if st.button('Find Me a Hero!'):
#   hero_stats = [s_int, s_str, s_spd, s_dur, s_pow, s_com]
#   ###### match hero_stats to character from dataframe
#   match_df=h_match_maker(hero_stats)
#   match_stats = match_df.iloc[0,2:8].values.tolist()
#   matched_hero = match_df.iloc[0,0]
#   match_stats = list(np.array(match_stats)/scl)
#   st.write(matched_hero)
#   #st.write(match_df)
#   char_query = str(matched_hero) + ' marvel fandom comic'
#   ####### Display stuff about the matched hero
#   col1, col2, = st.columns([1,2])
#   with col1:
#       char_img(char_query)
#   with col2:
#       stat_display(match_stats, 'h')
#
# if st.button('Find Me a Villain'):
#   hero_stats = [s_int, s_str, s_spd, s_dur, s_pow, s_com]
#   ###### match hero_stats to character from dataframe
#   match_df=v_match_maker(hero_stats)
#   match_stats = match_df.iloc[0,2:8].values.tolist()
#   matched_villain = match_df.iloc[0,0]
#   match_stats = list(np.array(match_stats)/scl)
#   st.write(matched_villain)
#   #st.write(match_df)
#   ####### Display stuff about the matched hero
#   char_query = str(matched_villain) + ' marvel fandom comic'
#   ####### Display stuff about the matched villain
#   col1, col2, = st.columns([1,2])
#   with col1:
#       char_img(char_query)
#   with col2:
#       stat_display(match_stats, 'v')
