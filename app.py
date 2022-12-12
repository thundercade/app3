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

with open("stats_df.pkl" , "rb") as file_1:
    stats = pkl.load(file_1)

with open("comic_df.pkl" , "rb") as file_2:
    df = pkl.load(file_2)

with open("comic_corpus_embeddings.pkl" , "rb") as file_3:
    corpus_embeddings = pkl.load(file_3)

with open("comic_corpus.pkl" , "rb") as file_4:
    corpus = pkl.load(file_4)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
# @st.cache(allow_output_mutation=True)
# def load_model():
#     return SentenceTransformer('all-MiniLM-L6-v2')
# embedder = load_model()

### page title ###
st.title('Welcome to Deadpools Hero Matcher!')

st.markdown("""
This app will let you define a hero by picking stats and powers, then match you with
the closest Marvel hero we can find.
""")

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

with st.form("my_form"):

   st.title("Make Me a Super-Hero!!")
   st.header("Select your stats below")

   s_int = scl*st.slider('Intelligence', min_value=1, max_value=scale)
   s_str = scl*st.slider('Strength', min_value=1, max_value=scale)
   s_spd = scl*st.slider('Speed', min_value=1, max_value=scale)
   s_dur = scl*st.slider('Durability', min_value=1, max_value=scale)
   s_pow = scl*st.slider('Power', min_value=1, max_value=scale)
   s_com = scl*st.slider('Combat', min_value=1, max_value=scale)
   hero_stats = [s_int, s_str, s_spd, s_dur, s_pow, s_com]

   #st.markdown("Stat Total:", sum(hero_stats))

   # Every form must have a submit button.
   submitted = st.form_submit_button("Match Me Up!")
   if submitted:
       hero_stats = [s_int, s_str, s_spd, s_dur, s_pow, s_com]
       st.write(hero_stats)

st.write("Outside the form")


###### match hero_stats to character from dataframe
stat_cols = stats.columns[2:8]
x = stats[stat_cols].values
y = stats[['Name']].values
hero = np.array([hero_stats])
hero_knn = KNeighborsClassifier(n_neighbors=3)
hero_knn.fit(x, y)
ydf = pd.DataFrame(y).rename(columns={0:"name"})
mvs = hero_knn.kneighbors(hero, n_neighbors=1, return_distance=False)
mvs = mvs.tolist()[0]
match_list = ydf['name'][ydf.index.isin(mvs)].values.tolist()
match_df = stats[stats["Name"].isin(match_list)]


####### Display stuff about the matched hero
st.write(match_df)

match_stats = match_df.iloc[0,2:8].values.tolist()
#if st.button('Click Here to Keep This Party Going!'):
match_stats = list(np.array(match_stats)/scl)

### function for horizontal bar chart of stats
def stat_display(stuff):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    stat_names = ['Intelligence', 'Strength', 'Speed', 'Durability', 'Power', 'Combat']
    x_pos = np.arange(0,26,5)
    y_pos = np.arange(len(stat_names))
    bar_stats = stuff
    stats_bar_text_color = 'white'

    ax.barh(y_pos, bar_stats, align='center', color='red')
    ax.set_yticks(y_pos)
    ax.set_xticks(x_pos)
    ax.set_yticklabels(stat_names)
    ax.set_xlim([0,25])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('\n\nWeak  -------------------------------- Super ------------------------------ Godly\n(but still better than you!)                                                                  ')
    ax.set_title('Your Hero Stats', color=stats_bar_text_color)

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

    return st.pyplot(fig)


stat_display(match_stats)
#hero_match =




query = ''
query = st.text_input("Enter what comic action you want to read about:")
# queries = list([queries])

# Find the closest 5 sentences of the corpus for query sentence based on cosine similarity
if query == '':
    st.markdown("""What other action are you looking for?
    Type something like Spiderman fights The Green Goblin and loses! or
    Wolverine teams up with Black Widow!""")
else:
    top_k = min(5, len(corpus))
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    st.markdown("""---""")
    st.write("You searched for:   ", query, "\n")
    st.subheader("""**You can read about it in these exciting titles:**""")

    for score, idx in zip(top_results[0], top_results[1]):
        # st.write("(Score: {:.4f})".format(score))
        # st.write(corpus[idx], "(Score: {:.4f})".format(score))
        st.markdown("""---""")
        comic=df['comic_name'][df['all_review']==corpus[idx]]
        row_dict = df.loc[df['all_review']== corpus[idx]]
        # row2_dict = sum_df.loc[sum_df['all_review']== corpus[idx]]
        # row3_dict = df1.loc[df1['comic_name']==row_dict['comic_name'].values[0]]
        st.write("Comic Title: " , row_dict['comic_name'].values[0])
        #st.write("Hotel Review Summary: " , row2_dict['summary'].values[0])
        #st.write("Tripadvisor Link: [here](%s)" %row3_dict['url'].values[0], "\n")
        st.markdown("""---""")

### IDEA ###
# (first character that comes up is a profile of superman, but x'd out)
# Deadpool: "Looks like Thanos WAS rigth, so half of the unverse was wiped out right before
# you picked your hero, and yes, that even includes people from other universes...so YOU CAN'T
# be Superman! But we have a great alternate match for you!!
#
# (if they picked that Deadpool was right, it does the same, but based him him being 'right'
# about his question he forgot to display - and that is, I can beat up Superman! ...so YOU CAN'T
# be Superman! But we have a great alternate match for you!!


# ### Web scraping of NBA player Stats
# @st.cache
# def load_data(year):
#     url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
#     html = pd.read_html(url, header = 0)
#     df = html[0]
#     raw = df.drop(df[df.Age == 'Age'].index) # deletes repeating header rows
#     raw = raw.fillna(0)
#     playerstats = raw.drop(['Rk'], axis=1)
#
#     return (playerstats)
#
# playerstats = load_data(selected_year)
#
# ### sidebar menu - team selection ###
# sorted_unique_team = sorted(playerstats.Tm.unique())
# selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
#
# ### sidebar menu - team selection ###
# unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
# selected_pos = st.sidebar.multiselect('Team', unique_pos, unique_pos)
#
# ### filtering data
# df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) &
#                                 (playerstats.Pos.isin(selected_pos))]
#
#
# st.header('Display Player Stats of Selected Team(s)')
# st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' +
#             str(df_selected_team.shape[1]) + ' columns')
# st.dataframe(df_selected_team)
#
# #download NBA player stats dataframe
# #https://dicsuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
# def filedownload(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode() # strings <-> bytes conversion
#     href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
#     return href
#
# st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)
#
#
# if st.button('Intercorrelation Heatmap'):
#     st.header('Intercorrelation Matrix Heatmap')
#     df_selected_team.to_csv('output.csv', index=False) #need this for the heatmap to work correctly
#     df = pd.read_csv('output.csv') #need this for the heatmap to work correctly
#
#     corr = df.corr()
#     mask = np.zeros_like(corr)
#     mask[np.triu_indices_from(mask)] = True
#     with sns.axes_style("white"):
#         f, ax = plt.subplots(figsize=(7,5))
#         ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
#     st.pyplot(f)


#bottom
