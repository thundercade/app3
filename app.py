# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:59:21 2022

@author: tonyd
"""
### imports ###
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier

with open("stats_df.pkl" , "rb") as file_1:
    stats = pkl.load(file_1)




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
