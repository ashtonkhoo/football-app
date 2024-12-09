import re
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from functools import reduce
from datetime import datetime
from urllib.error import HTTPError
from bs4 import BeautifulSoup as soup

import streamlit as st

def load_comp_data(comp, season):
    st.session_state['team_data_loaded'] = False
    comp_id_mapping = {
        'Champions League': '8',
        'Europa League': '19',
        'Premier League': '9',
        'La Liga': '12',
        'Serie A': '11',
        'Ligue 1': '13',
        'Bundesliga': '20'
    }
    comp_id = comp_id_mapping[comp]
    comp = comp.replace(' ', '-')
    
    url = f'https://fbref.com/en/comps/{comp_id}/{season}/schedule/{season}-{comp}-Scores-and-Fixtures'
    
    fixturedata = pd.DataFrame([])
    tables = pd.read_html(url)

    # get fixtures
    fixtures = tables[0][['Wk', 'Day', 'Date', 'Time', 'Home', 'Away', 'xG', 'xG.1', 'Score']].dropna()
    fixtures['season'] = url.split('/')[6]
    fixturedata = pd.concat([fixturedata,fixtures])

    # data preprocessing
    fixturedata['Wk'] = fixturedata['Wk'].astype(int)
    fixturedata = fixturedata.rename(columns={
        'xG': 'H_xG',
        'xG.1': 'A_xG'
    })
    fixturedata.insert(8, 'H_Goals', preprocess_scores(fixturedata['Score'], 'home'))
    fixturedata.insert(9, 'A_Goals', preprocess_scores(fixturedata['Score'], 'away'))
    fixturedata.drop(['Day', 'Score', 'season'], axis=1, inplace=True)
    
    fixturedata[['H_xG', 'A_xG']] = round(fixturedata[['H_xG', 'A_xG']], 2)
    
    st.session_state['team_data_loaded'] = True
    
    return url, fixturedata
    
def load_player_stats(url, team, metric):
    html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    links = list(set(soup(html.content, "html.parser").find_all('a')))
    squad_link = ['https://fbref.com' + l.get('href') for l in links if team in l][0]
    tables = pd.read_html(squad_link)
    
    player_stats = tables[0]

    player_col = player_stats.columns[0]
    cols = [player_col, *[c for c in player_stats.columns if c[0] == metric]]

    player_stats = player_stats[cols]
    player_stats = player_stats[~player_stats[player_col].str.contains('Total')]
    player_stats.columns = [c[1] for c in player_stats.columns]

    heatmap_data = player_stats.set_index('Player').sort_values(player_stats.columns[1], ascending=False)
    heatmap_data = heatmap_data[heatmap_data.notna().all(axis=1)]
    return heatmap_data

def preprocess_scores(series, side):
    scores_array = series.to_numpy()
    
    if side == 'home':
        return np.array([int(score[0]) for score in scores_array])
    else:
        return np.array([int(score[-1]) for score in scores_array])
        
def calc_result(series):
    results_array = series.to_numpy()
    return np.array([('W' if result[0] > result[1] else 'L' if result[1] > result[0] else 'D') if result[-1] == 'H' else ('W' if result[0] < result[1] else 'L' if result[1] < result[0] else 'D') for result in results_array])

# Streamlit app section
st.title("Football Analysis App")

cur_year = datetime.today().year
cur_year = cur_year if datetime.today().month > 6 else cur_year-1

possible_seasons = [f'{i}-{i+1}' for i in range(2017, cur_year+1)][::-1]
season = st.sidebar.selectbox("Select Season", possible_seasons)

possible_comps = ['Champions League', 'Europa League', 'Premier League', 'La Liga', 'Serie A', 'Ligue 1', 'Bundesliga']
comp = st.sidebar.selectbox("Select Competition", possible_comps)

with st.spinner("Retrieving data and analyzing..."):

    url, league_data = load_comp_data(comp, season)

    if st.session_state.get('team_data_loaded', False):
        gw1 = league_data[league_data['Wk'] == 1]
        possible_teams = [*gw1['Home'].tolist(), *gw1['Away'].tolist()]
        possible_teams.sort()
        team = st.sidebar.selectbox("Select Team", possible_teams)
        
        league_data = league_data[(league_data['Home'] == team) | (league_data['Away'] == team)]
        league_data = league_data[(league_data['Home'] == team) | (league_data['Away'] == team)]
        league_data['Side'] = league_data.apply(lambda x: 'H' if x['Home'] == team else 'A', axis=1)
        league_data['Result'] = calc_result(league_data[['H_Goals', 'A_Goals', 'Side']])
        
        recent_form = ''.join(league_data['Result'].tail().tolist()[::-1])
        
        # Map results to corresponding colors
        color_map = {"L": "red", "D": "yellow", "W": "green"}

        # Create styled HTML text
        styled_form = "".join(
            f'<span style="color: {color_map[char]};">{char}</span>'
            for char in recent_form
        )

        # Display the styled text in the Streamlit app
        st.subheader(team)
        st.markdown(f"### Recent Form: {styled_form}", unsafe_allow_html=True)
        
        metric = st.selectbox(
            "Select player metric",
            ("Playing Time", "Performance", "Expected", "Progression", "Per 90 Minutes"),
        )
        player_stats = load_player_stats(url, team, metric)
        # Sort options
        sort_options = ["Original", "Sort by Row (Ascending)", "Sort by Row (Descending)"]
        selected_sort = st.selectbox("Sort Heatmap", sort_options)

        # Perform sorting based on user input
        if selected_sort == "Sort by Row (Ascending)":
            player_stats = player_stats.loc[player_stats.sum(axis=1).sort_values(ascending=True).index]
        elif selected_sort == "Sort by Row (Descending)":
            player_stats = player_stats.loc[player_stats.sum(axis=1).sort_values(ascending=False).index]

        # Create the heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=player_stats.values.tolist(),
                x=player_stats.columns.tolist(),
                y=player_stats.index.tolist(),
                colorscale="thermal",
                text=[[str(v) for v in row] for row in player_stats.values.tolist()],
                texttemplate="%{text}",
            )
        )

        # Adjust text size and layout
        fig.update_traces(textfont=dict(size=12))
        fig.update_layout(
            title="Player Stats Heatmap",
            xaxis_title="Metrics",
            yaxis_title="Players",
            yaxis=dict(autorange="reversed"),
            width=1000,
            height=600,
        )

        # Render in Streamlit
        st.plotly_chart(fig)