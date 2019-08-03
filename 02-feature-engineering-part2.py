import numpy as np
import pandas as pd
import os.path
import os
import sys
from datetime import date, timedelta, datetime

'''
Creates relationships among the different attributes from the csv generated from 01-extract.py
these features may be helpful during our ML steps
'''

# initialize cols to be used
def create_cols(df):
    df['points'] = df['assists'] + df['goals']
    df['pp points'] = df['powerPlayGoals'] + df['powerPlayAssists']
    df['c_w_pts'] = 0
    df['c_wo_pts'] = 0

def points_y_n(x):
    if x > 0:
        return 1
    return 0

# ensure we compare the same player in the same season
def same_player(player_id1, season1, season2, player_id2):
    return (player_id1 == player_id2) and (season1 == season2)

# finds number of games played in the last n days
def games_last_ndays(df, curr_date, n, j, player_id, season):
    num_games = -1
    m = j

    for i in range(n):
        if (m < 0):
            break
        elif ( curr_date == df.loc[m]['date_time'] and m >= 0 and same_player(player_id, season, df.loc[m]['season'], df.loc[m]['player_id']) ):
            m = m - 1
            num_games += 1
        curr_date += timedelta(days=-1)

    return num_games

# finds number of points the last n games - pp_TF indicates if we're looking at powerplay points vs non-powerplay points
def points_last_n(df, n, j, player_id, season, pp_TF):
    pts = 0
    i = j
    k = i - n #termination constraint

    while i > 0 and i > k and same_player(player_id, season, df.loc[i-1]['season'], df.loc[i-1]['player_id']) :
        i = i - 1
        pts += df.at[i, pp_TF]

    return pts

# find number of consecutive games with points
def consec_with_points(df, i, player_id, season):
    if i > 0 and same_player(player_id, season, df.loc[i-1]['season'], df.loc[i-1]['player_id']) :
        if df.loc[i-1]['points'] == 0:
            df.at[i, 'c_w_pts'] = 0

        elif df.loc[i-1]['points'] > 0 and df.loc[i-1]['c_w_pts'] == 0 :
            df.at[i, 'c_w_pts'] = 1

        elif df.loc[i-1]['points'] > 0 and df.iloc[i-1]['c_w_pts'] > 0 :
            df.at[i,'c_w_pts'] = df.loc[i-1]['c_w_pts'] + 1
    
# find number of consecutive games without points
def consec_without_points(df, i, player_id, season):
    if i > 0 and same_player(player_id, season, df.loc[i-1]['season'], df.loc[i-1]['player_id']):
        if df.loc[i-1]['points'] > 0:
            df.at[i, 'c_wo_pts'] = 0

        elif df.loc[i-1]['points'] == 0 and df.loc[i-1]['c_wo_pts'] > 0 :
            df.at[i, 'c_wo_pts'] = df.loc[i-1]['c_wo_pts'] + 1

        elif df.loc[i-1]['points'] == 0 and df.iloc[i-1]['c_wo_pts'] == 0 :
            df.at[i,'c_wo_pts'] = 1

# find total number of points in season
def total_points(df, j, player_id, season):
    if j > 0 and same_player(player_id, season, df.loc[j-1]['season'], df.loc[j-1]['player_id']) :
        return df.loc[j]['points'] +  df.loc[j-1]['total pts']   
    return df.loc[j]['points'] 

# find total number of games in a season
def total_games(df, j, player_id, season):
    if j > 0 and same_player(player_id, season, df.loc[j-1]['season'], df.loc[j-1]['player_id']) :
        return 1 +  df.loc[j-1]['total games']   
    return 1

# update player's points per game
def ppg(df, i, player_id, season):
    if i > 0 and same_player(player_id, season, df.loc[i-1]['season'], df.loc[i-1]['player_id']) :
        return df.loc[i-1]['total pts'] / df.loc[i-1]['total games']
    return 0

# returns player's ppg over last n games
def ppg_l_n(df, n, j):
    if j < n:
        if j == 0:
            return 0
        return df.loc[j-1]['total pts'] / (j+1)
    return df.loc[j]['pts l_' + str(n)]/n

def main():

    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv(sys.argv[1],  parse_dates=['date_time'], date_parser=dateparse)
    # df = df.sort_values(['player_id', 'season', 'date_time_GMT'])
    create_cols(df)
    df['points y/n'] = df['points'].apply(points_y_n)

    for j in range(df.shape[0]):
        curr_date = df.loc[j]['date_time']
        curr_player_id = df.loc[j]['player_id']
        curr_season = df.loc[j]['season']

        # try:
        print(j, "Running", curr_player_id, "...        ", end="", flush=True)
        consec_with_points(df, j, curr_player_id, curr_season)
        consec_without_points(df, j, curr_player_id, curr_season)
        df.at[j, 'total games']            = total_games(df, j, curr_player_id, curr_season)
        df.at[j, 'total pts']              = total_points(df, j, curr_player_id, curr_season)
        df.at[j, 'ppg']                    = ppg(df, j, curr_player_id, curr_season)
        df.at[j, 'games l_21']             = games_last_ndays(df, curr_date, 21, j, curr_player_id, curr_season)
        df.at[j, 'games l_7']              = games_last_ndays(df, curr_date, 7, j, curr_player_id, curr_season )
        df.at[j, 'pts l_7']                = points_last_n(df, 7, j, curr_player_id, curr_season, 'points')
        df.at[j, 'pts l_3']                = points_last_n(df, 3, j, curr_player_id, curr_season, 'points')
        df.at[j, 'pp pts l_7']             = points_last_n(df, 7, j, curr_player_id, curr_season, 'pp points')
        df.at[j, 'pp pts l_3']             = points_last_n(df, 3, j, curr_player_id, curr_season, 'pp points')
        df.at[j, 'pts y/n l_7']            = points_last_n(df, 7, j, curr_player_id, curr_season, 'points y/n')
        df.at[j, 'pts y/n l_3']            = points_last_n(df, 3, j, curr_player_id, curr_season, 'points y/n')
        df.at[j, 'ppg l_7']                = ppg_l_n(df, 7, j)
        df.at[j, 'ppg l_3']                = ppg_l_n(df, 3, j)
        print("Done")

        # except:
        #     df.drop(index=j)
        #     print("WARNING: unable to create a file for", curr_player_id)

    # print(df)
    print('Finished')
    df.to_csv(path_or_buf=str(sys.argv[2]) + ".csv" , index=False)

if __name__ == '__main__':
    main()
