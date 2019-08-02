import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import sys
from datetime import date, timedelta


'''
Takes 3 different csv files and joins them on common attributes. If needed the program can
create individual player csv files containing game info for a player's career (from 2010-2011 to 2018-2019)
'''

# join on player_id
def match_player(df1, df2):
    return pd.merge(df1, df2, on=['player_id', 'player_id']) 

# join on game_id
def match_game(df1, df3):
    return [pd.merge(df1[i], df3, on=['game_id', 'game_id']) for i in range(len(df1))]

# chunking because join is pretty expensive
def chunk_df(df, n):
    return [df[i:i+n] for i in range(0,df.shape[0],n)]

# creates a csv for each player
def create_player_csv(player_id, df):
    player = df[df['player_id'] == player_id].sort_values(['season', 'date_time_GMT']).reset_index()
    
    try:
        file_name = str(player.loc[0]['firstName']) + " " + str(player.loc[0]['lastName']) + " " + str(player.loc[0]['player_id'])
        print("Running", file_name, "...        ", end="", flush=True)
        player.to_csv(path_or_buf=player_dir + file_name + ".csv", index=False)
        print("Done")
    except:
        print("WARNING: unable to create a file for", player_id)
        
def main():

    # read
    df1 = pd.read_csv(sys.argv[1])
    df2 = pd.read_csv(sys.argv[3])
    df3 = pd.read_csv(sys.argv[3])

    # join on player_id
    merge_player = match_player(df1, df2)

    # chunk because next join will be expensive
    chunked_df = chunk_df(merge_player, 20000)
    
    # join on game_id
    merge_game = match_game(chunked_df, df3)

    # concat chunked games
    df_complete = pd.concat(merge_game)

    # sort by player, season, then date
    df_complete = df_complete.sort_values(['player_id', 'season', 'date_time_GMT'])

    # print(df_complete)

    df_complete.to_csv(path_or_buf=str(sys.argv[4]) + ".csv", index=False)


if __name__ == '__main__':
    main()
