import numpy as np
import pandas as pd
import os.path
import os
import sys

def main():
    
    df = pd.read_csv(sys.argv[1])
    
    df = df.sort_values(['player_id', 'season', 'date_time_GMT'])
    df = df[~(df['primaryPosition'] == 'D')]
    print('Finished')
    df.to_csv(path_or_buf=str(sys.argv[2]) , index=False)

if __name__ == '__main__':
    main()
