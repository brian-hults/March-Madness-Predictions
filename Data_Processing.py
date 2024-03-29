# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:53:26 2021

NCAA Mens Basketball - Data Processing

@author: Brian Hults
"""

# Install pacakges
import pandas as pd
import numpy as np
import time
import datetime
from tqdm import tqdm # this is the progress bar module
from SQL_Utils import SQL_Utils
from sportsipy.ncaab.teams import Teams

class Data_Processing:
    def __init__(self):
        self.seed = 37
        self.today = datetime.date.today()
        self.unavailable = [] # ['brown','columbia','cornell','dartmouth','harvard','maryland-eastern-shore','pennsylvania','princeton','yale','cal-poly']
        self.sql = SQL_Utils()
        self.teams = Teams(year='2022')
        
    
    def update_schedule_db(self):
        # Query for and get Updated Team schedules
        start_time = time.time()
        
        # Get the late updated date
        try:
            last_updated = datetime.date.fromisoformat(pd.read_sql_table('updated_date', con=self.sql.engine)['0'][0].strftime('%Y-%m-%d'))
        except:
            last_updated = False
            
        print("Last Updated: ", last_updated)
        
        # Initialize empty dataframe to combine schedules into
        combined_df = pd.DataFrame()
        
        # Iterate through teams
        for team in tqdm(self.teams):
            #if team.abbreviation.lower() == 'kentucky':
            #print(team.abbreviation)
            # Check if a team is in the list of unavailable records
            if team.abbreviation.lower() in self.unavailable:
                continue
            # If the table has ever been updated
            elif last_updated:
                # If the last update was before today
                if datetime.date.today() > last_updated:
                    #try:
                    schedule = team.schedule.dataframe_extended
                    # Convert the date column to a datetime object to use for filtering
                    # Source: https://stackoverflow.com/questions/26387986/strip-time-from-an-object-date-in-pandas
                    schedule['date'] = pd.to_datetime(schedule['date'], format='%B %d, %Y').apply(lambda x:x.date())
                    
                    # Filter out games that were before the last updated date
                    schedule = schedule[schedule['date'] > last_updated]
                    
                    # Get the two team names playing
                    team_matchups = schedule['winning_abbr','losing_abbr'].replace({team.abbreviation: np.nan})
                    opponents = team_matchups['winning_abbr'].combine_first(team_matchups['losing_abbr'])
                    print(opponents)
                    
                    # Determine team's home/away status
                    team_status = pd.Series(schedule.index.str.contains(team.abbreviation.lower(), regex=False),
                                            index=schedule.index,
                                            name='home_team').replace({True:'Home', False:'Away'})
                    
                    # Add columns for the Home and Away team abbreviations
                    home_team = team_status.replace({'Home': team.abbreviation,'Away': opponents})
                    
                    # Add team status to the dataframe
                    schedule = pd.concat([team_status, schedule], axis=1)
                    
                    # Convert Nan values to zeros
                    schedule.fillna(0, axis=0, inplace=True)
                    
                    # Push team schedule to database
                    schedule.to_sql(team.abbreviation.lower(), con=self.sql.engine, if_exists='append')
                    
                    # Add records to the combined dataframe
                    combined_df = pd.concat([combined_df, schedule], axis=0)
# =============================================================================
#                     except:
#                         print('\n', team.abbreviation, ' Failed to update team schedule in database!')
# =============================================================================
                else:
                    continue

            else:
                #try:
                schedule = team.schedule.dataframe_extended
                # Convert the date column to a datetime object to use for filtering
                # Source: https://stackoverflow.com/questions/26387986/strip-time-from-an-object-date-in-pandas
                schedule['date'] = pd.to_datetime(schedule['date'], format='%B %d, %Y').apply(lambda x:x.date())
                
                # Determine team's home/away status
                team_status = pd.Series(schedule.index.str.contains(team.abbreviation.lower(), regex=False),
                                        index=schedule.index,
                                        name='home_team') #.replace({True:'Home', False:'Away'})
                
                # Add team status to the dataframe
                schedule = pd.concat([team_status, schedule], axis=1)
                
                # Convert Nan values to zeros
                schedule.fillna(0, axis=0, inplace=True)
                
                schedule.to_sql(team.abbreviation.lower(), con=self.sql.engine, if_exists='replace')
                
                # Add records to the combined dataframe
                combined_df = pd.concat([combined_df, schedule])
# =============================================================================
#                 except:
#                     print('\n', team.abbreviation, ' Failed to write team schedule to database!')
# =============================================================================
        
        # Reset index to a column so we can remove duplicates
        combined_df.reset_index(inplace=True)
        combined_df.rename(columns={'index':'Date_Home_Team_Index'}, inplace=True)
        
        # Remove duplicate records
        cleaned_combined_df = combined_df.drop_duplicates(subset='Date_Home_Team_Index').set_index('Date_Home_Team_Index')
        
        # Write combined dataframe to the database
        cleaned_combined_df.to_sql('cleaned_combined_schedule',
                                   con=self.sql.engine,
                                   if_exists='replace')
        
        # Record the last date the schedule database was updated
        updated_date = pd.Series(data={'date': datetime.date.today()}) # datetime.date(2021,11,1)})
        updated_date.to_sql('updated_date', con=self.sql.engine, if_exists='replace')

        # Print runtime
        print("Update Schedule Database Runtime: ", time.time() - start_time, ' seconds')

# Create an instance of the Data Processing object
db_updater = Data_Processing()

# Update team schedules
db_updater.update_schedule_db()

# =============================================================================
# NCAAB Boxscore noted issue on Github -- Source: https://github.com/roclark/sportsipy/issues/674
# Fix is to edit the boxscore.py local file per the screencap at that source page. Reference code below:
# boxscore.py line 388:
# 
# records = boxscore(BOXSCORE_SCHEME[field]).items()
# records = [x.text() for x in records if x.text() != '']
# 
# if len(records) > index:
#     return records[index]
# else:
#     return ''
# =============================================================================








