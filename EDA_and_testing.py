# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:53:26 2021

NCAA Mens Basketball - EDA and Testing

@author: Brian Hults
"""

# Install pacakges
import pandas as pd
import time
import datetime
from SQL_Utils import SQL_Utils
from sportsipy.ncaab.teams import Teams
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class EDA:
    def __init__(self):
        self.seed = 37
        self.today = datetime.date.today()
        self.unavailable = ['BROWN','COLUMBIA','CORNELL','DARTMOUTH','HARVARD',
                            'MARYLAND-EASTERN-SHORE','PENNSYLVANIA','PRINCETON',
                            'YALE','CAL-POLY']
        self.sql = SQL_Utils()
        self.teams = Teams()
        
    
    def update_schedule_db(self):
        # Query for and get Updated Team schedules
        start_time = time.time()
        
        # Get the late updated date
        try:
            last_updated = datetime.date.fromisoformat(pd.read_sql_table('updated_date', con=self.sql.engine)['0'][0].strftime('%Y-%m-%d'))
        except:
            last_updated = False
        
        for team in self.teams:
            # Get the dates for the games in the schedule that have already been played
            # gamedays = [datetime.datetime.strptime(game.date, '%a, %b %d, %Y').date() for game in Schedule(team.abbreviation)] 
            
            if team.abbreviation in self.unavailable:
                continue
            elif last_updated:
                if datetime.date.today() <= last_updated:
                    try:
                        schedule = team.schedule.dataframe_extended
                        # Convert the date column to a datetime object to use for filtering
                        # Source: https://stackoverflow.com/questions/26387986/strip-time-from-an-object-date-in-pandas
                        schedule['date'] = pd.to_datetime(schedule['date'], format='%B %d, %Y').apply(lambda x:x.date())
                        
                        # Filter out games that were before the last updated date
                        schedule = schedule[schedule['date'] > last_updated]
                        if len(schedule) > 0:
                            schedule.to_sql(team.abbreviation.lower(), con=self.sql.engine, if_exists='append')
                    except:
                        print('\n', team.abbreviation, ' Failed to update team schedule in database!')

            else:
                try:
                    schedule = team.schedule.dataframe_extended
                    # Convert the date column to a datetime object to use for filtering
                    # Source: https://stackoverflow.com/questions/26387986/strip-time-from-an-object-date-in-pandas
                    schedule['date'] = pd.to_datetime(schedule['date'], format='%B %d, %Y').apply(lambda x:x.date())
                    schedule.to_sql(team.abbreviation.lower(), con=self.sql.engine, if_exists='replace')
                except:
                    print('\n', team.abbreviation, ' Failed to write team schedule to database!')
        
        # Record the last date the schedule database was updated
        updated_date = pd.Series(data={'date': datetime.date.today()})
        updated_date.to_sql('updated_date', con=self.sql.engine, if_exists='replace')

        self.combine_schedules()
        # Print runtime
        print("Update Schedules Runtime: ", time.time() - start_time, ' seconds')
        # return combined_df
    
    
    def combine_schedules(self):
        # Read/Combine all team schedules for Variable Selection
        start_time = time.time()
        
        # Check what the latest date is in the existing database combined schedule
        
        
        combined_df = pd.DataFrame()
        for team in self.teams:
            if team.abbreviation in self.unavailable:
                continue
            else:
                team_df = pd.read_sql_table(team.abbreviation.lower(), self.sql.engine)
                combined_df = pd.concat([combined_df, team_df])
                
        # Reset index to a column so we can remove duplicates
        combined_df.reset_index(inplace=True)
        combined_df.rename(columns={'index':'Date_Home_Team_Index'}, inplace=True)
        
        # Remove NA values and duplicate records
        cleaned_combined_df = combined_df.dropna().drop_duplicates(subset='Date_Home_Team_Index').set_index('Date_Home_Team_Index')
        
        # Write combined dataframe to the database
        cleaned_combined_df.to_sql('cleaned_combined_schedule', con=self.sql.engine)
        
        # Print runtime
        print("Combine Schedules Runtime: ", time.time() - start_time, ' seconds')
        # return cleaned_combined_df


    def variable_selection(self):
        # Perform Lasso CV variable selection on combined schedule dataframe
        # NOTE: LassoCV by default normalizes X features before fitting regression
        # start_time = time.time()
        FIELDS_TO_DROP = ['away_points','home_points','losing_abbr','date','location',
                          'losing_name','winner','winning_abbr',
                          'winning_name','home_ranking','away_ranking']
        
        # Get the combined df from the database
        combined_df = pd.read_sql_table('cleaned_combined_schedule', con=self.sql.engine)
        
        # Split data into features and response values
        X = combined_df.drop(FIELDS_TO_DROP,1)
        y_home = self.combined_df['home_points']
        y_away = self.combined_df['away_points']
        
        # Create training and test splits
        X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(X,
                                                                                                 y_home,
                                                                                                 y_away,
                                                                                                 test_size=0.2,
                                                                                                 random_state=self.seed)
        # Fit Lasso Models
        lasso_home_model = LassoCV(cv=10, random_state=self.seed).fit(X_train, y_home_train)
        lasso_away_model = LassoCV(cv=10, random_state=self.seed).fit(X_train, y_away_train)
        
        # Score training fits
        home_train_r2 = lasso_home_model.score(X_train, y_home_train)
        away_train_r2 = lasso_away_model.score(X_train, y_away_train)
        
        # Make predictions
        home_preds = lasso_home_model.predict(X_test)
        away_preds = lasso_away_model.predict(X_test)
        
        # Score predictions
        home_test_MSE = mean_squared_error(y_home_test, home_preds)
        away_test_MSE = mean_squared_error(y_away_test, away_preds)
        
        # Save features and model coefficients in a dataframe
        feature_coef_df = pd.DataFrame.from_dict(data={'Features':X.columns,
                                                       'Home_Coefs':lasso_home_model.coef_,
                                                       'Away_Coefs':lasso_away_model.coef_})
        output = [lasso_home_model,
                  lasso_away_model,
                  home_train_r2,
                  away_train_r2,
                  home_test_MSE,
                  away_test_MSE,
                  feature_coef_df]
        
        #print("Lasso Runtime: ", time.time() - start_time, ' seconds \n')
        return output

# Create an instance of the EDA class
ncaaEDA = EDA()

# Update team schedules
ncaaEDA.update_schedule_db()

# Run Lasso Variable Selection and print out results
lasso_home, lasso_away, home_train_r2, away_train_r2, home_test_MSE, away_test_MSE, feature_coef_df = ncaaEDA.variable_selection()

print(' Home R2: ', home_train_r2, '\n',
      'Away R2: ', away_train_r2, '\n',
      'Home Test MSE: ', home_test_MSE, '\n',
      'Away Test MSE: ', away_test_MSE)

top_home_features = feature_coef_df[['Features','Home_Coefs']].sort_values(by='Home_Coefs', ascending=False).head(10)
top_away_features = feature_coef_df[['Features','Away_Coefs']].sort_values(by='Away_Coefs', ascending=False).head(10)








