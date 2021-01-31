# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:53:26 2021

NCAA Mens Basketball - EDA and Testing

@author: Brian Hults
"""

# Install pacakges
import pandas as pd
import time
from sportsipy.ncaab.teams import Teams
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class EDA:
    def __init__(self, ):
        self.seed = 37
        self.unavailable = ['BROWN','COLUMBIA','CORNELL','DARTMOUTH','HARVARD',
                            'MARYLAND-EASTERN-SHORE','PENNSYLVANIA','PRINCETON',
                            'YALE']
        self.teams = Teams()
        self.combined_df = self.combine_schedules_from_file()


    def query_and_save_schedules(self):
        # Query for and get Updated Team schedules
        start_time = time.time()
        for team in self.teams:
            if team.abbreviation in self.unavailable:
                continue
            else:
                try:
                    team.schedule.dataframe_extended.to_pickle('./team_schedule_data/%s.pkl' % team.abbreviation.lower())
                except:
                    print('\n', team.abbreviation, ' Failed to Read/Write team to pickle!')
        print("Runtime: ", time.time() - start_time, ' seconds')
    
    
    def combine_schedules_from_file(self):
        # Read/Combine all team schedules for Variable Selection
        start_time = time.time()
        combined_df = pd.DataFrame()
        for team in self.teams:
            if team.abbreviation in self.unavailable:
                continue
            else:
                team_df = pd.read_pickle('./team_schedule_data/%s.pkl' % team.abbreviation.lower())
                combined_df = pd.concat([combined_df, team_df])
                
        # Reset index to a column so we can remove duplicates
        combined_df.reset_index(inplace=True)
        combined_df.rename(columns={'index':'Date_Home_Team_Index'}, inplace=True)
        
        # Remove NA values and duplicate records
        cleaned_combined_df = combined_df.dropna().drop_duplicates(subset='Date_Home_Team_Index').set_index('Date_Home_Team_Index')
        
        cleaned_combined_df.to_pickle('./team_schedule_data/cleaned_combined_schedule_df.pkl')
        print("Combine Schedules Runtime: ", time.time() - start_time, ' seconds')
        return cleaned_combined_df


    def variable_selection(self):
        # Perform Lasso CV variable selection on combined schedule dataframe
        # start_time = time.time()
        FIELDS_TO_DROP = ['away_points','home_points','losing_abbr','date','location',
                          'losing_name','winner','winning_abbr',
                          'winning_name','home_ranking','away_ranking']
        
        # Split data into features and response values
        X = self.combined_df.drop(FIELDS_TO_DROP,1)
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

# Run Lasso Variable Selection and print out results
lasso_home, lasso_away, home_train_r2, away_train_r2, home_test_MSE, away_test_MSE, feature_coef_df = ncaaEDA.variable_selection()

print(' Home R2: ', home_train_r2, '\n',
      'Away R2: ', away_train_r2, '\n',
      'Home Test MSE: ', home_test_MSE, '\n',
      'Away Test MSE: ', away_test_MSE)

top_home_features = feature_coef_df[['Features','Home_Coefs']].sort_values(by='Home_Coefs', ascending=False).head(10)
top_away_features = feature_coef_df[['Features','Away_Coefs']].sort_values(by='Away_Coefs', ascending=False).head(10)








