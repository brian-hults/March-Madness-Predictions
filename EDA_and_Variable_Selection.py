# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 11:27:57 2022

NCAA Mens Basketball - EDA and Variable Selection

@author: Brian Hults
"""

# Install pacakges
import pandas as pd
import time
import datetime
from tqdm import tqdm
from SQL_Utils import SQL_Utils
from sportsipy.ncaab.teams import Teams
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class EDA:
    def __init__(self):
        self.seed = 37
        self.today = datetime.date.today()
        self.unavailable = ['brown','columbia','cornell','dartmouth','harvard',
                            'maryland-eastern-shore','pennsylvania','princeton',
                            'yale','cal-poly']
        self.sql = SQL_Utils()
        self.teams = Teams()
        
    def variable_selection(self):
        # Perform Lasso CV variable selection on combined schedule dataframe
        # NOTE: LassoCV by default normalizes X features before fitting regression
        
        # Drop response and descriptive columns
        FIELDS_TO_DROP = ['away_points','home_points','losing_abbr',
                          'home_wins','away_wins','home_losses','away_losses',
                          'home_minutes_played', 'away_minutes_played',
                          'home_win_percentage','away_win_percentage',
                          'date','location','losing_name','winner','winning_abbr',
                          'winning_name','home_ranking','away_ranking']
        
        # Get the combined df from the database
        combined_df = pd.read_sql_table('cleaned_combined_schedule', con=self.sql.engine, index_col='Date_Home_Team_Index')
        
        # Split data into features and response values
        X = combined_df.drop(FIELDS_TO_DROP,1)
        y_home = combined_df['home_points']
        y_away = combined_df['away_points']
        
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
        
        
        # Sort features for each model by coefficient magnitude
        sorted_home_features = feature_coef_df[['Features','Home_Coefs']].sort_values(by='Home_Coefs', ascending=False)
        sorted_away_features = feature_coef_df[['Features','Away_Coefs']].sort_values(by='Away_Coefs', ascending=False)
        
        # Write coefficients to database
        sorted_home_features.to_sql('lasso_home_sorted_features', con=self.sql.engine, if_exists='replace')
        sorted_away_features.to_sql('lasso_away_sorted_features', con=self.sql.engine, if_exists='replace')
        
        # Write model performance results to database
        results_df = pd.DataFrame(data={'home_train_r2': home_train_r2,
                                        'away_train_r2': away_train_r2,
                                        'home_test_MSE': home_test_MSE,
                                        'away_test_MSE': away_test_MSE},
                                  index=range(1))
        
        results_df.to_sql('lasso_results', con=self.sql.engine, if_exists='replace')
        
        return lasso_home_model, lasso_away_model, results_df, feature_coef_df
    
# Create an instance of the EDA class
ncaaEDA = EDA()

# Run Lasso Variable Selection and print out results
lasso_home, lasso_away, results_df, feature_coef_df = ncaaEDA.variable_selection()

print(' Home R2: ', results_df['home_train_r2'][0], '\n',
      'Away R2: ', results_df['away_train_r2'][0], '\n',
      'Home Test MSE: ', results_df['home_test_MSE'][0], '\n',
      'Away Test MSE: ', results_df['away_test_MSE'][0])