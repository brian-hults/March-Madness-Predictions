# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:51:41 2021

NCAA Mens Basketball - Predict Game Outcomes

@author: Brian Hults
"""

# Install pacakges
import pandas as pd
from SQL_Utils import SQL_Utils
from sportsipy.ncaab.teams import Teams
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

class Predict_Outcomes:
    def __init__(self):
        self.seed = 37
        self.unavailable = ['brown','columbia','cornell','dartmouth','harvard',
                            'maryland-eastern-shore','pennsylvania','princeton',
                            'yale', 'cal-poly']
        self.teams = [team.abbreviation.lower() for team in Teams()]
        self.sql = SQL_Utils()
        # self.prep_schedule_data(home_team, away_team, n_games)
    
    
    def prep_schedule_data(self, home_team, away_team, n_games=10):
        if (home_team in self.teams) and (away_team in self.teams):
            self.home_abbreviation = home_team
            self.away_abbreviation = away_team
            self.home_schedule, self.away_schedule = self.query_schedules()
        
            # Compare the length of the schedule history for each team
            home_sch_len = len(self.home_schedule.index)
            away_sch_len = len(self.away_schedule.index)
            if home_sch_len == away_sch_len:
                self.n_games = home_sch_len            
            elif home_sch_len < away_sch_len:
                self.n_games = home_sch_len
            elif away_sch_len < home_sch_len:
                self.n_games = away_sch_len
            else:
                self.n_games = n_games
        else:
            print('ERROR! - Invalid team name(s) not found in season schedule!')
        
        
    def query_schedules(self):
        if (self.home_abbreviation in self.unavailable) or (self.away_abbreviation in self.unavailable):
            print('ERROR! - Team schedule not available!')
            return 'NA', 'NA'
        else:
            home_team_df = pd.read_sql_table(self.home_abbreviation, con=self.sql.engine, index_col='index')
            away_team_df = pd.read_sql_table(self.away_abbreviation, con=self.sql.engine, index_col='index')
            return home_team_df, away_team_df
        
        
    def build_features(self):
        FIELDS_TO_DROP = ['away_points','home_points','losing_abbr',
                          'date','location','losing_name','winner','winning_abbr',
                          'winning_name','home_ranking','away_ranking']
        
        # Get the combined df from the database
        combined_df = pd.read_sql_table('cleaned_combined_schedule', con=self.sql.engine, index_col='Date_Home_Team_Index')
        
        # Split data into features and response values
        X_combined = combined_df.drop(FIELDS_TO_DROP,1)
        y_combined_cat = combined_df['winner']
        y_combined_cont_home = combined_df['home_points']
        y_combined_cont_away = combined_df['away_points']
        
        # Query the variable selection results for most important features
        sorted_home_features = pd.read_sql_table('lasso_home_sorted_features', con=self.sql.engine).drop('index',1)
        sorted_away_features = pd.read_sql_table('lasso_away_sorted_features', con=self.sql.engine).drop('index',1)
        
        # Get the subsets of most important features
        top_home_features = sorted_home_features[abs(sorted_home_features['Home_Coefs']) > 0.1]['Features']
        top_away_features = sorted_away_features[abs(sorted_away_features['Away_Coefs']) > 0.1]['Features']
        
        # Determine teams home/away status in their schedule [NOTE: 'winner' column label is only used for logical comparison in following steps]
        # home_idx = pd.Series(self.home_schedule.index.str.contains(self.home_abbreviation, regex=False),
        #                      index=self.home_schedule.index,
        #                      name='winner').replace({True:'Home', False:'Away'})
        # away_idx = pd.Series(self.away_schedule.index.str.contains(self.away_abbreviation, regex=False),
        #                      index=self.away_schedule.index,
        #                      name='winner').replace({True:'Home', False:'Away'})
        
        # Separate the home and away features for each team's schedule
        home_features = pd.concat([self.home_schedule[self.home_schedule['home_team']==True].drop(FIELDS_TO_DROP,1).filter(regex='home', axis=1),
                                   self.home_schedule[self.home_schedule['home_team']==False].drop(FIELDS_TO_DROP,1).filter(regex='away', axis=1)], axis=0)
        away_features = pd.concat([self.away_schedule[self.away_schedule['home_team']==True].drop(FIELDS_TO_DROP,1).filter(regex='home', axis=1),
                                   self.away_schedule[self.away_schedule['home_team']==False].drop(FIELDS_TO_DROP,1).filter(regex='away', axis=1)], axis=0)
        home_pace = self.home_schedule['pace']
        away_pace = self.away_schedule['pace']
        
        # Compile the team full feature dataframes
        home_full_df = pd.concat([self.home_schedule.drop(FIELDS_TO_DROP,1).filter(regex='home', axis=1), self.away_schedule.drop]).tail(self.n_games)
        away_full_df = self.away_schedule.drop(FIELDS_TO_DROP,1).tail(self.n_games)
        
        # Compile the selected feature dataframes
        home_feature_df = self.home_schedule.filter(top_home_features, axis=1).tail(self.n_games)
        away_feature_df = self.away_schedule.filter(top_away_features, axis=1).tail(self.n_games)
        
        # Build the corresponding responses (Loss=0, Win=1)
        home_cat_response = pd.Series(self.home_schedule['winner']==self.home_schedule['team_status'],
                                      index=self.home_schedule.index,
                                      name='Outcome').replace({True:'Win',False:'Loss'})
        away_cat_response = pd.Series(self.away_schedule['winner']==self.away_schedule['team_status'],
                                      index=self.away_schedule.index,
                                      name='Outcome').replace({True:'Win',False:'Loss'})
        
        # Rename the index columns to avoid confusion now that we are done matching, and take the tail
        self.home_idx = pd.Series(data=home_idx.values, index=home_idx.index, name='team_status')
        self.away_idx = pd.Series(data=away_idx.values, index=away_idx.index, name='team_status')
        
        # Get the points per game for home and away
        home_sch_points = pd.concat([self.home_idx, self.home_schedule['home_points'], self.home_schedule['away_points'], home_cat_response], axis=1)
        away_sch_points = pd.concat([self.away_idx, self.away_schedule['home_points'], self.away_schedule['away_points'], away_cat_response], axis=1)
        
        # Build the corresponding continuous responses
        home_resp_list = []
        away_resp_list = []
        
        for i, j in zip(range(len(home_sch_points.index)), range(len(away_sch_points.index))):
            if home_sch_points['team_status'].iloc[i]=='Home':
                home_resp_list.append(home_sch_points['home_points'].iloc[i])
            else:
                home_resp_list.append(home_sch_points['away_points'].iloc[i])
                
            if away_sch_points['team_status'].iloc[j]=='Home':
                away_resp_list.append(away_sch_points['home_points'].iloc[j])
            else:
                away_resp_list.append(away_sch_points['away_points'].iloc[j])
            
        home_cont_response = pd.Series(home_resp_list).tail(self.n_games)
        away_cont_response = pd.Series(away_resp_list).tail(self.n_games)
        
        home_cat_response = home_cat_response.tail(self.n_games)
        away_cat_response = away_cat_response.tail(self.n_games)
        
        output = [X_combined,
                  y_combined_cont_home,
                  y_combined_cont_away,
                  y_combined_cat,
                  home_full_df,
                  away_full_df,
                  home_feature_df,
                  away_feature_df,
                  home_cat_response,
                  home_cont_response,
                  away_cat_response,
                  away_cont_response]
        
        return output
    
    def build_test_point(self, X_home, X_away, n_game_avg=5):
        # TODO: Figure out how to better represent the test data
        self.home_idx
        
        X_home_test = pd.concat([X_home.filter(regex='home'), ])
        
        X_home_test = X_home_test.tail(n_game_avg).mean(axis=0).to_frame().transpose()
        X_away_test = X_away.tail(n_game_avg).mean(axis=0).to_frame().transpose()
        return X_home_test, X_away_test
    
    def train_regressors(self, X_train, y_train_home, y_train_away):
        # Fit linear regression models
        # home_linReg_model = LinearRegression(normalize=True).fit(X_home, y_home)
        # away_linReg_model = LinearRegression(normalize=True).fit(X_away, y_away)
        
        # # Score trained models
        # home_lr_r2 = home_linReg_model.score(X_home, y_home)
        # away_lr_r2 = away_linReg_model.score(X_away, y_away)
        
        # print('Linear Regression:\n',
        #       'Home R2: ', home_lr_r2, '\n',
        #       'Away R2: ', away_lr_r2, '\n')
        
        home_rf_model = RandomForestRegressor(n_estimators=200, random_state=self.seed).fit(X_train, y_train_home)
        away_rf_model = RandomForestRegressor(n_estimators=200, random_state=self.seed).fit(X_train, y_train_away)
        
        # Score trained models
        home_rf_r2 = home_rf_model.score(X_train, y_train_home)
        away_rf_r2 = away_rf_model.score(X_train, y_train_away)
        
        # print('Random Forest Regression:\n',
        #       'Home R2: ', home_rf_r2, '\n',
        #       'Away R2: ', away_rf_r2, '\n')
        
        return home_rf_model, away_rf_model, home_rf_r2, away_rf_r2 # home_linReg_model, away_linReg_model,
    
    
    def train_classifiers(self, X_train, y_train):
        # Fit Gaussian Naive Bayes models
        gnb_model = GaussianNB().fit(X_train, y_train)
        
        # Score trained model
        gnb_acc = gnb_model.score(X_train, y_train)
        
        # print('Gaussian Naive Bayes:\n',
        #       'Home Accuracy: ', home_acc, '\n',
        #       'Away Accuracy: ', away_acc, '\n')
        
        return gnb_model, gnb_acc
        
    
    def reg_predict(self,
                    X_home,
                    X_away,
                    home_rf_model,
                    away_rf_model):
        # Build test point from team's prior schedule
        X_home_test, X_away_test = self.build_test_point(X_home, X_away)
        
        # home_lr_pred = home_lr_model.predict(X_home_test)[0]
        # away_lr_pred = away_lr_model.predict(X_away_test)[0]
        
        # print('Linear Regression:\n',
        #       'Predicted Home Points: ', home_lr_pred, '\n',
        #       'Predicted Away Points: ', away_lr_pred, '\n')
        
        home_rf_pred = home_rf_model.predict(X_home_test)[0]
        away_rf_pred = away_rf_model.predict(X_away_test)[0]
        
        # print('Random Forest Regression:\n',
        #       'Predicted Home Points: ', home_rf_pred, '\n',
        #       'Predicted Away Points: ', away_rf_pred, '\n')
        
        return home_rf_pred, away_rf_pred
    
    
    def clf_predict(self,
                    X_home,
                    X_away,
                    home_gnb_model,
                    away_gnb_model,
                    n_game_avg=10):
        # Build test point from team's prior schedule
        X_home_test, X_away_test = self.build_test_point(X_home, X_away)
        
        home_gnb_pred = home_gnb_model.predict_proba(X_home_test)
        away_gnb_pred = away_gnb_model.predict_proba(X_away_test)
        
        # print('Gaussian Naive Bayes:\n',
        #       'Home Model Probabilities: ', home_gnb_pred, '\n',
        #       'Away Model Probabilities: ', away_gnb_pred, '\n')
        
        return home_gnb_pred, away_gnb_pred



        
        
        
        
        
        
        
        
        