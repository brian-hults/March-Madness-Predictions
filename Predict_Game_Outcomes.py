# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:51:41 2021

NCAA Mens Basketball - Predict Game Outcomes

@author: Brian Hults
"""

# Install pacakges
import pandas as pd
from tqdm import tqdm
from SQL_Utils import SQL_Utils
from sportsipy.ncaab.teams import Teams
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

class Predict_Outcomes:
    # TODO: Figure out tournament matchup to predict on multiple games at a time
    def __init__(self, home_team, away_team, n_games=10):
        self.seed = 37
        self.unavailable = ['brown','columbia','cornell','dartmouth','harvard',
                            'maryland-eastern-shore','pennsylvania','princeton',
                            'yale', 'cal-poly']
        self.teams = [team.abbreviation.lower() for team in Teams()]
        self.sql = SQL_Utils()
        # self.game_list = pd.DataFrame(data={'Home': home_team, 'Away': away_team})
        self.prep_schedule_data(home_team, away_team, n_games)
            
        
    # def read_games(self):
    #     games_df = pd.read_excel('tournament_matchups.xlsx')
    #     return games_df
    
    
    def prep_schedule_data(self, home_team, away_team, n_games):
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
        FIELDS_TO_DROP = ['away_points','home_points','losing_abbr','date','location',
                          'losing_name','winner','winning_abbr',
                          'winning_name','home_ranking','away_ranking']
        # Compile the full feature dataframes
        home_full_df = self.home_schedule.drop(FIELDS_TO_DROP,1).tail(self.n_games)
        away_full_df = self.away_schedule.drop(FIELDS_TO_DROP,1).tail(self.n_games)
        
        # Query the variable selection results for most important features
        sorted_home_features = pd.read_sql_table('lasso_home_sorted_features', con=self.sql.engine).drop('index',1)
        sorted_away_features = pd.read_sql_table('lasso_away_sorted_features', con=self.sql.engine).drop('index',1)
        
        # Get the subsets of most important features
        top_home_features = sorted_home_features[abs(sorted_home_features['Home_Coefs']) > 0.05]['Features']
        top_away_features = sorted_away_features[abs(sorted_away_features['Away_Coefs']) > 0.05]['Features']
        
        # Compile the selected feature dataframes
        home_feature_df = self.home_schedule[top_home_features].tail(self.n_games)
        away_feature_df = self.away_schedule[top_away_features].tail(self.n_games)
        
        # Determine teams home/away status in their schedule
        home_idx = pd.Series(self.home_schedule.index.str.contains(self.home_abbreviation, regex=False),
                             index=self.home_schedule.index,
                             name='winner').replace({True:'Home', False:'Away'})
        away_idx = pd.Series(self.away_schedule.index.str.contains(self.away_abbreviation, regex=False),
                             index=self.away_schedule.index,
                             name='winner').replace({True:'Home', False:'Away'})
        
        # Build the corresponding responses (Loss=0, Win=1)
        home_cat_response = pd.Series(self.home_schedule['winner']==home_idx,
                                      index=self.home_schedule.index,
                                      name='Outcome').replace({True:'Win',False:'Loss'})
        away_cat_response = pd.Series(self.away_schedule['winner']==away_idx,
                                      index=self.away_schedule.index,
                                      name='Outcome').replace({True:'Win',False:'Loss'})
        
        # Rename the index columns to avoid confusion now that we are done matching, and take the tail
        home_idx = pd.Series(data=home_idx.values, index=home_idx.index, name='team_status')
        away_idx = pd.Series(data=away_idx.values, index=away_idx.index, name='team_status')
        
        # Get the points per game for home and away
        home_sch_points = pd.concat([home_idx, self.home_schedule['home_points'], self.home_schedule['away_points'], home_cat_response], axis=1)
        away_sch_points = pd.concat([away_idx, self.away_schedule['home_points'], self.away_schedule['away_points'], away_cat_response], axis=1)
        
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
        
        # home_cont_response = pd.Series([home_sch_points.loc[i,'home_points'] if home_sch_points.loc[i,'winner']=='Home' else home_sch_points.loc[i,'away_points'] for i in home_sch_points.index]).tail(self.n_games)
        # away_cont_response = pd.Series([away_sch_points.loc[i,'home_points'] if away_sch_points.loc[i,'winner']=='Home' else away_sch_points.loc[i,'away_points'] for i in away_sch_points.index]).tail(self.n_games)
        
        output = [home_full_df,
                  away_full_df,
                  home_feature_df,
                  away_feature_df,
                  home_cat_response,
                  home_cont_response,
                  away_cat_response,
                  away_cont_response]
        
        return output
    
    def train_regressors(self, X_home, X_away, y_home, y_away):
        # Fit linear regression models
        home_linReg_model = LinearRegression(normalize=True).fit(X_home, y_home)
        away_linReg_model = LinearRegression(normalize=True).fit(X_away, y_away)
        
        # Score trained models
        home_lr_r2 = home_linReg_model.score(X_home, y_home)
        away_lr_r2 = away_linReg_model.score(X_away, y_away)
        
        # print('Linear Regression:\n',
        #       'Home R2: ', home_lr_r2, '\n',
        #       'Away R2: ', away_lr_r2, '\n')
        
        home_rf_model = RandomForestRegressor(random_state=self.seed).fit(X_home, y_home)
        away_rf_model = RandomForestRegressor(random_state=self.seed).fit(X_away, y_away)
        
        # Score trained models
        home_rf_r2 = home_rf_model.score(X_home, y_home)
        away_rf_r2 = away_rf_model.score(X_away, y_away)
        
        print('Random Forest Regression:\n',
              'Home R2: ', home_rf_r2, '\n',
              'Away R2: ', away_rf_r2, '\n')
        
        return home_linReg_model, away_linReg_model, home_rf_model, away_rf_model
    
    
    def train_classifiers(self, X_home, X_away, y_home, y_away):
        # Fit Gaussian Naive Bayes models
        home_gnb_model = GaussianNB().fit(X_home, y_home)
        away_gnb_model = GaussianNB().fit(X_away, y_away)
        
        # Score trained models
        home_acc = home_gnb_model.score(X_home, y_home)
        away_acc = away_gnb_model.score(X_away, y_away)
        
        print('Gaussian Naive Bayes:\n',
              'Home Accuracy: ', home_acc, '\n',
              'Away Accuracy: ', away_acc, '\n')
        
        return home_gnb_model, away_gnb_model
        
    
    def reg_predict(self,
                     X_home,
                     X_away,
                     home_lr_model,
                     away_lr_model,
                     home_rf_model,
                     away_rf_model,
                     n_game_avg=5):
        # TODO: Figure out how to better represent the test data
        X_home_test = X_home.tail(n_game_avg).mean(axis=0).to_frame().transpose()
        X_away_test = X_away.tail(n_game_avg).mean(axis=0).to_frame().transpose()
        
        home_lr_pred = home_lr_model.predict(X_home_test)[0]
        away_lr_pred = away_lr_model.predict(X_away_test)[0]
        
        print('Linear Regression:\n',
              'Predicted Home Points: ', home_lr_pred, '\n',
              'Predicted Away Points: ', away_lr_pred, '\n')
        
        home_rf_pred = home_rf_model.predict(X_home_test)[0]
        away_rf_pred = away_rf_model.predict(X_away_test)[0]
        
        print('Random Forest Regression:\n',
              'Predicted Home Points: ', home_rf_pred, '\n',
              'Predicted Away Points: ', away_rf_pred, '\n')
        
        return home_lr_pred, away_lr_pred, home_rf_pred, away_rf_pred
    
    
    def clf_predict(self,
                    X_home,
                    X_away,
                    home_gnb_model,
                    away_gnb_model,
                    n_game_avg=5):
        X_home_test = X_home.tail(n_game_avg).mean(axis=0).to_frame().transpose()
        X_away_test = X_away.tail(n_game_avg).mean(axis=0).to_frame().transpose()
        
        home_gnb_pred = home_gnb_model.predict(X_home_test)
        away_gnb_pred = away_gnb_model.predict(X_away_test)
        
        print('Gaussian Naive Bayes:\n',
              'Home Model Probabilities: ', home_gnb_pred, '\n',
              'Away Model Probabilities: ', away_gnb_pred, '\n')
        
        return home_gnb_pred, away_gnb_pred

games_df = pd.read_excel('tournament_matchups.xlsx')
# correct_names = games_df.isin(teams)

round_results = []

for i in tqdm(range(len(games_df.index))):
    home = games_df['Home'].iloc[i]
    away = games_df['Away'].iloc[i]
    
    ncaab_Predictor = Predict_Outcomes(home, away)
    
    # Build features for game
    X_home_full, X_away_full, X_home, X_away, y_cat_home, y_cont_home, y_cat_away, y_cont_away = ncaab_Predictor.build_features()
    
    # Train regressors
    home_sel_lr_model, away_sel_lr_model, home_sel_rf_model, away_sel_rf_model = ncaab_Predictor.train_regressors(X_home, X_away, y_cont_home, y_cont_away)
    home_full_lr_model, away_full_lr_model, home_full_rf_model, away_full_rf_model = ncaab_Predictor.train_regressors(X_home_full, X_away_full, y_cont_home, y_cont_away)
    
    # Make regression predictions
    home_lr_pred1, away_lr_pred1, home_rf_pred1, away_rf_pred1 = ncaab_Predictor.reg_predict(X_home, X_away, home_sel_lr_model, away_sel_lr_model, home_sel_rf_model, away_sel_rf_model)
    home_lr_pred2, away_lr_pred2, home_rf_pred2, away_rf_pred2 = ncaab_Predictor.reg_predict(X_home_full, X_away_full, home_full_lr_model, away_full_lr_model, home_full_rf_model, away_full_rf_model)

    # Append results to round results list
    round_results.append([home_rf_pred2, away_rf_pred2])
    



# print('Selected Features Model:\n')
# home_sel_lr_model, away_sel_lr_model, home_sel_rf_model, away_sel_rf_model = ncaab_Predictor.train_regressors(X_home, X_away, y_cont_home, y_cont_away)

# print('Full Features Model:\n')
# home_full_lr_model, away_full_lr_model, home_full_rf_model, away_full_rf_model = ncaab_Predictor.train_regressors(X_home_full, X_away_full, y_cont_home, y_cont_away)

# print('Missouri (H) vs Kentucky (A) Prediction (selected features):', '\n')
# missouri_lr_pred1, kentucky_lr_pred1, missouri_rf_pred1, kentucky_rf_pred1 = ncaab_Predictor.reg_predict(X_home, X_away, home_sel_lr_model, away_sel_lr_model, home_sel_rf_model, away_sel_rf_model)

# print('Missouri (H) vs Kentucky (A) Prediction (full features):', '\n')
# missouri_lr_pred2, kentucky_lr_pred2, missouri_rf_pred2, kentucky_rf_pred2 = ncaab_Predictor.reg_predict(X_home_full, X_away_full, home_full_lr_model, away_full_lr_model, home_full_rf_model, away_full_rf_model)
        
        
        
        
        
        
        
        
        