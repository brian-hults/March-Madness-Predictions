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
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

class Predict_Outcomes:
    def __init__(self):
        self.seed = 37
        self.unavailable = ['brown','columbia','cornell','dartmouth','harvard',
                            'maryland-eastern-shore','pennsylvania','princeton',
                            'yale', 'cal-poly']
        self.FIELDS_TO_DROP = ['home_team','away_points','home_points','losing_abbr',
                          'home_minutes_played', 'away_minutes_played',
                          'home_wins','away_wins','home_losses','away_losses',
                          'home_win_percentage','away_win_percentage',
                          'date','location','losing_name','winner','winning_abbr',
                          'winning_name','home_ranking','away_ranking']
        self.teams = [team.abbreviation.lower() for team in Teams()]
        self.sql = SQL_Utils()
    
    
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
        
        
    def build_train_features(self):
        # Get the combined df from the database
        combined_df = pd.read_sql_table('cleaned_combined_schedule', con=self.sql.engine, index_col='Date_Home_Team_Index')
        
        # Split data into features and response values
        self.X_train = combined_df.drop(self.FIELDS_TO_DROP,1)
        self.y_train_cat = combined_df['winner']
        self.y_train_cont_home = combined_df['home_points']
        self.y_train_cont_away = combined_df['away_points']
    
    
    def build_test_features(self):
        # Separate the home and away features for each team's schedule
        # Source to merge column headers: https://stackoverflow.com/questions/39741429/pandas-replace-a-character-in-all-column-names
        home1_features = self.home_schedule[self.home_schedule['home_team']==True].drop(self.FIELDS_TO_DROP,1).filter(regex='home', axis=1)
        # Get the features when the home team was away in their schedule and merge them with the home features in this df
        home2_features = self.home_schedule[self.home_schedule['home_team']==False].drop(self.FIELDS_TO_DROP,1).filter(regex='away', axis=1)
        home2_features.columns = home2_features.columns.str.replace('away','home')
        home_features = pd.concat([home1_features, home2_features], axis=0).tail(self.n_games).reset_index(drop=True)
        
        # Get the features when the away team was home in their schedule and merge them with the away features in this df
        away1_features = self.away_schedule[self.away_schedule['home_team']==True].drop(self.FIELDS_TO_DROP,1).filter(regex='home', axis=1)
        away1_features.columns = away1_features.columns.str.replace('home','away')
        away2_features = self.away_schedule[self.away_schedule['home_team']==False].drop(self.FIELDS_TO_DROP,1).filter(regex='away', axis=1)
        away_features = pd.concat([away1_features, away2_features], axis=0).tail(self.n_games).reset_index(drop=True)
        
        # Get the pace from both schedules and average them between the teams to balance the feature.
        home_pace = self.home_schedule['pace'].tail(self.n_games).reset_index(drop=True)
        away_pace = self.away_schedule['pace'].tail(self.n_games).reset_index(drop=True)
        avg_pace = pd.concat([home_pace, away_pace], axis=1).mean(axis=1)
        
        # Compile the team full feature dataframes
        full_test_df = pd.concat([away_features, home_features, avg_pace], axis=1)
        self.full_test_df = full_test_df.rename(columns={0:'pace'})
        
        # Take averages to build the test point for this game
        self.build_test_point()
    
    
    def build_test_point(self, n_game_avg=5):
        # TODO: Figure out how to better represent the test data
        self.X_test_avg = self.full_test_df.tail(n_game_avg).mean(axis=0).to_frame().transpose()
    
    
    def train_regressors(self):
        # Query best parameters from grid search results in database
        best_params_df = pd.read_sql_table('grid_search_best_params', con=self.sql.engine, index_col='index')
        
        # Fit Random Forest Regression models
        home_rf_model = RandomForestRegressor(n_estimators=best_params_df.loc['n_estimators','home_params'],
                                              max_depth=best_params_df.loc['max_depth','home_params'],
                                              random_state=self.seed).fit(self.X_train, self.y_train_cont_home)
        away_rf_model = RandomForestRegressor(n_estimators=best_params_df.loc['n_estimators','away_params'],
                                              max_depth=best_params_df.loc['max_depth','away_params'],
                                              random_state=self.seed).fit(self.X_train, self.y_train_cont_away)
        
        # Score trained models
        home_rf_r2 = home_rf_model.score(self.X_train, self.y_train_cont_home)
        away_rf_r2 = away_rf_model.score(self.X_train, self.y_train_cont_away)
        
        return home_rf_model, away_rf_model, home_rf_r2, away_rf_r2 
    
    
    def train_classifiers(self):
        # Fit Gaussian Naive Bayes model
        gnb_model = GaussianNB().fit(self.X_train, self.y_train_cat)
        
        # Score trained model
        gnb_acc = gnb_model.score(self.X_train, self.y_train_cat)
        
        return gnb_model, gnb_acc
        
    
    def reg_predict(self, home_rf_model, away_rf_model):
        home_rf_pred = home_rf_model.predict(self.X_test_avg)[0]
        away_rf_pred = away_rf_model.predict(self.X_test_avg)[0]
        return home_rf_pred, away_rf_pred
    
    
    def clf_predict(self, gnb_model):
        gnb_probs = gnb_model.predict_proba(self.X_test_avg)[0]
        return gnb_probs
    
    
    def rf_grid_search(self, param_grid=None):
        if not param_grid:
            param_grid = {'n_estimators': list(range(100,550,50)), 'max_depth': list(range(5,25,5))}
            
        # Build training features
        self.build_train_features()
            
        rf_model1 = RandomForestRegressor()
        rf_model2 = RandomForestRegressor()
        
        home_grid_search_results = GridSearchCV(rf_model1,
                                                param_grid,
                                                cv=10,
                                                n_jobs=-1,
                                                verbose=1).fit(self.X_train, self.y_train_cont_home)

        away_grid_search_results = GridSearchCV(rf_model2,
                                                param_grid,
                                                cv=10,
                                                n_jobs=-1,
                                                verbose=1).fit(self.X_train, self.y_train_cont_away)
        
        # Save grid search best parameters to the database
        home_params_df = pd.DataFrame.from_dict(data=home_grid_search_results.best_params_, orient='index')
        away_params_df = pd.DataFrame.from_dict(data=away_grid_search_results.best_params_, orient='index')
        grid_search_best_params = pd.concat([home_params_df, away_params_df], axis=1)
        grid_search_best_params.columns = ['home_params', 'away_params']
        grid_search_best_params.to_sql('grid_search_best_params', con=self.sql.engine, if_exists='replace')
        
        return home_grid_search_results, away_grid_search_results



        
        
        
        
        
        
        
        
        