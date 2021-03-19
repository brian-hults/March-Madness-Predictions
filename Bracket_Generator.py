# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:49:15 2021

NCAA Mens Basketball - Bracket Selections Generator

@author: Brian Hults
"""

# Install packages
import numpy as np
import pandas as pd
from tqdm import tqdm
from SQL_Utils import SQL_Utils
from Predict_Game_Outcomes import Predict_Outcomes
from sportsipy.ncaab.teams import Teams

class Bracket_Generator:
    def __init__(self):
        self.seed = 37
        self.unavailable = ['brown','columbia','cornell','dartmouth','harvard',
                            'maryland-eastern-shore','pennsylvania','princeton',
                            'yale', 'cal-poly']
        self.teams = [team.abbreviation.lower() for team in Teams()]
        self.sql = SQL_Utils()
        self.predictor = Predict_Outcomes()
        
        # Read the tournament matchup dataset
        self.round64_matchups = pd.read_excel('tournament_matchups.xlsx')
        # correct_names = games_df.isin(teams)
        
        
    def train_models(self):
        # Build features for game
        self.predictor.build_train_features() 
        
        # Train classifiers
        self.full_gnb_model, self.full_gnb_acc = self.predictor.train_classifiers()
        
        # Train regressors
        self.home_full_rf_model, self.away_full_rf_model, self.home_full_r2, self.away_full_r2 = self.predictor.train_regressors()
        
        return self.home_full_rf_model, self.away_full_rf_model, self.full_gnb_model
        
    
    def tune_rf_settings(self):
        # Run Grid Search on Random Forest model if indicated
        self.home_grid_search_results, self.away_grid_search_results = self.predictor.rf_grid_search()
        return self.home_grid_search_results, self.away_grid_search_results
    

    def predict_round(self, round_matchups, manual_picks=False, last_round=False):
        round_results = []
        
        if manual_picks:
                round_matchups = pd.read_excel(round_matchups)

        for i in tqdm(range(len(round_matchups.index))):
            # Prep the two team schedules and build the test df
            home = round_matchups['Home'].iloc[i]
            away = round_matchups['Away'].iloc[i]
            
            self.predictor.prep_schedule_data(home, away)
            self.predictor.build_test_features()
            
            # Make regression predictions
            home_rf_pred_full, away_rf_pred_full = self.predictor.reg_predict(self.home_full_rf_model, self.away_full_rf_model)
            
            # Make classification predictions
            gnb_probs_full = self.predictor.clf_predict(self.full_gnb_model)
        
            # Append results to round results list
            round_results.append([home_rf_pred_full, away_rf_pred_full, round(gnb_probs_full[1],4), round(gnb_probs_full[0],4)])
            
        round_df = pd.concat([round_matchups, pd.DataFrame(data=round_results, columns=['home_points','away_points', 'clf_home_prob', 'clf_away_prob'])], axis=1)
        round_df['winner'] = np.where(round_df['home_points'] > round_df['away_points'], round_df['Home'], round_df['Away'])
        
        if not last_round:
            next_round_matchups = pd.concat([pd.Series(round_df['winner'].iloc[::2], name='Home').reset_index(drop=True), pd.Series(round_df['winner'].iloc[1::2], name='Away').reset_index(drop=True)], axis=1)
        else:
            next_round_matchups = None
        
        return round_df, next_round_matchups
    
    
    def build_bracket(self, manual_picks=False):
        # Make Predictions for the Round of 64
        self.round64_results, self.round32_matchups = self.predict_round(self.round64_matchups)
        
        if manual_picks:
            # Make Predictions for the Round of 32
            self.round32_results, self.round16_matchups = self.predict_round('round32-my-picks.xlsx', manual_picks=True)
            # Make Predictions for the Sweet 16
            self.round16_results, self.round8_matchups = self.predict_round('round16-my-picks.xlsx', manual_picks=True)
            # Make Predictions for the Elite 8
            self.round8_results, self.round4_matchups = self.predict_round('round8-my-picks.xlsx', manual_picks=True)
            # Make Predictions for the Final Four
            self.round4_results, self.round2_matchups = self.predict_round('round4-my-picks.xlsx', manual_picks=True)
            # Make a Prediction for the Championship
            self.round2_results, _ = self.predict_round('round2-my-picks.xlsx', manual_picks=True, last_round=True)
            
        else:
            # Make Predictions for the Round of 32
            self.round32_results, self.round16_matchups = self.predict_round(self.round32_matchups)
            # Make Predictions for the Sweet 16
            self.round16_results, self.round8_matchups = self.predict_round(self.round16_matchups)
            # Make Predictions for the Elite 8
            self.round8_results, self.round4_matchups = self.predict_round(self.round8_matchups)
            # Make Predictions for the Final Four
            self.round4_results, self.round2_matchups = self.predict_round(self.round4_matchups)
            # Make a Prediction for the Championship
            self.round2_results, _ = self.predict_round(self.round2_matchups, last_round=True)
        
        # Save results
        self.round64_results.to_excel('round64-results.xlsx')
        self.round32_results.to_excel('round32-results.xlsx')
        self.round16_results.to_excel('round16-results.xlsx')
        self.round8_results.to_excel('round8-results.xlsx')
        self.round4_results.to_excel('round4-results.xlsx')
        self.round2_results.to_excel('round2-results.xlsx')
        spacer = pd.DataFrame(data=np.nan, index=range(1), columns=self.round64_results.columns)
        overall_results_df = pd.concat([self.round64_results,
                                        spacer,
                                        self.round32_results,
                                        spacer,
                                        self.round16_results,
                                        spacer,
                                        self.round8_results,
                                        spacer,
                                        self.round4_results,
                                        spacer,
                                        self.round2_results], axis=0)
        
        overall_results_df.to_sql('tournament-results', con=self.sql.engine, if_exists='replace')
        overall_results_df.to_excel('tournament-results.xlsx')



##############################################################################

# Initialize the class object
bracket_generator = Bracket_Generator()

# Tune Random Forest models
# home_grid_search_results, away_grid_search_results = bracket_generator.tune_rf_settings()

# Train models
home_rf_model, away_rf_model, gnb_model = bracket_generator.train_models()

# Make predictions and build the bracket
bracket_generator.build_bracket(manual_picks=True)