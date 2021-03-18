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
        
        # Make Predictions for the Round of 64
        self.round64_results, self.round32_matchups = self.predict_round(self.round64_matchups)
        
        # Make Predictions for the Round of 32
        self.round32_results, self.round16_matchups = self.predict_round(self.round32_matchups)
        
        # Make Predictions for the Sweet 16
        self.round16_results, self.round8_matchups = self.predict_round(self.round16_matchups)
        
        # Make Predictions for the Elite 8
        self.round8_results, self.round4_matchups = self.predict_round(self.round8_matchups)
        
        # Make Predictions for the Final Four
        self.round4_results, self.round2_matchups = self.predict_round(self.round4_matchups)
        
        # Make a Prediction for the Championship
        self.round2_results, _ = self.predict_round(self.round2_matchups)
        
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

    def predict_round(self, round_matchups, last_round=False):
        round_results = []

        for i in tqdm(range(len(round_matchups.index))):
            home = round_matchups['Home'].iloc[i]
            away = round_matchups['Away'].iloc[i]
            
            self.predictor.prep_schedule_data(home, away)
            
            # Build features for game
            X_train, y_train_cont_home, y_train_cont_away, y_train_cat, X_full_test, X_select_home, X_select_away = self.predictor.build_features()
            
            # Train regressors
            home_full_rf_model, away_full_rf_model, home_full_r2, away_full_r2 = self.predictor.train_regressors(X_train, y_train_cont_home, y_train_cont_away)
            
            # Train classifiers
            full_gnb_model, full_gnb_acc = self.predictor.train_classifiers(X_train, y_train_cat)
            
            # Make regression predictions
            home_rf_pred_full, away_rf_pred_full = self.predictor.reg_predict(X_full_test, home_full_rf_model, away_full_rf_model)
            
            # Make classification predictions
            home_gnb_prob_full, away_gnb_prob_full = self.predictor.clf_predict(X_full_test, full_gnb_model)
        
            # Append results to round results list
            round_results.append([home_rf_pred_full, away_rf_pred_full, home_gnb_prob_full, away_gnb_prob_full])
            
        round_df = pd.concat([round_matchups, pd.DataFrame(data=round_results, columns=['home_points','away_points', 'home_clf_pred', 'away_clf_pred'])], axis=1)
        round_df['winner'] = np.where(round_df['home_points'] > round_df['away_points'], round_df['Home'], round_df['Away'])
        
        if not last_round:
            next_round_matchups = pd.concat([pd.Series(round_df['winner'].iloc[::2], name='Home').reset_index(drop=True), pd.Series(round_df['winner'].iloc[1::2], name='Away').reset_index(drop=True)], axis=1)
        else:
            next_round_matchups = None
        
        return round_df, next_round_matchups


Bracket_Generator()