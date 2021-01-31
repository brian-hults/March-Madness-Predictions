# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:53:26 2021

NCAA Mens Basketball - EDA and Testing

@author: Brian Hults
"""

# Install pacakges
import pandas as pd
import time
from sportsreference.ncaab.teams import Teams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

start_time = time.time()

FIELDS_TO_DROP = ['away_points', 'home_points', 'date', 'location',
                  'losing_abbr', 'losing_name', 'winner', 'winning_abbr',
                  'winning_name', 'home_ranking', 'away_ranking']

dataset = pd.DataFrame()
teams = Teams()
for team in teams:
    dataset = pd.concat([dataset, team.schedule.dataframe_extended])
X = dataset.drop(FIELDS_TO_DROP, 1).dropna().drop_duplicates()
y = dataset[['home_points', 'away_points']].values

# Save the datasets to CSV for quicker load times
X.to_csv('ncaa-mens-bball-team-data.csv')
y.to_csv('ncaa-mens-bball-points-response.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}
model = RandomForestRegressor(**parameters).fit(X_train, y_train)
model.fit(X_train, y_train)
print(model.predict(X_test).astype(int), y_test)
print("Runtime: ", time.time() - start_time, ' seconds')








