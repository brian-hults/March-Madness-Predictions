# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:41:52 2022

NCAA Mens Basketball - LRMC Testing

@author: Brian Hults
"""
# Install pacakges
import pandas as pd
import numpy as np
import pickle
import datetime
import random
from SQL_Utils import SQL_Utils
from sportsipy.ncaab.teams import Teams
from sportsipy.ncaab.schedule import Schedule
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import time
#import Bracket_Generator

class Data_Processing:
    def __init__(self):
        self.seed = 37
        self.today = datetime.date.today()
        self.unavailable = [] #['brown','columbia','cornell','dartmouth','harvard','maryland-eastern-shore','pennsylvania','princeton','yale','cal-poly']
        self.sql = SQL_Utils()
        self.teams = Teams(year='2022')
        
    
    def update_schedule_db(self):
        # Query for and get Updated Team schedules
        start_time = time.time()
        
        # Get the late updated date
# =============================================================================
#         try:
#             last_updated = datetime.date.fromisoformat(pd.read_sql_table('updated_date', con=self.sql.engine)['0'][0].strftime('%Y-%m-%d'))
#         except:
# =============================================================================
        last_updated = False
            
        print("Last Updated: ", last_updated)
        
        # Initialize empty dataframe to combine schedules into
        combined_df = pd.DataFrame()
        
        # Iterate through teams
        for i, team in tqdm(enumerate(self.teams)):
            #if team.abbreviation.lower() == 'kentucky':
            #print(team.abbreviation)
            # Check if a team is in the list of unavailable records
            if team.abbreviation.lower() in self.unavailable:
                continue
            
            # Skip teams that were already loaded by index number if needed
# =============================================================================
#             elif i < 276:
#                 continue
# =============================================================================
            
            # If the table has ever been updated
            elif last_updated:
                # If the last update was before today
                if datetime.date.today() > last_updated:
                    #try:
                    schedule = team.schedule.dataframe_extended
                    # Convert the date column to a datetime object to use for filtering
                    # Source: https://stackoverflow.com/questions/26387986/strip-time-from-an-object-date-in-pandas
                    schedule['date'] = pd.to_datetime(schedule['date'], format='%B %d, %Y').apply(lambda x:x.date())
                    
                    # Filter out games that were before the last updated date and those on today's date to avoid data glitches
                    schedule = schedule[(schedule['date'] > last_updated) & (schedule['date'] < datetime.date.today())]
                    
# =============================================================================
#                     # Get the two team names playing
#                     team_matchups = schedule['winning_abbr','losing_abbr'].replace({team.abbreviation: np.nan})
#                     opponents = team_matchups['winning_abbr'].combine_first(team_matchups['losing_abbr'])
#                     print(opponents)
# =============================================================================
                    
                    # Determine team's home/away status
                    team_status = pd.Series(schedule.index.str.contains(team.abbreviation.lower(), regex=False),
                                            index=schedule.index,
                                            name='home_team').replace({True:'Home', False:'Away'})
                    
                    # Add columns for the Home and Away team abbreviations
# =============================================================================
#                     home_team = team_status.replace({'Home': team.abbreviation,'Away': opponents})
# =============================================================================
                    
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
                
                # Filter out games that were before the last updated date and those on today's date to avoid data glitches
                schedule = schedule[schedule['date'] < datetime.date.today()]
                
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
#db_updater = Data_Processing()

# Update team schedules
#db_updater.update_schedule_db()

sql = SQL_Utils()

combined_df = pd.read_sql_table('cleaned_combined_schedule', con=sql.engine, index_col='Date_Home_Team_Index').sort_values('date', ascending=True)

# Get the full set of unique Team names
teams = []
for team in Teams(year='2022'):
    teams.append(team.abbreviation)

def transition_df_setup(df, teams):
    # Create the Markov dictionary to hold all required data
    markov_dict = {}
    for t in teams:
        markov_dict[t] = pd.DataFrame(data=0, 
                                      index=teams, 
                                      columns=['Wins',
                                               'Losses',
                                               'Home_Margin',
                                               'Away_Margin',
                                               'Neutral_Margin',
                                               'Transition_Prob',
                                               'rxH_Prob',
                                               'rxA_Prob']
                                      )
        markov_dict[t]['Home_Margin'] = np.empty((len(teams), 0)).tolist()
        markov_dict[t]['Away_Margin'] = np.empty((len(teams), 0)).tolist()
        markov_dict[t]['Neutral_Margin'] = np.empty((len(teams), 0)).tolist()
        markov_dict[t]['rxH_Prob'] = np.empty((len(teams), 0)).tolist()
        markov_dict[t]['rxA_Prob'] = np.empty((len(teams), 0)).tolist()
    
    # Initialize the transition matrix in the same dictionary
    transition_df = pd.DataFrame(data=0, index=teams, columns=teams)
    
    # Iterate through the combined games dataframe and update the Markov dict
    for row in tqdm(df.itertuples()):
        winner = row[86]
        loser = row[82]
        
        if (winner in teams) and (loser in teams):
            winner_df = markov_dict[winner].copy()
            loser_df = markov_dict[loser].copy()
            
            # Updating wins and losses counts
            winner_df.at[loser,'Wins'] += 1
            loser_df.at[winner,'Losses'] += 1
            
            # Getting the total number of games each team has played
    # =============================================================================
    #         num_games_winner_played = winner_df[['Wins','Losses']].sum().sum()
    #         num_games_loser_played = loser_df[['Wins','Losses']].sum().sum()
    # =============================================================================
            
            # Get the total number of wins and losses for each team
    # =============================================================================
    #         winner_num_wins = winner_df['Wins'].sum()
    #         winner_num_losses = winner_df['Losses'].sum()
    #         loser_num_wins = loser_df['Wins'].sum()
    #         loser_num_losses = loser_df['Losses'].sum()
    # =============================================================================
            
            # Get the total number of wins/losses for this specific team pairing
    # =============================================================================
    #         winner_num_wins_this_opponent = winner_df.loc[loser,'Wins']
    #         winner_num_losses_this_opponent = winner_df.loc[loser,'Losses']
    #         loser_num_wins_this_opponent = loser_df.loc[winner,'Wins']
    #         loser_num_losses_this_opponent = loser_df.loc[winner,'Losses']
    # =============================================================================
            
            # Get the winning point differential of this game
            home_point_margin = row[63] - row[23] # (Home - Away) points
            away_point_margin = row[23] - row[63]
            
            # Get the location of the game
            #location = row[81]
            
            # Get the winning team schedule
            winner_sch = Schedule(winner)
            
            # Find this game in the schedule
            for game in winner_sch:
                date_str = game.date.split(" - ")[0]
                parsed_date = pd.to_datetime(date_str, format='%a, %b %d, %Y')
                #print("Game object date: ", parsed_date, " // Combined_DF Date: ", row[41])
                # Determine if the location is home, away, or neutral court for winning team
                if parsed_date == row[41]:
                    court_status = game.location
                    #print("Arena Status: ", court_status)
                    break
            
    # =============================================================================
    #         # Get the current transition probabilities
    #         winner_trans_prob = winner_df.at[winner,'Transition_Prob']
    #         loser_trans_prob = loser_df.at[loser,'Transition_Prob']
    #     
    #         # Update the standard transition probabilities
    #         winner_df['Transition_Prob'].loc[winner] = (1/num_games_winner_played) * ( (winner_num_wins * winner_trans_prob) + (winner_num_losses * (1-winner_trans_prob)) )
    #         winner_df['Transition_Prob'].loc[loser] = (1/num_games_winner_played) * ( (winner_num_wins_this_opponent * (1-winner_trans_prob)) + (winner_num_losses_this_opponent * winner_trans_prob) )
    #         loser_df['Transition_Prob'].loc[loser] = (1/num_games_loser_played) * ( (loser_num_wins * loser_trans_prob) + (loser_num_losses * (1-loser_trans_prob)) )
    #         loser_df['Transition_Prob'].loc[winner] = (1/num_games_loser_played) * ( (loser_num_wins_this_opponent * (1-loser_trans_prob)) + (loser_num_losses_this_opponent * loser_trans_prob) )
    #         
    #         # Get the current outscore method transition probabilities
    #         winner_trans_prob = winner_df.at[winner,'Outscore_Trans_Prob']
    #         loser_trans_prob = loser_df.at[loser,'Outscore_Trans_Prob']
    # =============================================================================
            
            # Use different approaches depending on court status
            if court_status == 'Home':
                # Add the point margin to the team dataframes
                winner_df['Home_Margin'].loc[loser].append(home_point_margin)
                loser_df['Away_Margin'].loc[winner].append(away_point_margin)
                
                # Update the outscore method transition probabilities
    # =============================================================================
    #             winner_df['Outscore_Trans_Prob'].loc[winner] = (1/num_games_winner_played)
    # =============================================================================
            
            elif court_status == 'Away':
                # Add the point margin to the team dataframes
                winner_df['Away_Margin'].loc[loser].append(away_point_margin) 
                loser_df['Home_Margin'].loc[winner].append(home_point_margin)
                
                # Update the outscore method transition probabilities
    # =============================================================================
    #             winner_df['Outscore_Trans_Prob'].loc[winner] = (1/num_games_winner_played)
    # =============================================================================
                
            elif court_status == 'Neutral':
                # Add the point margin to the team dataframes
                winner_df['Neutral_Margin'].loc[loser].append(abs(home_point_margin))
                loser_df['Neutral_Margin'].loc[winner].append(-1*abs(away_point_margin))
                
                # Update the outscore method transition probabilities
    # =============================================================================
    #             winner_df['Outscore_Trans_Prob'].loc[winner] = (1/num_games_winner_played)
    # =============================================================================
            
            else:
                print("Court Status could not be determined for team matchup: ", winner, ' vs ', loser)
            
            markov_dict[winner] = winner_df.copy()
            markov_dict[loser] = loser_df.copy()
    
    # Save and return output transition matrix
    transition_df.to_csv('transition_df.csv')
    with open('markov_dict.pkl', 'wb') as f:
        pickle.dump(markov_dict, f)
    
# =============================================================================
#     with open('markov_dict.pkl', 'wb') as f:
#         pickle.dump(markov_dict, f)
# =============================================================================
    
    return transition_df, markov_dict


# Run the transition df and markov dict generating function
#transition_df, markov_dict = transition_df_setup(combined_df, teams)

# Read the transition df and markov dict from file
#transition_df = pd.read_csv('transition_df.csv', index_col=0)
with open('markov_dict.pkl', 'rb') as f:
   markov_dict = pickle.load(f)

# Initialize Train and Test dictionaries
X_train_dict = {}
X_test_dict = {}
Y_train_dict = {}

X_Train_overall = []
Y_Train_overall = []

X_Test_overall = []

# Initialize a dictionary to hold the coefficients and probability results of Logistic Regression fits
LogReg_fit = {}

# Compile X matrix and Y response
for team in tqdm(markov_dict):
    df = markov_dict[team].copy()
    
    # Get training data where the pairwise teams played each other on each court
    X_train = pd.DataFrame(df['Home_Margin'].loc[(df['Home_Margin'].apply(len)!=0) & (df['Away_Margin'].apply(len)!=0)])
    
    if len(X_train) > 0:
        X_train_dict[team] = X_train
        [X_Train_overall.append(i) for i in X_train['Home_Margin'].explode('Home_Margin').tolist()]
        
        y_train_series = df['Away_Margin'].loc[(df['Home_Margin'].apply(len)!=0) & (df['Away_Margin'].apply(len)!=0)]
        Y_train = y_train_series.apply(lambda x: [1 if i>0 else 0 for i in x])
        Y_train_dict[team] = Y_train
        [Y_Train_overall.append(i) for i in Y_train.explode('Away_Margin').tolist()]
    else:
        pass
        # TODO: add notification for teams that did not have a pairwise matchup?
    
    # Get test data where the pairwise teams did not play on both courts
    X_test = pd.DataFrame(df['Home_Margin'].loc[(df['Home_Margin'].apply(len)!=0) & (df['Away_Margin'].apply(len)==0)])
    X_test_dict[team] = X_test
    [X_Test_overall.append(i) for i in X_test['Home_Margin'].explode('Home_Margin').tolist()]

X_Train_arr = np.array(X_Train_overall).reshape((-1,1))
Y_Train_arr = np.array(Y_Train_overall)
X_Test_arr = np.array(X_Test_overall).reshape((-1,1))

# Build a Logistic Regression classifier to determine the away win probabilities      
clf = LogisticRegression(random_state=11).fit(X_Train_arr, Y_Train_arr)

# Store the model coefficient and intercept for easy access later
LogReg_fit['a_coeff'] = clf.coef_[0,0]
LogReg_fit['b_intercept'] = clf.intercept_[0]

# Get the away win probabilities of the training data where we have a pairwise away game
#X_train_probs = clf.predict_proba(X_Train_arr)

# Iterate through the training team dictionaries to insert the away and neutral win probs, and home court advantage
for team in tqdm(X_train_dict):
    df = X_train_dict[team].copy()
    
    if len(df) > 0:
        df['Away_Win_Prob'] = [clf.predict_proba(np.array(x).reshape((-1,1)))[0][1] for x in df['Home_Margin']]
        df['Home_Advantage'] = [abs(-1*(LogReg_fit['a_coeff'] / LogReg_fit['b_intercept']) - h) for h in df['Home_Margin']]
        df['Neutral_Win_Prob'] = [clf.predict_proba(np.array(x + h).reshape((-1,1)))[0][1] for x, h in zip(df['Home_Margin'].tolist(), df['Home_Advantage'].tolist())]
        
        # Get current transition probs
        home_probs = markov_dict[team].loc[df.index,'rxH_Prob'].copy()
        away_probs = markov_dict[team].loc[df.index,'rxA_Prob'].copy()
        
        [home_probs.loc[i].append(x) for i, x in zip(df.index, df['Neutral_Win_Prob'].values)]
        [away_probs.loc[i].append(x) for i, x in zip(df.index, 1-df['Neutral_Win_Prob'].values)]
        
        # Update team transition probabilities
        markov_dict[team].loc[df.index,'rxH_Prob'] = home_probs
        markov_dict[team].loc[df.index,'rxA_Prob'] = away_probs
    else:
        print("No games in training data for: ", team)
    
    
# Set up the index of teams that have probabilities (plus teams that made it into the Tournament)
probs_idx = list(X_train_dict.keys()) + ['YALE']

# Initialize the transition matrix in the same dictionary
transition_df = pd.DataFrame(data=0, index=probs_idx, columns=probs_idx)

# Use RPI rankings as a seed for the state vector
rpi_rankings = pd.read_csv('RPI-Rankings.csv', usecols=['Team Abbr','RPI'], index_col=0, header=0).sort_index().squeeze(axis=1)
rpi_rankings = (rpi_rankings.loc[probs_idx].sort_index().reindex_like(transition_df).fillna(0.8)).sort_index()
rpi_rankings.rename("0", inplace=True)
#rpi_rankings = 2*((rpi_rankings - rpi_rankings.min()) / (rpi_rankings.max() - rpi_rankings.min()))

# Iterate through each team's dictionary and update the transition matrix with probabilities
for team in tqdm(markov_dict):
    if team in probs_idx:
        if team == 'YALE':
            transition_df.loc[:,team] = transition_df.loc[:,['MONTANA-STATE','LONGWOOD','COLGATE']].mean(axis=1)
            transition_df.loc[team,:] = transition_df.loc[['MONTANA-STATE','LONGWOOD','COLGATE'],:].mean(axis=0)
        else:
            df = markov_dict[team].copy()
            
            num_games_played = df[['Wins','Losses']].sum().sum()
            
            tii = (1/num_games_played) * (rpi_rankings.loc[team]) * (df['rxA_Prob'].apply(np.sum).sum() + df['rxH_Prob'].apply(np.sum).sum()) # 
            tij = (1/num_games_played) * (rpi_rankings.loc[rpi_rankings.index != team]) * (1-df['rxA_Prob'].loc[df.index != team].apply(np.sum) + (1-df['rxA_Prob'].loc[df.index != team].apply(np.sum))) # 
            
            transition_df.loc[team,team] = tii
            transition_df.loc[team, transition_df.index != team] = tij
            

    else:
        continue

transition_df = transition_df.div(transition_df.sum(axis=1), axis=0)

# Perform calculations for Pi_i for each team. [Steady-State probability of being in the state of team i]
#Ti = pd.Series(1 / (1-np.diag(transition_df)), index=transition_df.index)
#non_diag_df = pd.DataFrame(np.where(np.equal(*np.indices(transition_df.shape)), 0, transition_df.values), transition_df.index, transition_df.columns)

steady_state_probs = transition_df.sum(axis=1).div(transition_df.sum(axis=1), axis=0) # rpi_rankings.dot(transition_df)

state = steady_state_probs.div(steady_state_probs.sum(), axis=0)
state_hist = state
state_diff = [1]
diff = 1
while diff > 0.000000000000001:
    state_hist = state
    state = state.dot(transition_df)
    state = state.div(state.sum(), axis=0)
    diff = state.subtract(state_hist).abs().sum()
    state_diff.append(diff)
    
line_plt = plt.plot(state_diff)
plt.xlabel('Iterations')
plt.ylabel('State Transition Differential')
plt.title('Plot of State Transition Differential over Iterations')
plt.grid(True)
plt.show()

tournament_teams = pd.read_excel('tournament-teams.xlsx', header=0)

tournament_team_probs = state.loc[tournament_teams['Teams'].tolist()]

        
def matchup_prob(home_team, away_team, state):
    home_team_prob = state[home_team]
    away_team_prob = state[away_team]
    
    home_team_win_prob = 1 - (np.exp(-1834.72 * (home_team_prob - away_team_prob)) / (1 + np.exp(-1834.72 * (home_team_prob - away_team_prob))) )
    
    home_team_margin = 9180*(home_team_prob - away_team_prob)
    
    if home_team_win_prob > 0.5:
        winner = home_team
        win_prob = home_team_win_prob
        margin = home_team_margin
    elif home_team_win_prob < 0.5:
        winner = away_team
        win_prob = 1 - home_team_win_prob
        margin = -1*home_team_margin
    else:
        winner = random.choice([home_team, away_team])
        if winner == home_team:
            win_prob = home_team_win_prob
            margin = home_team_margin
        else:
            win_prob = 1 - home_team_win_prob
            margin = -1*home_team_margin
    
    return winner, win_prob, margin

first_round_matchups = pd.read_csv('first_round_matchups.csv', header=0)

first_round_results = []

for idx, row in first_round_matchups.iterrows():
    home_team = row['home']
    away_team = row['away']
    winner, win_prob, margin = matchup_prob(home_team, away_team, state)
    first_round_results.append([winner, win_prob, margin])
    
first_round_results_df = pd.DataFrame(first_round_results, columns=['winner','prob', 'margin'])
first_round_results_df.to_excel('first_round_results.xlsx')

second_round_matchups = pd.concat([pd.Series(first_round_results_df['winner'].iloc[::2], name='home').reset_index(drop=True), pd.Series(first_round_results_df['winner'].iloc[1::2], name='away').reset_index(drop=True)], axis=1)

second_round_results = []

for idx, row in second_round_matchups.iterrows():
    home_team = row['home']
    away_team = row['away']
    winner, win_prob, margin = matchup_prob(home_team, away_team, state)
    second_round_results.append([winner, win_prob, margin])
        
second_round_results_df = pd.DataFrame(second_round_results, columns=['winner','prob', 'margin'])
second_round_results_df.to_excel('second_round_results.xlsx')
        
third_round_matchups = pd.concat([pd.Series(second_round_results_df['winner'].iloc[::2], name='home').reset_index(drop=True), pd.Series(second_round_results_df['winner'].iloc[1::2], name='away').reset_index(drop=True)], axis=1)

third_round_results = []
        
for idx, row in third_round_matchups.iterrows():
    home_team = row['home']
    away_team = row['away']
    winner, win_prob, margin = matchup_prob(home_team, away_team, state)
    third_round_results.append([winner, win_prob, margin])
        
third_round_results_df = pd.DataFrame(third_round_results, columns=['winner','prob', 'margin'])
third_round_results_df.to_excel('third_round_results.xlsx')

fourth_round_matchups = pd.concat([pd.Series(third_round_results_df['winner'].iloc[::2], name='home').reset_index(drop=True), pd.Series(third_round_results_df['winner'].iloc[1::2], name='away').reset_index(drop=True)], axis=1)

fourth_round_results = []
        
for idx, row in fourth_round_matchups.iterrows():
    home_team = row['home']
    away_team = row['away']
    winner, win_prob, margin = matchup_prob(home_team, away_team, state)
    fourth_round_results.append([winner, win_prob, margin])
        
fourth_round_results_df = pd.DataFrame(fourth_round_results, columns=['winner','prob', 'margin'])
fourth_round_results_df.to_excel('fourth_round_results.xlsx')

fifth_round_matchups = pd.concat([pd.Series(fourth_round_results_df['winner'].iloc[::2], name='home').reset_index(drop=True), pd.Series(fourth_round_results_df['winner'].iloc[1::2], name='away').reset_index(drop=True)], axis=1)

fifth_round_results = []
        
for idx, row in fifth_round_matchups.iterrows():
    home_team = row['home']
    away_team = row['away']
    winner, win_prob, margin = matchup_prob(home_team, away_team, state)
    fifth_round_results.append([winner, win_prob, margin])
        
fifth_round_results_df = pd.DataFrame(fifth_round_results, columns=['winner','prob', 'margin'])
fifth_round_results_df.to_excel('fifth_round_results.xlsx')

final_round_matchups = pd.concat([pd.Series(fifth_round_results_df['winner'].iloc[::2], name='home').reset_index(drop=True), pd.Series(fifth_round_results_df['winner'].iloc[1::2], name='away').reset_index(drop=True)], axis=1)

final_round_results = []
        
for idx, row in final_round_matchups.iterrows():
    home_team = row['home']
    away_team = row['away']
    winner, win_prob, margin = matchup_prob(home_team, away_team, state)
    final_round_results.append([winner, win_prob, margin])

final_round_results_df = pd.DataFrame(final_round_results, columns=['winner','prob', 'margin'])
final_round_results_df.to_excel('final_round_results.xlsx')



