# Player Point Predictor

##### By Matthew Jung 301261171, Alex Hua 301261352

#### Data

All data used in the project came from a user on Kaggle.

#### Question

##### 1. Can we predict the amount of points a player will score in a future match based off of past performance.

##### 2. Can we predict whether a player gets a point in a game.

#### Getting Results

The project utilizes statistical methods, machine learning tools and computer science to answer the question. Correlative statistics is used to determine what features are the most statistically relevant to the answer. The Support Vector Classifier (SVC) is used to predict whether a player gets a point.

#### Data Analysis

A skater is defined as any player that is not a goalie. A filter stage filtered out skaters who had not played any games or had no game data. The dataset is further filtered for skaters who were non-defenseman, leaving only forwards.

Preliminary Analysis of the dataset consisted of production of histograms to identify potential outliers, scatterplots for visualization of the datasets.

New variables that are based off of the given dataset were created in attempt to create more meaningful relationships between the data.

Scatterplot and correlation analyses are done to identify which features are to be used in the machine learning model. Predictin whether a player gets a point is a binary classification problem, where 1 means a player gets a point and 0 means a player does not get a point. SVC was then chosen to make our predictions. The dataset is split into a 3:1 (or is it 4:1) ratio corresponding to a training:testing dataset. MinMaxScaler() was also used so that the features would be on the same scale.

The features chosen for the final model are:

1. timeOnIce (TOI) (r^2=0.286)
2. shots (r^2 = 0.252)
3. powerPlayTimeOnIce (ppTOI) (r^2 = 0.273)
4. points per game (ppg) (r^2 = 0.273)

GridSearchCV() was used to efficiently determine what parameters were optimal for the SVC model. 


## Results

A value of C = 4 with a linear kernel was determined to be the optimal parameter for the dataset. The model achieved a training score of 0.676 and a testing score of 0.676 ![Visual Representation of SVC model with C=4 and kernel = linear. Training Score = 0.676, testing score = 0.676](/model.png)

A Confusion matrix was also created to aid with the visual representation of false positives and false negatives [Confusion Matrix using the SVC model Total predictions = 68314. Correct predictions: category 0 (no points) = 41047 Top left, category 1 (points) = 5019 bottom right. Incorrect predictions: false negatives = 19085 bottom left, false positives = 3163 top right](/confusion_matrix.png)

## Discussion

For our model’s features, we analyzed time on ice (timeOnIce), shots, power play time on ice (powerPlayTimeOnIce), points per game (ppg), number of games in the last seven in which a player received at least a point (pts y/n l_7), number of games in the last three in which a player received at least a point (pts y/n l_3), number of games in the last twenty one days (games l_21), and number of games in the last seven days (games l_7) (Table 1, Appendix I-III). Of the features that we manually engineered, only ppg had improved our model (score = 0.658 without vs score = 0.676 with). Although pts y/n l_7 and pts y/n l_3 appeared to have a positively correlated association with points, the model could not create a decision boundary when they were both included – so they were removed. Because of games l_21 and game l_7’s very weak positive correlation with points, we opted not to use them. It was hypothesized that perhaps a better prediction could be made if the model could capture more subtle aspects of player performance: phenomenon such as momentum/point streaks, consistency, and fatigue. Specifically, the feature pts y/n l_x (x = number of games) was meant to track if a player was on a point streak. The more games a player had with points, the more likely he was on a hot streak, and therefore they could ride their momentum into the next game. Ppg was a feature meant to track a player’s consistency. The more points a player had per game could indicate how likely the player is to get a point in the next game – a player with a higher ppg is more likely than a player with a lower ppg. Finally, games l_d (games in the last d days) was meant to represent how fatigued a player could be. It was hypothesized that a player who plays more games in the last d days would be more fatigued than another player who has played less games over those same d days. This could result in a poorer player performance for the game being predicted. 
Here lies a limitation to our model: we assumed that all features had a linear relationship with points. However, in the absence of a linear relationship (~ 0 correlation) – such as games l_d – there could be a more sophisticated relationship between fatigue and points: one that is not linear. 


##### The data turned out


# test
## test
### test
#### test
##### test