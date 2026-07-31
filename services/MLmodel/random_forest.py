import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class LeetRankingPrediction:
    def __init__(self, n_estimators=200, max_depth=10, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

        self.features_x = [
            "hard_ratio", "med_ratio","acceptance_ratio", "momentum", "attemps_easy", "attemps_med", "attemps_hard", "has_hard"
        ]

    def train(self, x_data, y_scores):
        x_df = pd.DataFrame(x_data, columns=self.features_x) #Define our matrix that were gonna train for RFR

        #This methods calibrates the Desicion Trees to learn the patters of penalization
        self.model.fit(x_df,y_scores) 

    def evaluate(self, x_test, y_test):

        x_df_test = pd.DataFrame(x_test, columns=self.features_x)
        predictions = self.model.predict(x_df_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return {
            "mean_squared_error" : mse,
            "r2_score" : r2
        }

    def predict_newPlayers(self, newPlayers):

        X_df_newPlayers = pd.DataFrame(newPlayers, columns=self.features_x)

        prediction = self.model.predict(X_df_newPlayers)

        return prediction.tolist()

    def saveModel(self, filepath='random_forest.joblib'):

        joblib.dump(self, filepath)
        print("Model save to prevent retrining in realtime")

    @staticmethod
    def loadModel(filepath='random_forest.joblib'):
        return joblib.load(filepath)
    


