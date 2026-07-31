from MLmodel.random_forest import LeetRankingPrediction


def getPredRanking(usernames, user_data):
    predictor = LeetRankingPrediction.loadModel()

    player_index = {}

    for index, username in enumerate(usernames):
        if not index in player_index:
            player_index[index] = username

    final_scores = predictor.predict_newPlayers(user_data)

    user_score = {}

    for index, score in enumerate(final_scores):
        username = player_index[index]
        user_score[username] = score

    sorted_scores = dict(sorted(user_score.items(), key= lambda item: item[1], reverse=True))
    return sorted_scores
