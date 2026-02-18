
from utils.math_func import powerElement, multiplyElements, sumElements, normalize, ratioMedium
import sympy as sp
from numpy import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List
from models.playerItem import Player
from services.gauss_logic import gaussGetVariables


def predict_winner(players : List[Player]):
    #1. Limpiar los datos antes de pasarlos a la matrix
    #1.1 Inicializar las filas con los datos de los players
    # X Lines =
    medRatio_x = [p.medRatio for p in players]
    daysActive_x = [p.daysActive for p in players]
    acceptanceRatio_x = [p.acceptanceRatio for p in players] 

    # Y Line = 
    scores_y = [p.score for p in players] 

    n = len(scores_y)
    #1.2 TODO - Si algun valor es 0 o nulo se debe poner el promedio de las columnas para evitar 

    data = [medRatio_x, daysActive_x, acceptanceRatio_x]
    dataArray = np.array(data).T

    scaler = MinMaxScaler(feature_range=(0,1))
    dataStandarizared = scaler.fit_transform(dataArray)

    medRatio_x = dataStandarizared[:, 0]
    daysActive_x = dataStandarizared[:,1]
    acceptanceRatio_x = dataStandarizared[:,2]

    #2 Construct the matrix based on the regresion lineal - gauss method
    row1 = [n,sumElements(medRatio_x), sumElements(daysActive_x),sumElements(acceptanceRatio_x), sumElements(scores_y)]
    row2 = [sumElements(medRatio_x), powerElement(medRatio_x), multiplyElements(medRatio_x,daysActive_x), multiplyElements(medRatio_x, acceptanceRatio_x), multiplyElements(medRatio_x,scores_y)]
    row3 = [sumElements(daysActive_x),multiplyElements(daysActive_x, medRatio_x),powerElement(daysActive_x), multiplyElements(daysActive_x, acceptanceRatio_x), multiplyElements(daysActive_x,scores_y)]
    row4 = [sumElements(acceptanceRatio_x), multiplyElements(acceptanceRatio_x,medRatio_x), multiplyElements(acceptanceRatio_x,daysActive_x), powerElement(acceptanceRatio_x), multiplyElements(acceptanceRatio_x, scores_y)]

    matrix = [row1,row2,row3,row4]

    #3. Send to the function of calculate the variables for the mathematic expression using gauss
    variables = gaussGetVariables(matrix)

    print("Vairables despues del request : ",variables)

    #5. Calculate by each player their prediction score, that would help us to determinate who can win
    map = {}

    b0 = variables[0]
    b1 = variables[1]
    b2 = variables[2]
    b3 = variables[3]

    for i,player in enumerate(players):
        m_norm = dataStandarizared[i,0]
        d_norm = dataStandarizared[i,1]
        a_norm = dataStandarizared[i,2]
        res = b0 + (b1 * m_norm) + (b2 * d_norm) + (b3 * a_norm)
        
        map[player.name] = float(res)
    
    #6. Finally I sort the map, to get the higher people based on their overall score
    sorted_map = sorted(map.items(), key =lambda item : item[1], reverse=True)
    map = dict(sorted_map)

    return map







    