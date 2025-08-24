from fastapi import APIRouter

import sympy as sp
from numpy import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List
from models.playerItem import Player



router = APIRouter(prefix="/players", tags=["players"])

def gaussGetVariables(arr):

    print('Eliminacion por Gauss Simple')
    #Utilizamos eliminacion por gauss simple para la obtencion de las icognitas

    m = matrix(arr)

    #print(m)

    #print(m[1, 1])
    #print(m[1, 2])

    #print(arr)
    print("Matriz:" , m)

    print(m)
    print(m.shape)

    print("Matrix in Function" , m)

    row = 0
    column = 0
    isZero = False
    arr = np.array(m)

    #Feel It - (From Invincivle) by d4vd

    while isZero != True : 

        if m[row,column] == 0: 
            maxIndex = 0

            #Esto es para gacer pivoteo en caso de que alguno de los valores de la determinante sea 0, el cual no es el caso en este problema
            for i in range (len(arr)): #Aqui obtener el indice del row con el valor maximo
                if abs(arr[i][0]) > arr[maxIndex][0]:
                    maxIndex = i

            arr[[row,maxIndex]] = arr[[maxIndex,row]] #Aqui cambiamos de posicionn a los rows
            print("Matrix Swapeada : ", arr)

        for i in range(row + 1, m.shape[0]): #Aqui son las operaciones de la eliminacion hacia adelante

            newRow = arr[row, :] * (arr[i,column] / arr[row,column]) 
            arr[i,:] = arr[i,:] - newRow
            print(f"Ecuacion { i + 1 } : ", newRow)
            print("Nueva Row Prima : ", arr[i,:])

        print("New Matrix : " , arr)

        isZero = np.all(np.tril(arr,-1) == 0)

        if isZero :
            break

        row = row + 1
        column = column + 1

        #Por si se excede de los limites de busqueda
        if row >= arr.shape[0] or column >= arr.shape[0]:
            break

    
    print("Final Matrix : " , arr)


    #Eliminacion hacia atras

    m = matrix(arr)

    result = []
    column_size = m.shape[1]

    isOver = False
    row = m.shape[0] - 1
    column = m.shape[1] - 2
    j = 0

    #Closer - The Chainsmokers, Halseay 11:50 pm 8/Abril/2025

    while isOver != True :

        temp = m[row, column_size - 1] #Ya jalo

        while row >= 0 and column >= 0 : #Aqui se realizan las operaciones de gauss para obtener la elimiancion hacia atras

            if row == column:

                value = temp / m[row,column]
                
                print(f"Valor de la icognita del row {row + 1} : " , value)
                print(m[row,column])

                row = row - 1
                column = column_size - 2
                j = 0
                result.append(float(value))
                temp = m[row, column_size - 1]
            
            else:

                temp -= (m[row,column] * result[j])
                print("Valor actual : " , temp)
                j += 1
                column = column - 1
        

        if row == -1:
            isOver = True


    print("Resultado : " , result)
    return result

def sumElements(row): #Esta funcion es para sumar los elementos de la columna
    sum = 0
    for i in range(len(row)):
        sum += row[i]
    
    return float(sum)

def multiplyElements(row1,row2): #Este metodo es para multiplicar los datos de dos columnas y sumar sus resultados
    row1 = np.array(row1)
    row2 = np.array(row2)
    newRow = np.multiply(row1,row2)

    sumMultiply = sumElements(newRow)

    return float(sumMultiply)

def powerElement(row): #Esta funcione s para elevar los elementos de una columna al cuadrano y obtener la sumatoria
    sum = 0

    for i in range(len(row)):
        sum += row[i] * row[i]

    return float(sum)

def normalize(arr):
    arr = np.array(arr,dtype=float)
    return (arr - np.mean(arr)) / np.std(arr)

@router.post("/predict")
async def getPredictionResults(players : List[Player]): 

    countMed = [p.cantMed for p in players]
    countHard= [p.cantHard for p in players]
    timeUsed = [p.time for p in players]
    percentageOfWins = [p.percentageOfWin for p in players]

    n = len(countMed)


    data = [countMed,countHard,timeUsed,percentageOfWins]
    print(data)
    dataArray = np.array(data).T

    scaler = MinMaxScaler(feature_range=(1,100))
    dataStandarizared = scaler.fit_transform(dataArray)

    countMed = dataStandarizared[: , 0]
    countMed = dataStandarizared[: , 1]
    countMed = dataStandarizared[: , 2]
    countMed = dataStandarizared[: , 3]

    print(countMed)
    print(countHard)
    print(timeUsed)
    

    row1 = [n,sumElements(countMed), sumElements(countHard), sumElements(timeUsed), sumElements(percentageOfWins)]
    row2 = [sumElements(countMed),powerElement(countMed),multiplyElements(countMed,countHard),multiplyElements(countMed,timeUsed), multiplyElements(countMed,percentageOfWins)]
    row3 = [sumElements(countHard),multiplyElements(countHard,countMed), powerElement(countHard), multiplyElements(countHard,timeUsed), multiplyElements(countHard,percentageOfWins)]
    row4 = [sumElements(timeUsed), multiplyElements(timeUsed,countMed), multiplyElements(timeUsed,countHard),powerElement(timeUsed),multiplyElements(timeUsed,percentageOfWins)]

    m = [row1,row2,row3,row4]

    variables = gaussGetVariables(m)

    x1 = sp.Symbol('x1')
    x2 = sp.Symbol('x2')
    x3 = sp.Symbol('x3')
    
    exp = ''

    for i in range(len(variables)): #Here I develop the logic to get the mathematic  expression

        current = str(variables[i])

        if float(current) >= 0:
            exp += " + "
        
        if i == len(variables) - 1:
            exp += current
        else:
            exp += f"{current}*x{len(variables) - 1 - i}"

    print("Equation : ", exp)

    map = {}

    min_countMed = min(countMed)
    max_countMed = max(countMed)
    min_countHard = min(countHard)
    max_countHard = max(countHard)
    min_time = min(timeUsed)
    max_time = max(timeUsed)

    expSym = sp.simplify(exp)

    for player in players: #In thsi part for each player I compare and store their probabilities
        x1_norm = (player.cantMed - min_countMed) / (max_countMed - min_countMed)
        x2_norm = (player.cantHard - min_countHard) / (max_countHard - min_countHard)
        x3_norm = (player.time - min_time) / (max_time - min_time)

        evalFunc = expSym.evalf(4, subs={x1: x1_norm, x2: x2_norm, x3: x3_norm})
        map[player.name] = float(evalFunc)
    
    sorted_map = sorted(map.items(), key =lambda item : item[1], reverse=True) #Finally i sort the map, to get the higher poeple
    map = dict(sorted_map)

    return map


 

