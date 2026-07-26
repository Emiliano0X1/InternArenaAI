""" import numpy as np
from numpy import *

def gaussGetVariables(arr):

    print('Eliminacion por Gauss Simple')

    m = matrix(arr)
    print("Matriz:" , m)

    print(m)
    print(m.shape)

    print("Matrix in Function" , m)

    row = 0
    column = 0
    isZero = False
    arr = np.array(m)

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
    return result """