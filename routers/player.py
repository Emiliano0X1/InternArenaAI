from fastapi import APIRouter

import sympy as sp
from numpy import *
import numpy as np

router = APIRouter(prefix="/players", tags=["players"])

def gaussGetVariables(arr):
    m = matrix(arr)
    row = 0
    column = 0
    isZero = False
    arr = np.array(m)

    while isZero != True:
        if m[row,column] == 0: #Made the pivoteo, in case there are zero in the diagonal matrix
            maxIndex = 0
            for i in range(len(arr)):
                if abs(arr[i][0]) > arr[maxIndex][0]:
                    maxIndex = i
            arr[[row,maxIndex]] = arr[[maxIndex,row]]
        for i in range(row + 1,m.shape[0]): #In this part we madethe elimination forward, to get the values of the variables
            newRow = arr[row,:] * (arr[i,column] / arr[row,column]) #
            arr[i, : ] = arr[i, :] - newRow
        isZero = np.all(np.tril(arr,-1) == 0)

        if isZero:
            break

        row = row + 1
        column = column + 1

        if row >= arr.shape[0] or column >= arr.shape[0]: #If exceed the limits of the matrix
            break
    
    #Starts the backward elimination
    m = matrix(arr)

    result = []
    column_size = m.shape[1]
    isOver = False
    row = m.shape[0] - 1
    column = m.shape[1] - 2
    j = 0

    while isOver != True:
        temp = m[row, column_size - 1] #We nade an temp row to mark the reference of the last elmenet in matrix
        while row >= 0 and column >= 0:
            if row == column:
                value = temp / m[row,column] #Ww calculate the row-value for these variable
                row = row - 1
                column = column_size - 2

                result.append(float(value))
                temp = m[row,column_size - 1]
            else:
                temp -= (m[row,column] * result[j]) #If not we added it to the temp variable until we get the last index pos
                j += 1
                column -= 1
        if row == -1:
            isOver = True
    

    return result
    
def sumElements(row): #These are helper functions for get the matrix operations, this sum all the values in the row
    sum = 0
    for num in row:
        sum += num
    return float(sum)
 
def multiplyElements(row1,row2): # Multiply two rows and sum the values of the newRow
    row1 = np.array(row1)
    row2 = np.array(row2)
    newRow = np.multiply(row1,row2)

    sum = sumElements(newRow)
    return float(sum)

def powerElements(row): #This sum the pow of all elemnets and return it
    sum = 0
    for num in row:
        sum += pow(num,2)
    return float(sum)


router.post("/")
async def getPredictionResults(json): 
    return {"message" : "This is the endpoint of multivariable calculator"}


 

