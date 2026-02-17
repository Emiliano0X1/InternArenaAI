import numpy as np

### FUNCIONES PARA LA MATRIZ - GAUSS ###

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


### FUNCIONES PARA EL CALCULO DE TODOS LOS INPUTS NECESARIOS PARA EL CALCULO FINAL  ###

def ratioMedium(cantMed, total):
    return cantMed / total

def calculateScore(cantEasy, cantMed,cantHard): #Esta es la funcion que funciona para calcular el score total del player
    return (cantEasy + (cantMed * 3) + (cantHard * 5)) 