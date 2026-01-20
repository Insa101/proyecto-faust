# 0 º Importamos la libreria Numpy

import numpy as np

## 1º Hacemos un Numpy Array para almacenar las notas, por alumno y en orden a cada asignatura.
    # Sera un array de 2 dimensiones.
notas = np.array([
    [9,4,8,3], # notas de Francisco
    [7,8,10,5], # notas de Lucia
    [10,8,6,8], #notas de Juan
    [7,4,8,4], #notas de Paula
    [8,5,6,5] #notas de Alba
    ])
print(40*"--")

## 2º Creamos una funcion con def: llamada "mostrar_suspensos"
    # Calculara cuantos alumnos suspendieron cada asignatura y mostrar el resultado
    # Haremos uso de una mascara booleana y contabilizaremos mediante np.sum() los valores 

def mostrar_suspensos (array):

    # Creamos una lista con asignaturas para mostrar los resultados de cada asignatura
    asignaturas = ["HTML", "JavaScript", "BBDD", "Programacion"]

    # Creamos un bucle for con rango 4 para pasar por las columnas o asignaturas
        # Crear una mascara booleana y evaluamos cada columna con nota si esta suspensas o no
        # Contabilizar el conjunto de notas suspensas para poder mostrarlas
        # Hacer un print con la asignatura y el conjunto de suspensas
    for n in range(4):
        suspenso = array[:, n] < 5
        count_suspensos = np.sum(suspenso)
        print ( f"{asignaturas[n]} ha sido suspendida por {count_suspensos} alumnos")

    # Invocamos la funcion para ver el resultado
mostrar_suspensos (notas)
print(40*"--")

## 3º Definimos una funcion llamada "calcular_media" para:
    # Calcular las medias de las notas de los alumnos
    # Mostrarlo por pantalla

def calcular_media (array):
    
    # Creamos una lista con los nombres de los alumnos para mostrar los resultados
    alumnos = ["Francisco", "Lucia", "Juan", "Paula", "Alba"]

    # Creamos un bucle for con rango 5 para pasar por cada fila o alumno.
        # Seleccionamos la fila, calculamos la media mediante .mean y redondeo a 2 decimales.
        # Mostramos los resultados para cada alumno con su media.
    for i in range (5):
        media_alumno = array[i,:].mean().round(2)
        print (f"{alumnos[i]} ha obtenido una nota media de: {media_alumno}")

    # Invocamos la funcion para ver el resultado:
calcular_media(notas)
print(40*"--")

## 4º Definimos una funcion llamada "calcular_aprobados" para:
    # Mostrar los nombres de los alumnos que han aprobado el curso
    # Aprueban aquellos que tengan nota media > 5 y sin suspensos.

def calcular_aprobados(array):

    # Creo una lista con los nombres, para mostrarlos, y una de aprobados vacia para agruparlos
    alumnos = ["Francisco", "Lucia", "Juan", "Paula", "Alba"]
    aprobados =[]

    # Genero un bucle for con un rango de 5 para pasar por todas las filas/alumnos para:
        # Filtrar que el alumno tiene todas las asignaturas con notas superiores o iguales a 5 con .all()
        # Filtrar que la media de las notas del alumno es superior o igual a 5
        # Añado en la lista de aprobados con .append()
        # Muestro toda la lista de aprobados
    for i in range (5):
        if np.all(array[i,:] >= 5):
            media_alumno = array[i,:].mean()
            if media_alumno >= 5:
                aprobados.append(alumnos[i])
    
    print (f"Los alumnos que han aprobado el curso son:", f"{aprobados}")    

    # Invoco a la funcion para mostrar los resultados:
calcular_aprobados(notas)
print(40*"--")