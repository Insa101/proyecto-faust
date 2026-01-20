# 0º Traer la libreria Pandas y Numpy

import pandas as pd
import numpy as np

# 1º Crea una lista que:
    # Contenga 10 numeros enteros aletorios entre el 0 y el 20.
    # Convertir la lista en una Serie de pandas y llamarla "serieA" 
    # Mostrar el contenido de la variable

lista_1 = np.random.randint(0,20,10)
serieA = pd.Series(lista_1, name = "serieA")
print(f" \nLa SerieA contiene los siguientes numeros:")
print(f"\n{serieA}")
print (40*"--")

# 2º Crear un array que:
    # Contenga 10 numeros enteros aletorios entre el 0 y el 20.
    # Convertir la lista en una Serie de pandas llamada "serieB"
    # Mostrar el contenido de la variable

array_1 = np.array(np.random.randint(0,20,10))
serieB = pd.Series(array_1, name = "serieB")
print(f" \nLa SerieB contiene los siguientes numeros:")
print(f"\n{serieB}")
print (40*"--")

# 3º Definir una funcion llamada "encontrar_posicion" que realice:
    # Reciba una serie como parametro
    # Compruebe los numeros de la serie que sean multiplos de 3
    # Los muestre por pantalla

def encontrar_posicion (serie):

    # Crearemos una lista para almacenar los multiplos de 3
    multiplos_3 = []

    # Haremos un bucle for para recorrer todos los numeros de la serie
    # Haremos un condicional para recopilar aquellos que sean multiples de 3.
    # Añadiremos estos a la variable
    # Haremos un print con el resultado
    for i in serie:
        if (i % 3 == 0) is True:
            multiplos_3.append(i)
    print(f" \nLos multiplos de 3 de la {serie.name} son:")
    print (f"\n{multiplos_3}")
    print (40*"--") 

    # Invocaremos la funcion con "serieA" como parametro.
encontrar_posicion(serieA)

# 4º Definir una funcion llamada "encontrar_comunes" que realice:
    # Reciba 2 series como parametro
    # Compruebe los elementos comunes 
    # Los muestre por pantalla el resultado

def encontrar_comunes (serie1, serie2):

    # Crearemos una lista para almacenar los comunes
    n_comunes = []

    # Haremos un doble bucle for para recorrer todos los numeros de una de las series
    # Haremos un condicional para recopilar aquellos que sean comunes entre ambas series
    # Añadimos un condicionante para evitar duplicar numeros ya incluidos en la lista.
    # Añadiremos estos a la variable
    # Haremos un print con el resultado
    for i in serie1:
        for n in serie2:
            if (i == n) is True:
                if (i in n_comunes) is False:
                    n_comunes.append(i)
    print(f" \nLos elementos comunes entre las series {serie1.name} y {serie2.name} son:")
    print (f"\n{n_comunes}")
    print (40*"--")

    # Invocaremos la funcion con "serieA" y "serieB" como parametros.
encontrar_comunes(serieA,serieB)

# 5º Definir una funcion llamada "encontrar_unicos" que realice:
    # Reciba 2 series como parametro
    # Compruebe los elementos de la primera serie que no se encuentren en la segunda
    # Los muestre por pantalla el resultado

def encontrar_unicos (serie1, serie2):

    # Crearemos una lista para almacenar los valores unicos
    n_unicos = []

    # Haremos un doble bucle for para recorrer todos los numeros de la 1º serie sobre los de la 2º.
    # Añadimos un condicionante para comprobar si el valor es unico
        # En caso de que ya este en el listado, lo eliminaremos
    # Añadiremos los valores comprobados a la variable con valores unicos
    # Haremos un print con el resultado
    for i in serie1:
        for n in serie2:
            if (i == n) is True:
                if (i in n_unicos) is True:
                    n_unicos.remove(i)
                else:
                    n_unicos.append(i)
    print(f" \nLos elementos unicos entre las series {serie1.name} y {serie2.name} son:")
    print (f"\n{n_unicos}")
    print (40*"--")

    # Invocaremos la funcion con "serieA" y "serieB" como parametros.
encontrar_unicos(serieA,serieB)

# 6º Generar un DataFrame de pandas combinando las 2 seriees y asignando un nombre al indice
    # Primero genero el dataframe
    # Luego le asigno un nombre al indice con index.name

data_set = pd.DataFrame(data = [serieA, serieB])
data_set.index.name = "Nombre de series"

print(data_set)
print (40*"--")

# 7º Genera un Dataframe con esta sintaxis "serieC = pd.Series(np.random.randint(1, 10, 35))"
    # Primero generamos el Dataframe con numeros aletorios
    # Luego creamos otro que tenga forma (7,5), haciendo uso de .values y .reshape

serieC = pd.Series(np.random.randint(1, 10, 35))
nw_serieC = pd.DataFrame (serieC.values.reshape(7,5))
print(nw_serieC)
