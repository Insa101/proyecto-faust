# 0º Importamos Numpy
# (Haremos el ejercicio con un maximo de 5 en el aleatorio, ya que no permite visualizar bien el resultado)

import numpy as np

## 1º Ej: Crear una variable que contenga cantidad, longitud y valores aleatorios, siguiendo estos pasos:
# Haremos una lista para guardar los valores de los vectores en numpy arrays aleatorios
mis_vectores = []

# Generaremos un nº de vectores aletorios con .random.randint entre 1 y 5.
n_vectores = np.random.randint(1, 5)

# Hacemos un bucle for para crear los vectores n numero de veces como n_vectores haya:
    # Definimos una longitud aleatoria entre 2 (minimo) y 5.
    # Creamos los n vectores con valores aleatorios del 0 al 6 y con longitud aleatoria.
    # Los añadimos a medida que se crean en nuestra lista
        # Los cambiamos a una lista de numeros int, con .tolist() para mejorar la visualizacion

for i in range(n_vectores):    
    longitud_aleatoria = np.random.randint(2, 5)
    vector = np.random.randint(0, 6, size=longitud_aleatoria)
    mis_vectores.append(vector.tolist())

## 2º Ej: Definir una funcion "producto_cartesiano" que realice lo siguiente:
    # Obtenga el conjunto de vectores creados
    # Realizar el producto cartesiano del conjunto y muestre por pantalla el resultado

# Definimos la funcion
def producto_cartesiano(vectores):

# Creamos una lista para recoger las listas con todas las combinaciones posibles
    combi_cartesiano = [[]]

# Hacemos un bucle for para pasar por cada vector para las combinaciones:
# Generamos otra lista temporal para guardar las combinaciones "intermedias"
    for n_vector in vectores:
        combi_temporal = []

# Hacemos un doble bucle for para:
        # Recuperar las combinaciones anteriores de combi_cartesiano
        # Utilizar un vector distinto al original n_vectores
        # Generamos una combinacion entre la anterior guardada y el vector nuevo:
        # Añadimos la lista temporal de combinaciones
        # Actualizamos la lista con combinaciones finales con la nueva combinacion temporal
        for c in combi_cartesiano:
            for n in n_vector:
                nueva_combi = c + [n]
                combi_temporal.append(nueva_combi)
        combi_cartesiano = combi_temporal
    
# Mostrar el resultado, utilizando un bucle for para mostrar todas las combianciones.
    print(f"\n El Producto Cartesiano de mis {len(vectores)} vectores es en total: {len(combi_cartesiano)} combinaciones")
    print(f"\n A continuacion se indicaran todas las combinaciones:")
    for combi in combi_cartesiano:
        print(f"\n {combi}")

## 3º Ej: Invocamos a la funcion "producto_cartesiano"

producto_cartesiano (mis_vectores)