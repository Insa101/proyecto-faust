#0º Importamos Numpy

import numpy as np

print(40*"--")
# 1º Crear un array de 4 dimensiones y comprobar que tiene 4 dimensiones.
    # Comprobamos el nº de dimensiones con .ndim
array = np.random.random((2,3,4,5))
print(f" El número de dimensiones del array creado son: {array.ndim} dimensiones")
print(40*"--")

# 2º Mostrar por pantalla las dimensiones del array y su contenido
    # Mostraremos las dimensiones con .shape
print(f" La forma del array es {array.shape}")
print(40*"--")

    # Mostraremos el array creado con un print
print(array)
print(40*"--")

# 3º Calcular la suma de los elementos en funcion de sus 2 ultimos ejes y mostrar el resultado
    # Generamos un array nuevo
    # Sus elementos seran la suma np.sum() de los 2 ultimos ejes (-2, -1) del array original 
    # El resultado tendra una forma (2,3)
array_nuevo = np.sum(array, axis = (-2, -1))
print(array_nuevo)