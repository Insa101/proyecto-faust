# 0º Importamos la libreria Pandas

import pandas as pd

# 1º Crea un Dataframe con el arcivho "“COVID_01-01-2021.csv" y realiza lo siguiente:
    # Obtener la informacion de pais, casos confirmados, fallecidos y recuperados de los 10 paises con mas confirmados
Datos = pd.read_csv(r"M02_Python_Librerias_Big_Data\UX_Trabajo_Final\Python_AnalisisDatos\COVID_01-01-2021.csv")
dataset4 = pd.DataFrame (data = Datos)

# Primero comprobaremos con un print + .columns el nombre de las columnas
print(f"\nLos nombres de las columnas son:")
print(f"\n{dataset4.columns}")
print(40*"--")


# Agruparemos por paises con .groupby + .sum() para sumar todos los valores de confirmados
# Mediante .sort_values + ascending =False + .head(10) elegimos los 10 paises en una variable
agrupacion = dataset4.groupby("Country_Region")["Confirmed"].sum()                                             
top_10 = agrupacion.sort_values(ascending=False).head(10)

# Hacemos una mascara booleana para filtrar estos 10 paises, usando .isin() y eligiendo el indice.
mascara = dataset4["Country_Region"].isin(top_10.index)
dataset_top_10 = dataset4[mascara]

# Seleccionaremos las columnas de Pais, confirmados, fallecidos y recuperados
seleccion_final = dataset_top_10[["Country_Region", 'Confirmed', 'Deaths', 'Recovered']]

print(f"\nLos 10 paises con mas casos confirmados son US, India, Brazil, Russia, France, UK, Turkey, Italy, Spain, Germany")
print(f"\nEn el siguiente cuadro se ve el desglose de los datos de estos paises")
print(f"\n{seleccion_final}")
print(40*"--")