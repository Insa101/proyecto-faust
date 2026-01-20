# 0º Importamos la libreria Pandas

import pandas as pd

# 1º Crea un Dataframe con el arcivho "“COVID_01-01-2021.csv" y realiza lo siguiente:
    # Obtendras las provincias y el pais al que pertenecen, pero solo de aquellos sin recuperados
        # Primero comprobaremos con un print + .columns el nombre de las columnas
        # Realizaremos una mascara booleana para filtrar por aquellos que no esten recuperados
            # Eliminare del filtrado los NaN's con .dropna()
        # Seleccionaremos las columnas de provincias y paises

Datos = pd.read_csv(r"M02_Python_Librerias_Big_Data\UX_Trabajo_Final\Python_AnalisisDatos\COVID_01-01-2021.csv")
dataset3 = pd.DataFrame (data = Datos)

print(f"\nLos nombres de las columnas son:")
print(f"\n{dataset3.columns}")
print(40*"--")

provincias = dataset3[dataset3["Recovered"] == 0][["Province_State","Country_Region"]].dropna()

print(f"\nLas provincias de cada pais sin recuperados son:")
print(f"\n{provincias}")