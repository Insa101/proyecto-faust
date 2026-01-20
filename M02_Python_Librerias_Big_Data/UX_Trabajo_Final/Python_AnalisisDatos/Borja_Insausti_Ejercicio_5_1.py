# 0º Importamos la libreria Pandas

import pandas as pd

# 1º Crea un Dataframe con el arcivho "“COVID_01-01-2021.csv" y realiza lo siguiente:
    # Mostraras la informacion del dataframe
    # Calcularas y mostraras la cantidad de NaN's del data frame por columna
        # Usare .isna () para localizar los Nulos
        # Usare .sum() para sumarlos.
    # Mostraras las 5 primeras filas del dataframe

Datos = pd.read_csv(r"M02_Python_Librerias_Big_Data\UX_Trabajo_Final\Python_AnalisisDatos\COVID_01-01-2021.csv")
dataset1 = pd.DataFrame (data = Datos)

print(f"\nLa informacion del archivo “COVID_01-01-2021.csv” es la siguiente:")
print(f"\n{dataset1}")
print(40*"--")

print(f"\nLos NaN's del archivo “COVID_01-01-2021.csv” se veran en la siguiente tabla:")
print(f"\n{dataset1.isna().sum()}")
print(40*"--")

print(f"\nEstos son los encabezados de las 5 primeras filas de “COVID_01-01-2021.csv”:")
print(f"\n{dataset1.head()}")
print(40*"--")