# 0º Importamos la libreria Pandas

import pandas as pd

# 1º Crea un Dataframe con el arcivho "“COVID_01-01-2021.csv" y realiza lo siguiente:
    # Calcularas el total de casos confirmados, fallecidos, recuperados y activos de Covid por Pais
        # Primero compruebo el nombre de las columnas con un print + .columns
        # Utilizare .groupby() para agrupar por pais y seleccionare la columna
        # Utilizare .sum() para hayar los totales de cada columna

Datos = pd.read_csv(r"M02_Python_Librerias_Big_Data\UX_Trabajo_Final\Python_AnalisisDatos\COVID_01-01-2021.csv")
dataset2 = pd.DataFrame (data = Datos)

print(f"\nLos nombres de las columnas son:")
print(f"\n{dataset2.columns}")
print(40*"--")


totales = dataset2.groupby("Country_Region")[["Confirmed",'Deaths', 'Recovered', 'Active']].sum()

print(f"\nLos totales para nº de casos, fallecidos, recuperados y activos por país es:")
print(f"\n{totales}")