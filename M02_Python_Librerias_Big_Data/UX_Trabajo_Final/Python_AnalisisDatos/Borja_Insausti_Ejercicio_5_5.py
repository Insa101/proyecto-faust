# 0º Importamos la libreria Pandas, Seaborn y Pyplot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1º Crea un Dataframe con el arcivho "“COVID_01-01-2021.csv" y realiza lo siguiente:
    # Crear un grafico de barras que muestre el total de confirmados, fallecidos y recuperados
        # Unicamente de paises con fallecidos <150
Datos = pd.read_csv(r"M02_Python_Librerias_Big_Data\UX_Trabajo_Final\Python_AnalisisDatos\COVID_01-01-2021.csv")
dataset5 = pd.DataFrame (data = Datos)

# Primero comprobaremos con un print + .columns el nombre de las columnas
print(f"\nLos nombres de las columnas son:")
print(f"\n{dataset5.columns}")
print(40*"--")

# Agrupamos los paises con .groupby y calculamos el total de confirmados fallecidos y recuperados
totales = dataset5.groupby("Country_Region")[['Confirmed', 'Deaths', 'Recovered']].sum()

# Ahora realizamos un filtrado para escoger unicamente paises con menos de 150 fallecidos
    # Haremos una mascara booleana para realizar el filtrado y se lo aplicaremos al grupo
totales_filtrado = totales[(totales["Deaths"]<150)]

# Ahora "recuperamos" el pais como columna y asi poder utilizarlo en el grafico
    # Usaremos .reset_index() + .melt() para configurar la nueva columna del pais

final_df = totales_filtrado.reset_index().melt(
    id_vars = "Country_Region", 
    var_name = "Tipo", 
    value_name = "Cantidad"
)

# Realizamos el grafico de barras con sns.barplot() para el dataframe filtrado
    # Al ser muchos datos, cambiaremos esteticamente:
        # Haremos que la orientacion sea horizontal en vez de vertical
        # Homogeanizo datos con la escala logaritmica (pero disocia la realidad de los datos)
        # Rotaremos y añadiremos etiquetas con plt.xticks
sns.barplot(data = final_df, y = "Country_Region", x ="Cantidad", hue ="Tipo", orient = "h")
plt.xscale("log")
plt.yticks( fontsize = 4)
plt.show()