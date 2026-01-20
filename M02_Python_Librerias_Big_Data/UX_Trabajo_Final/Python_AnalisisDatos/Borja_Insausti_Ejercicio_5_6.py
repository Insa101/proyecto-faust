# 0º Importamos la libreria Pandas, Seaborn y Pyplot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1º Crearemos un Dataframe para cada archivo de Enero, Febrero y Marzo
    # Crear un grafico que indique:
        # 1 - Permita visualizar la evolucion de los confirmados, recuperados y fallecidos 
        # 2 - Eje X = casos
        # 3 - Eje Y = los meses
        # 4 - Diferenciar entre tipos de casos
        # 5 - Ponerle el titulo: “Evolución COVID primer trimestre 2021”
Datos_enero = pd.read_csv(r"M02_Python_Librerias_Big_Data\UX_Trabajo_Final\Python_AnalisisDatos\COVID_01-01-2021.csv")
Datos_febrero = pd.read_csv(r"M02_Python_Librerias_Big_Data\UX_Trabajo_Final\Python_AnalisisDatos\COVID_01-02-2021.csv")
Datos_marzo= pd.read_csv(r"M02_Python_Librerias_Big_Data\UX_Trabajo_Final\Python_AnalisisDatos\COVID_01-03-2021.csv")
dataset_enero = pd.DataFrame (data = Datos_enero)
dataset_febrero = pd.DataFrame (data = Datos_febrero)
dataset_marzo = pd.DataFrame (data = Datos_marzo)

# Primero comprobaremos con un print + .columns el nombre de las columnas de un dataframe
print(f"\nLos nombres de las columnas son:")
print(f"\n{dataset_enero.columns}")
print(40*"--")

# Creamos columna "Mes" para cada dataframe con su mes correspondiente
dataset_enero["Mes"] = "Enero"
dataset_febrero["Mes"] = "Febrero"
dataset_marzo["Mes"] = "Marzo"

# Crearemos un DataFrame combinando los otros 3 dataframes mediante .concat()
dataset_trimestral = pd.concat([dataset_enero, dataset_febrero, dataset_marzo])

# Agrupamos por "Mes" .groupby() y obtenemos los totales con .sum()
    # Resetearemos el Indice para "recuperar" a "Mes" como columna
dataset_agrupado = dataset_trimestral.groupby("Mes")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()


# Generamos en la columna Mes, el tipo de dato para diferenciar los casos
dataset_grafico = dataset_agrupado.melt(
    id_vars = "Mes", 
    var_name = "Tipo_Caso", 
    value_name = "Casos_Totales"
)

# Generamos un grafico de barras para poder ver la evolucion cumpliendo:
    # Eje X = casos
    # Eje Y = los meses
    # Diferenciar entre tipos de casos
    # Ponerle el titulo: “Evolución COVID primer trimestre 2021”
sns.barplot(data = dataset_grafico, x = "Casos_Totales", y = "Mes", hue ="Tipo_Caso", orient = "h")
plt.title("Evolución COVID primer trimestre 2021", fontsize=16)
plt.ylabel("Meses", fontsize=12)
plt.xlabel("Totales de Casos (Decenas de millones)", fontsize=12)
plt.show()
