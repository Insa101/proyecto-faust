# 0º Importamos la libreria Pandas, Seaborn y Pyplot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1º Crearemos un Dataframe para cada archivo de Enero, Febrero y Marzo
    # Representaremos la evolucion trimestral de los confirmados y recuperados en las provincias españolas

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
dataset_trimestre = pd.concat([dataset_enero, dataset_febrero, dataset_marzo])

# Seleccionare las observaciones de España, con una mascara booleana
dataset_filtrado= dataset_trimestre[(dataset_trimestre["Country_Region"] == "Spain")]

# Genero un dataframe para quedarme con la informacion que me interesa
dataset_spain = dataset_filtrado[['Province_State','Mes','Confirmed', 'Recovered']]

# Genero un dataframe con los sumatorios y tipos de caso con .melt()
dataset_final = dataset_spain.melt(
    id_vars=['Province_State', 'Mes'], 
    value_vars=['Confirmed', 'Recovered'],
    var_name='Tipo_Caso',
    value_name='Cantidad'
)

# Haremos un grafico por provincia para ver su evolucion
    # Crearemos un bucle for para iterar la creacion del grafico
        # Usare la funcion .unique para que no me repita las provincias
    # Seleccionare los datos de la provincia con una mascara booleana
        # Incluire un condicional para eliminar "Unknown" de la iteracion.
    # Crearemos graficos de barras con titulos y etiquetas para el eje y.
for provincia in dataset_final["Province_State"].unique():

    dataset_grafico = dataset_final[dataset_final["Province_State"] == provincia]
    if provincia == "Unknown":
        continue

    sns.barplot(data = dataset_grafico, y = "Cantidad" , x = "Mes", hue = "Tipo_Caso")
    plt.title ( f"Evolución COVID 1º trimestre 2021 provincia de {provincia}", fontsize = 16 )
    plt.ylabel (" Casos Totales " , fontsize = 16)
    plt.show()