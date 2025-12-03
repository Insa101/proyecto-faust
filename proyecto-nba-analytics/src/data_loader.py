# -*- coding: utf-8 -*-
import requests
def obtener_partidos(temporada):
    print (f" -- Inicializando busqueda para la temporada {temporada}----")
    datos_simulados = [
        {"fecha": "2023/10/24",
        "local":"lakers",
        "visitante": "Nuggets",
        "puntos_local": 107,
        "puntos_visitante": 119
        },
        {"fecha": "2023/10/24",
         "local": "Suns", 
         "visitante": "Warriors", 
         "puntos_local": 108, 
         "puntos_visitante": 104
         }
        ]
    return datos_simulados
if __name__ == "__main__":
    resultado = obtener_partidos(2023)
    print("Datos recuperados:")
    print(resultado)
    print("¡Prueba de infraestructura exitosa!")

