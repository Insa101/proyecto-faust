# 1. Limpieza y librerías
rm(list = ls())
library(tidyverse)
library(reshape2)
library(moments) # Para skewness y kurtosis
library(GGally)  # Para ggpairs

# 2. Carga de ficheros con nombres únicos
df_estaciones_info <- read.delim("C:/proyecto-faust/M04_R/Documentos & Ejercicios temario/U3_Estadística descriptiva y análisis exploratorio de datos/estaciones_meteo.txt", sep="\t")

# 🔍 INSPECCIÓN 1: Nombres de columnas de estaciones
colnames(df_estaciones_info) 

# Filtramos según el enunciado
df_estaciones_filtro <- df_estaciones_info %>% 
  filter(estacion %in% c("MADRID", "BARCELONA", "SEVILLA", "ZARAGOZA", "BILBAO"))

df_meteo_bruto <- read.delim("C:/proyecto-faust/M04_R/Documentos & Ejercicios temario/U3_Estadística descriptiva y análisis exploratorio de datos/meteo_data.csv", sep=";", stringsAsFactors = FALSE)

# 🔍 INSPECCIÓN 2: Nombres de columnas meteorológicas
# ⚠️ ADVERTENCIA: Fíjate si 'Date' o 'Mean_TemperatureC' aparecen escritos así exactamente.
colnames(df_meteo_bruto) 

# 3. Transformación y filtrado final
df_meteo_final <- df_meteo_bruto %>% 
  mutate(Date = as.Date(Date)) %>% 
  filter(estacion %in% df_estaciones_filtro$estacion)

# Resumen estadístico inicial [cite: 80]
summary(df_meteo_final)

# Histogramas para Zaragoza
# ⚠️ ADVERTENCIA: Confirma que la columna es 'Mean_TemperatureC' y la estación 'ZARAGOZA'
ggplot(df_meteo_final %>% filter(estacion == "ZARAGOZA"), aes(x = Mean_TemperatureC)) +
  geom_histogram(binwidth = 1, fill = "orange", color = "white") +
  labs(title = "Distribución de Temperatura Media en Zaragoza")

# Comparativa entre estaciones (Polígonos y Densidades) [cite: 92]
# Usamos position="identity" para que no se apilen y se vea la diferencia real [cite: 89]
ggplot(df_meteo_final, aes(x = Mean_TemperatureC, color = estacion)) +
  geom_freqpoly(binwidth = 1, position = "identity") +
  labs(title = "Polígonos de Frecuencia por Estación")

ggplot(df_meteo_final, aes(x = Mean_TemperatureC, fill = estacion)) +
  geom_density(alpha = 0.3) +
  labs(title = "Densidades de Temperatura Media")

# Densidad de Precipitación [cite: 93]
# ⚠️ ADVERTENCIA: Verifica si la columna es 'Precipitationmm'
ggplot(df_meteo_final, aes(x = Precipitationmm, fill = estacion)) +
  geom_density(alpha = 0.4) +
  xlim(0, 10) # Acotamos para ver mejor el detalle cerca de 0

# 1. Selección de variables numéricas clave [cite: 97]
# ⚠️ ADVERTENCIA: Asegúrate de que estos nombres coinciden con el colnames() anterior
vars_clave <- c("Mean_TemperatureC", "Mean_Wind_SpeedKm_h", "WindDirDegrees", 
                "Precipitationmm", "CloudCover", "Mean_Humidity")

df_meteo_long <- melt(df_meteo_final, 
                      id.vars = c("estacion", "Date"), 
                      measure.vars = vars_clave)

# 2. Visualización con facets [cite: 96]
ggplot(df_meteo_long, aes(x = value, fill = estacion)) +
  geom_density(alpha = 0.3) +
  facet_wrap(~variable, scales = "free") +
  labs(title = "Distribuciones por Variable y Estación")

# Boxplot de Temperaturas Máximas por Estación
ggplot(df_meteo_final, aes(x = estacion, y = Max_TemperatureC, fill = estacion)) +
  geom_boxplot() +
  labs(title = "Distribución de Temperaturas Máximas")

# Análisis de lluvia: ¿Quién se moja más? [cite: 106]
# ⚠️ ADVERTENCIA: Verifica si las columnas lógicas son 'Rain' y 'Fog'
resumen_eventos <- df_meteo_final %>%
  group_by(estacion) %>%
  summarise(
    Dias_Lluvia_Evento = sum(Rain == TRUE, na.rm = TRUE),
    Dias_Precip_Real   = sum(Precipitationmm > 0, na.rm = TRUE),
    Dias_Niebla        = sum(Fog == TRUE, na.rm = TRUE)
  )

print(resumen_eventos)

# Estadísticos en un solo paso (Usando dplyr moderno) [cite: 115]
# ⚠️ ADVERTENCIA: Si usas una versión muy antigua de R, usa 'summarise_each'
res_stats_posicion <- df_meteo_final %>%
  group_by(estacion) %>%
  summarise(across(all_of(vars_clave), 
                   list(media = ~mean(.x, na.rm = TRUE), 
                        mediana = ~median(.x, na.rm = TRUE),
                        truncada = ~mean(.x, trim = 0.1, na.rm = TRUE))))

# 🔍 INSPECCIÓN 3: Nombres de las nuevas columnas de estadísticas
colnames(res_stats_posicion)

# Filtrar solo medias de temperatura usando matches [cite: 116, 117]
solo_medias_temp <- res_stats_posicion %>% 
  select(estacion, matches("TemperatureC_media"))

print(solo_medias_temp)

# Filtramos Barcelona y variables numéricas
df_barcelona <- df_meteo_final %>% 
  filter(estacion == "BARCELONA") %>%
  select(all_of(vars_clave), Max_TemperatureC, Min_TemperatureC)

# Matriz de correlación gráfica [cite: 147]
ggpairs(df_barcelona) + labs(title = "Correlaciones en Barcelona")