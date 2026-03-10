# Limpieza absoluta del entorno
rm(list = ls())

# Carga de librerías necesarias
library(dplyr)
library(ggplot2)
library(tidyr)
library(knitr)

# Configuración del directorio y carga del archivo
setwd("C:/proyecto-faust/M04_R/Documentos & Ejercicios temario/U2_Introduccion_R_&_R_Studio")

# Usamos un nombre de objeto único para evitar colisiones
df_apuestas <- read.csv("apuestas.csv", sep = ",") 

# PASO CRÍTICO: Muestra los nombres de las columnas
# Fíjate bien en la consola para confirmar si se llaman 'HomeTeam', 'FTHG', 'B365H', etc.
colnames(df_apuestas)

# 4. Cálculo de beneficio medio general
# ADVERTENCIA: He usado 'B365H', 'B365D' y 'B365A'. 
# Si en el colnames() viste algo como 'preciomd_eurMw' o similar, cámbialo aquí.
res_fijo <- df_apuestas %>%
  summarise(
    n_partidos = n(),
    b1 = mean(ifelse(FTHG > FTAG, (B365H - 1), -1), na.rm = TRUE),
    bx = mean(ifelse(FTHG == FTAG, (B365D - 1), -1), na.rm = TRUE),
    b2 = mean(ifelse(FTHG < FTAG, (B365A - 1), -1), na.rm = TRUE)
  )

# Mostramos la tabla formateada
kable(res_fijo, digits = 2)

# 5. Beneficio según equipo LOCAL
# ⚠️ ADVERTENCIA: Verifica si tu columna se llama 'HomeTeam'
res_por_local <- df_apuestas %>%
  group_by(HomeTeam) %>%
  summarise(
    Victoria = mean(ifelse(FTHG > FTAG, (B365H - 1), -1), na.rm = TRUE),
    Empate   = mean(ifelse(FTHG == FTAG, (B365D - 1), -1), na.rm = TRUE),
    Derrota  = mean(ifelse(FTHG < FTAG, (B365A - 1), -1), na.rm = TRUE)
  ) %>%
  pivot_longer(cols = -HomeTeam, names_to = "Tipo_Apuesta", values_to = "Beneficio")

ggplot(res_por_local, aes(x = HomeTeam, y = Beneficio, fill = Tipo_Apuesta)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  labs(title = "Rentabilidad por Equipo (Local)", y = "Beneficio Medio")

# 6. Beneficio según equipo VISITANTE (Barras APILADAS)
# ⚠️ ADVERTENCIA: Verifica si tu columna se llama 'AwayTeam'
res_por_visitante <- df_apuestas %>%
  group_by(AwayTeam) %>%
  summarise(
    Victoria = mean(ifelse(FTHG > FTAG, (B365H - 1), -1), na.rm = TRUE),
    Empate   = mean(ifelse(FTHG == FTAG, (B365D - 1), -1), na.rm = TRUE),
    Derrota  = mean(ifelse(FTHG < FTAG, (B365A - 1), -1), na.rm = TRUE)
  ) %>%
  pivot_longer(cols = -AwayTeam, names_to = "Tipo_Apuesta", values_to = "Beneficio")

ggplot(res_por_visitante, aes(x = AwayTeam, y = Beneficio, fill = Tipo_Apuesta)) +
  geom_bar(stat = "identity", position = "stack") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  labs(title = "Rentabilidad por Equipo (Visitante - Apilado)")

# 7. Combinamos datos de Local y Visitante
# ⚠️ ADVERTENCIA: Verifica si las cuotas son 'B365H' y 'B365A'
df_total_equipos <- bind_rows(
  df_apuestas %>% select(Equipo = HomeTeam, FTHG, FTAG, cuota = B365H) %>%
    mutate(ganado = FTHG > FTAG),
  df_apuestas %>% select(Equipo = AwayTeam, FTHG, FTAG, cuota = B365A) %>%
    mutate(ganado = FTHG < FTAG)
) %>%
  mutate(beneficio = ifelse(ganado, cuota - 1, -1))

res_global <- df_total_equipos %>%
  group_by(Equipo) %>%
  summarise(Beneficio_Medio = mean(beneficio, na.rm = TRUE))

ggplot(res_global, aes(x = Equipo, y = Beneficio_Medio, group = 1)) +
  geom_line(color = "darkgreen") +
  geom_point() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  labs(title = "Rentabilidad Total por Equipo (Líneas)")

# 8. Beneficio de apostar a +2.5 goles
# ⚠️ ADVERTENCIA: Mira en colnames() si el nombre es 'BbAv.2.5' o similar (ej: 'preciomd_eurMw')
res_over_goles <- df_apuestas %>%
  group_by(HomeTeam) %>%
  summarise(
    B_Over = mean(ifelse((FTHG + FTAG) > 2.5, (BbAv.2.5 - 1), -1), na.rm = TRUE)
  )

print(head(res_over_goles))