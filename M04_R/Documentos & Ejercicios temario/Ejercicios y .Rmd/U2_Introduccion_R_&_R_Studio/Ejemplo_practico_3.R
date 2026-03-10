# 1. Limpieza total del entorno
rm(list = ls())

# 2. Configuración de directorio y carga
setwd("C:/proyecto-faust/M04_R/Documentos & Ejercicios temario/U2_Introduccion_R_&_R_Studio")

# Llamamos a la tabla 'datos_luz' para que no choque con la columna 'consumo_kwh'
datos_luz <- read.delim("consumo_horario_cliente.csv", 
                        sep = ";", 
                        colClasses = c("POSIXct", "numeric", "numeric"))

# 3. Creación de variables (Asignación directa para mayor seguridad)
datos_luz$coste_eur <- datos_luz$consumo_kwh * datos_luz$precio_kwh
datos_luz$Fecha     <- as.Date(datos_luz$datetime)
datos_luz$Hora      <- format(datos_luz$datetime, "%H")
datos_luz$Mes_num   <- format(datos_luz$datetime, "%m")
datos_luz$wd        <- weekdays(datos_luz$datetime)

# 4. Filtrado del 31 de diciembre
consumo_fin_anio <- datos_luz[datos_luz$Fecha == "2016-12-31", ]

# 5. Búsqueda de máximos y mínimos
fila_max_consumo <- datos_luz[which.max(datos_luz$consumo_kwh), ]
fila_max_coste   <- datos_luz[which.max(datos_luz$coste_eur), ]
min_consumo      <- min(datos_luz$consumo_kwh)
consumos_bajos   <- datos_luz[datos_luz$consumo_kwh < 0.02, ]

# 6. Cálculo de medias (Corregido para evitar el error de las 840 filas)
# Aquí usamos 'consumo_kwh' que es la columna, NO 'datos_luz' que es la tabla
medias_M_V <- aggregate(cbind(consumo_kwh, coste_eur) ~ wd, 
                        data = datos_luz[datos_luz$wd %in% c("martes", "viernes"), ], 
                        FUN = mean)

# Convertimos a factor con el orden cronológico y etiquetas de una sola letra
datos_luz$wd <- factor(datos_luz$wd, 
                       levels = c("lunes", "martes", "miércoles", "jueves", 
                                  "viernes", "sábado", "domingo"),
                       labels = c("L", "M", "X", "J", "V", "S", "D"))

# Verificamos el cambio
table(datos_luz$wd)

# Sumamos el consumo y el coste agrupando por el número de mes
resumen_mensual <- aggregate(cbind(consumo_kwh, coste_eur) ~ Mes_num, 
                             data = datos_luz, 
                             FUN = sum)

# Identificamos el mes de mayor consumo y el de mayor coste
mes_max_consumo <- resumen_mensual[which.max(resumen_mensual$consumo_kwh), ]
mes_max_coste   <- resumen_mensual[which.max(resumen_mensual$coste_eur), ]

# Imprimimos resultados para comparar
print(resumen_mensual)

# 1. Cargamos los datos del mercado mayorista
# Usamos read.delim asumiendo que el separador es ";" como en el anterior [cite: 127]
precio_md <- read.delim("precio_md.csv", sep = ";")

# 2. TRUCO DE SEGURIDAD: Convertimos la fecha de la nueva tabla a POSIXct
# Sin este paso, el merge suele devolver 0 filas [cite: 97, 99]
precio_md$datetime <- as.POSIXct(precio_md$datetime)

# 3. Realizamos la unión
# Ahora que ambas columnas 'datetime' son POSIXct, el cruce será efectivo [cite: 126, 127]
datos_comparativos <- merge(datos_luz, precio_md, by = "datetime")

# Ejecuta esto primero para ver los nombres reales en la consola
names(datos_comparativos)

# 4. Verificación rápida: Si esto devuelve 0, hay un problema en el archivo CSV
print(paste("Filas encontradas tras la unión:", nrow(datos_comparativos)))

# 5. Calculamos la diferencia
# Usamos los nombres exactos: precio minorista vs mayorista 
datos_comparativos$dif_precio <- datos_comparativos$precio_kwh - datos_comparativos$preciomd_eurMw

# 6. Ahora el aggregate ya tendrá filas con las que trabajar 
diff_media_mensual <- aggregate(dif_precio ~ Mes_num, 
                                data = datos_comparativos, 
                                FUN = mean)

print(diff_media_mensual)

# 1. ¿Es el mes de mayor consumo el de mayor coste?
# Buscamos la fila con el máximo de cada columna en el resumen mensual
mes_top_consumo <- resumen_mensual$Mes_num[which.max(resumen_mensual$consumo_kwh)]
mes_top_coste <- resumen_mensual$Mes_num[which.max(resumen_mensual$coste_eur)]

print(paste("Mes con mayor consumo:", mes_top_consumo))
print(paste("Mes con mayor coste:", mes_top_coste))

if(mes_top_consumo == mes_top_coste) {
  print("Sí, el mes de mayor consumo coincide con el de mayor coste.")
} else {
  print("No coinciden. Esto se debe a la variación de los precios mensuales.")
}

# 2. Ver las diferencias medias mensuales (lo que ya calculamos)
# Esto muestra cuánto más pagó el cliente de media cada mes respecto al mercado
print("Margen medio mensual (Minorista - Mayorista):")
print(diff_media_mensual)
