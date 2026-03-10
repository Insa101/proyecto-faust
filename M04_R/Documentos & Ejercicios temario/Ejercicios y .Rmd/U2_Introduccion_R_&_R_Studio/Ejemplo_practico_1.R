# Datos iniciales: 13 lecturas (desde 31-dic-2015 hasta 31-dic-2016)
lecturas <- c(3007, 3292, 3568, 3783, 3979, 4169, 4351, 4565, 4749, 5001, 5219, 5438, 5685) 

# Cálculo del consumo mensual mediante la función diff()
consumo_mensual <- diff(lecturas)

# 1.1 Total de energía consumida
total_energia <- sum(consumo_mensual)

# 1.2 Consumo por semestres (6 meses cada uno)
consumo_s1 <- sum(consumo_mensual[1:6])
consumo_s2 <- sum(consumo_mensual[7:12])

# 1.3 Consumo por trimestres (3 meses cada uno)
consumo_t1 <- sum(consumo_mensual[1:3])
consumo_t2 <- sum(consumo_mensual[4:6])
consumo_t3 <- sum(consumo_mensual[7:9])
consumo_t4 <- sum(consumo_mensual[10:12])

# 2. Configuración de precio fijo
precio_anual <- 0.1236 # 

# Importe mensual y total del término de energía
importe_mensual_fijo <- consumo_mensual * precio_anual # 
importe_total_fijo <- sum(importe_mensual_fijo) #

# 3. Precios mensuales PVPC 2016 (Euros/kwh)
pvpc_mes <- c(0.1034, 0.0923, 0.0917, 0.0867, 0.0878, 0.1005, 
              0.1026, 0.1040, 0.1061, 0.1185, 0.1226, 0.1302) # [cite: 17]

# Coste mensual y anual con PVPC
coste_pvpc_mensual <- consumo_mensual * pvpc_mes # [cite: 18]
coste_pvpc_anual <- sum(coste_pvpc_mensual) # [cite: 18]

# Coste medio por trimestres
coste_medio_trim <- c(mean(coste_pvpc_mensual[1:3]), mean(coste_pvpc_mensual[4:6]),
                      mean(coste_pvpc_mensual[7:9]), mean(coste_pvpc_mensual[10:12])) # [cite: 19]

# Ahorro con PVPC frente a precio fijo
ahorro_pvpc <- importe_total_fijo - coste_pvpc_anual # [cite: 20]

# Precio de equilibrio para que la tarifa fija iguale a la PVPC
precio_fijo_equilibrio <- coste_pvpc_anual / total_energia # [cite: 21]

# 4. Parámetros de facturación completa
pot_cliente <- 4.6 # [cite: 33]
cst_potkw_anual <- 38.04343 # [cite: 34]
dias_mes <- c(31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31) # 
tasa_impuesto <- 5.1127 / 100 # [cite: 41]
precio_alquiler_dia <- 0.026551 # [cite: 41]
iva_tasa <- 0.21 # [cite: 41]

# Cálculo del término de potencia mensual (proporcional a los días de 2016, que fue bisiesto)
coste_potencia_mensual <- (pot_cliente * cst_potkw_anual / 366) * dias_mes # 

# Alquiler de equipos mensual
alquiler_mensual <- precio_alquiler_dia * dias_mes # [cite: 37]

# Función para aplicar la estructura de costes completa
calcular_factura <- function(energia_mes) {
  # Base del impuesto: Energía + Potencia
  base_impuesto <- energia_mes + coste_potencia_mensual # 
  impuesto_elec <- base_impuesto * tasa_impuesto # 
  
  # Subtotal antes de IVA: Base + Impuesto + Alquiler
  subtotal <- base_impuesto + impuesto_elec + alquiler_mensual # 
  
  # Total con IVA
  total_factura <- subtotal * (1 + iva_tasa) # 
  return(total_factura)
}

# Ejecución de los dos escenarios
facturas_fijo_final <- calcular_factura(importe_mensual_fijo) # 
facturas_pvpc_final <- calcular_factura(coste_pvpc_mensual) # 

# Ahorro anual final
ahorro_anual_factura <- sum(facturas_fijo_final) - sum(facturas_pvpc_final) # [cite: 40]