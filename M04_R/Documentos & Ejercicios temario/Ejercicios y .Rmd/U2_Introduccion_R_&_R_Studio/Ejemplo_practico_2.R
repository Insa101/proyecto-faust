calc_pi <- function(n) {
  # Generamos n puntos aleatorios para x e y entre 0 y 1
  x <- runif(n)
  y <- runif(n)
  
  # Verificamos si el punto cae dentro del círculo (x^2 + y^2 < 1)
  # Usamos sum() directamente sobre la condición lógica para contar los aciertos
  puntos_dentro <- sum(x^2 + y^2 < 1)
  
  # Calculamos el área (proporción) y la multiplicamos por 4 para estimar pi
  area_estimada <- puntos_dentro / n
  pi_estimado <- area_estimada * 4
  
  return(pi_estimado)
}

# Definimos los tamaños de muestra
n_valores <- c(10, 100, 1000, 10000, 100000)

# Aplicamos la función a cada valor de n
estimaciones_sapply <- sapply(n_valores, calc_pi)

# Mostrar resultados
names(estimaciones_sapply) <- n_valores
print(estimaciones_sapply)

# Generamos 100 repeticiones para cada n
# sapply nos devolverá una matriz donde cada n es una columna, por lo que usamos t() para transponerla
matriz_pi <- t(sapply(n_valores, function(n) replicate(100, calc_pi(n))))

# Asignamos nombres a las filas para identificar el tamaño de n
rownames(matriz_pi) <- paste("N =", n_valores)

# Media de cada fila (cada grupo de n)
medias_filas <- apply(matriz_pi, 1, mean)

# Desviación típica de cada fila
sd_filas <- apply(matriz_pi, 1, sd)

# Combinamos los resultados para visualizarlos mejor
resumen_estadistico <- data.frame(Media = medias_filas, Desviacion_SD = sd_filas)
print(resumen_estadistico)

# Dibujamos el boxplot transponiendo de nuevo para que los grupos estén en el eje X
boxplot(t(matriz_pi), 
        main = "Convergencia de Montecarlo para estimar PI",
        xlab = "Número de puntos (N)", 
        ylab = "Valor estimado de PI",
        col = "lightblue",
        las = 1)

# Añadimos una línea roja en el valor real de pi para comparar
abline(h = pi, col = "red", lwd = 2, lty = 2)