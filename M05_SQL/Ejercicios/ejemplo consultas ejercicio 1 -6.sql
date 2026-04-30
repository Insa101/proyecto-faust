USE equiposbbdd;

# Consulta 1 - Muestra el nombre y la region de los paises
SELECT nombre, region FROM PAISES;

# Consulta 2 - Muestra la cantidad total de paises
SELECT COUNT(*) AS 'Totales' FROM paises; 

# Consulta 3 - Muestra la fecha de inicio de la primera y de la ultima temporada
SELECT MAX(fecha_inicio_fin) AS 'Ultima temporada', 
MIN(fecha_inicio_fin) AS 'Primera temporada' 
FROM temporadas;

# Consulta 4 - Muestra las 5 primeras filas de la temporada
SELECT * FROM temporadas LIMIT 5;

#Consulta 5 - Mostrar equipos que empiecen por "Real" y terminen por "a"
SELECT * FROM equipos WHERE nombre LIKE '%Real%' AND nombre LIKE '%a';

#Consulta 6 - Mostrar paises que tengan la "e", mayus o minus
SELECT * FROM paises WHERE nombre LIKE '%E%' AND nombre LIKE '%e%';