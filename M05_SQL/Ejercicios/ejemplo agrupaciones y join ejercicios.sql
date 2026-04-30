USE equiposbbdd;

# Consulta 1 - mostrar nombre de los paises y cantidad de equipos que hay en cada pais
SELECT 
	paises.nombre AS 'Nombre Pais',
	COUNT(equipos.equipo_id) AS 'Nº equipos'
FROM paises
LEFT JOIN equipos ON (paises.pais_id = equipos.pais_id)
GROUP BY paises.nombre;

# Consulta 2 - mostrar nombre de los paises y cantidad de equipos que hay en cada pais SI HAY AL MENOS 5
SELECT 
	paises.nombre AS 'Nombre Pais',
	COUNT(equipos.equipo_id) AS 'Nº equipos'
FROM paises, equipos
WHERE (paises.pais_id = equipos.pais_id)
GROUP BY paises.nombre
HAVING COUNT(equipos.equipo_id)>5
ORDER BY paises.nombre ASC;

#Consulta 3 - Lo mismos de antes pero con equipos que empiecen por E y mostrar en orden de cantidad de equipos descendente
SELECT 
	paises.nombre AS 'Nombre Pais',
	COUNT(equipos.equipo_id) AS 'Nº equipos'
FROM paises, equipos
WHERE (paises.pais_id = equipos.pais_id) AND (paises.nombre LIKE 'E%')
GROUP BY paises.nombre
ORDER BY COUNT(equipos.equipo_id) DESC;

# CONSULTA 4 - 3 EQUIPOS CON MAS CAMPEONATOS

SELECT campeonatos.campeon AS 'equipos campeones', COUNT(campeonatos.campeon) AS 'nº de campeonatos'
FROM campeonatos
GROUP BY campeonatos.campeon
ORDER BY COUNT(campeonatos.campeon) DESC
LIMIT 3;

#CONSULTA 5 - MOSTRAR COMPETICIONES Y Nº VECES CADA UNA

SELECT 
	competiciones.nombre AS 'Competiciones',
    COUNT(campeonatos.competicion_id) AS 'Nº de ediciones'
FROM competiciones
JOIN campeonatos ON (competiciones.competicion_id = campeonatos.competicion_id)
GROUP BY competiciones.nombre
ORDER BY COUNT(campeonatos.competicion_id) DESC;

#CONSULTA 6 - NOMBRRE COMPETICIONES, Nº VECES CAMPEON Y Nº VECES SUBCAMPEON BARSA

SELECT
	competiciones.nombre AS 'Competicion',
    SUM(campeonatos.campeon LIKE '%Fútbol Club Barcelona%') AS 'Nº Campeones',
    SUM(campeonatos.subcampeon LIKE '%Fútbol Club Barcelona%') AS 'Nº Subcampeon'
FROM campeonatos
JOIN competiciones ON (competiciones.competicion_id = campeonatos.competicion_id)
GROUP BY competiciones.nombre
ORDER BY competiciones.nombre ASC;

