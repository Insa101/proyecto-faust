USE equiposbbdd;

#Consulta 7 - Mostrar en 1 sola consulta:
# Cantidad de campeonatos que se han realizado, equipos que han sido campeones, cantidad de equipos diferentes
SELECT COUNT(campeonatos_id) AS 'Nº campeonatos' FROM campeonatos;

# cantidad de campeonatos ganados por campeon
SELECT campeon AS equipos, COUNT(*) AS 'Total campeonatos' 
FROM campeonatos GROUP by campeon ORDER BY 'TOTAL TITULOS' DESC;

#cantidad de campeones distintos
SELECT DISTINCT campeon AS equipos FROM campeonatos;

# cantidad de equipos subcampeones
SELECT subcampeon AS equipos, COUNT(*) AS 'Total subcampeonatos' 
FROM campeonatos GROUP by subcampeon ORDER BY 'TOTAL TITULOS' DESC;

#cantidad de campeones distintos
SELECT DISTINCT subcampeon AS equipos FROM campeonatos;

#cantidad de veces que no ha habido campeon
SELECT COUNT(*) AS "Sin campeon" FROM campeonatos WHERE campeon LIKE '%-%';

#cantidad de veces que no ha habido subcampeon
SELECT COUNT(*) AS "Sin subcampeon" FROM campeonatos WHERE subcampeon LIKE '%-%';