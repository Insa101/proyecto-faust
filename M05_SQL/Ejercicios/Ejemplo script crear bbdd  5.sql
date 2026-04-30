USE nombrebbdd;

SELECT 
autores.nombre AS "El pibe", 
CASE
		WHEN autores.fecha_nacimiento IS NULL THEN " Ni puto caso"
		WHEN autores.fDeceso IS NOT NULL THEN CONCAT_WS ( " - ", 
		TIMESTAMPDIFF (YEAR, autores.fecha_nacimiento, autores.fDeceso), "Fallecido")
		ELSE TIMESTAMPDIFF (YEAR, autores.fecha_nacimiento, CURRENT_DATE())
	END AS "Edad"
FROM autores;