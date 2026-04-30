# 1. Crear BBDD nueva
CREATE DATABASE IF NOT EXISTS equiposBBDD CHARACTER SET utf8mb4 COLLATE utf8mb4_spanish_ci;

USE equiposBBDD;

# 2. Crear las entidades con sus atributos
CREATE TABLE IF NOT EXISTS  PAISES (
pais_id INT NOT NULL AUTO_INCREMENT,
PRIMARY KEY (pais_id),
nombre VARCHAR (50) NOT NULL,
region ENUM ('Europa', 'Medio Oriente', 'Asia Oriental', 'Oceanía', 'Caribe', 'Norte America', 'Centro America', 'Sur America', 'Africa') NOT NULL
) ENGINE InnoDB;

CREATE TABLE IF NOT EXISTS  EQUIPOS (
equipo_id INT NOT NULL AUTO_INCREMENT,
PRIMARY KEY (equipo_id),
pais_id INT NOT NULL,
CONSTRAINT pais_id_fk
FOREIGN KEY (pais_id) REFERENCES PAISES (pais_id)
ON DELETE CASCADE
ON UPDATE CASCADE,
nombre VARCHAR (100) NOT NULL
) ENGINE InnoDB;

CREATE TABLE IF NOT EXISTS  TEMPORADAS (
temporada_id INT NOT NULL AUTO_INCREMENT,
PRIMARY KEY (temporada_id),
fecha_inicio_fin VARCHAR  (10) NOT NULL
) ENGINE InnoDB;

CREATE TABLE IF NOT EXISTS  COMPETICIONES (
competicion_id INT NOT NULL AUTO_INCREMENT,
PRIMARY KEY (competicion_id),
region ENUM ('Europa', 'Medio Oriente', 'Asia Oriental', 'Oceanía', 'Caribe', 'Norte America', 'Centro America', 'Sur America', 'Africa') NOT NULL,
nombre VARCHAR (100) NOT NULL
) ENGINE InnoDB;

CREATE TABLE IF NOT EXISTS  CAMPEONATOS(
campeonatos_id INT NOT NULL AUTO_INCREMENT,
PRIMARY KEY (campeonatos_id),
competicion_id INT NOT NULL,
CONSTRAINT competicion_id_fk
FOREIGN KEY (competicion_id) REFERENCES COMPETICIONES (competicion_id)
ON DELETE CASCADE
ON UPDATE CASCADE,
temporada_id INT NOT NULL,
CONSTRAINT temporada_id_fk
FOREIGN KEY (temporada_id) REFERENCES TEMPORADAS (temporada_id)
ON DELETE CASCADE
ON UPDATE CASCADE,
campeon VARCHAR (100) NOT NULL,
subcampeon VARCHAR (100) NOT NULL
) ENGINE InnoDB;
