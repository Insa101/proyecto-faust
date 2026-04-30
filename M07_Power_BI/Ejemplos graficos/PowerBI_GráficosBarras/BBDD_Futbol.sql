-- Creación de la base de datos
CREATE DATABASE IF NOT EXISTS bbdd_futbol CHARACTER SET utf8mb4 COLLATE utf8mb4_spanish_ci;

-- Selección de la base de datos
USE bbdd_futbol;

-- Creación de la entidad países
CREATE TABLE IF NOT EXISTS paises(
	id_pais INT NOT NULL AUTO_INCREMENT,
    nombre VARCHAR(40) NOT NULL UNIQUE,
    abreviatura VARCHAR(3) NOT NULL UNIQUE,
    PRIMARY KEY(id_pais)
)ENGINE=INNODB;

-- Creación de la entidad equipos
CREATE TABLE  IF NOT EXISTS equipos(
	id_equipo INT NOT NULL AUTO_INCREMENT,
    nombre VARCHAR(60) NOT NULL UNIQUE,
    pais INT NOT NULL,
    PRIMARY KEY(id_equipo),
    CONSTRAINT fk_pais
        FOREIGN KEY (pais)
            REFERENCES paises(id_pais)
            ON DELETE NO ACTION
            ON UPDATE CASCADE
)ENGINE=INNODB;

-- Creación de la entidad temporadas
CREATE TABLE IF NOT EXISTS temporadas(
	id_temporada INT NOT NULL AUTO_INCREMENT,
    anyo_inicio YEAR NOT NULL,
    anyo_fin YEAR NOT NULL,
    PRIMARY KEY(id_temporada)
)ENGINE=INNODB;

-- Creación de la entidad competiciones
CREATE TABLE IF NOT EXISTS competiciones(
	id_competicion INT NOT NULL AUTO_INCREMENT,
    nombre VARCHAR(80) NOT NULL UNIQUE,
    PRIMARY KEY(id_competicion)
)ENGINE=INNODB;

-- Creación de la entidad campeonatos
CREATE TABLE IF NOT EXISTS campeonatos(
	id_campeonato INT NOT NULL AUTO_INCREMENT,
	competicion INT NOT NULL,
    temporada INT NOT NULL,
    equipo_campeon INT NOT NULL,
    equipo_subcampeon INT,
    PRIMARY KEY(id_campeonato),
    CONSTRAINT fk_competicion
        FOREIGN KEY (competicion)
            REFERENCES competiciones(id_competicion)
            ON DELETE NO ACTION
            ON UPDATE CASCADE,
    CONSTRAINT fk_temporada
        FOREIGN KEY (temporada)
            REFERENCES temporadas(id_temporada)
            ON DELETE NO ACTION
            ON UPDATE CASCADE,
    CONSTRAINT fk_equipoCampeon
        FOREIGN KEY (equipo_campeon)
            REFERENCES equipos(id_equipo)
            ON DELETE NO ACTION
            ON UPDATE CASCADE,
    CONSTRAINT fk_equipoSubcampeon
        FOREIGN KEY (equipo_subcampeon)
            REFERENCES equipos(id_equipo)
            ON DELETE SET NULL
            ON UPDATE CASCADE
)ENGINE=INNODB;

-- SELECCION DE LA BASE DE DATOS
USE bbdd_futbol;

-- INSERCIÓN DE LOS DATOS DE LOS PAISES
INSERT INTO paises (abreviatura, nombre) VALUES ('AFG', 'Afganistán');
INSERT INTO paises (abreviatura, nombre) VALUES ('ALB', 'Albania');
INSERT INTO paises (abreviatura, nombre) VALUES ('GER', 'Alemania');
INSERT INTO paises (abreviatura, nombre) VALUES ('AND', 'Andorra');
INSERT INTO paises (abreviatura, nombre) VALUES ('ANG', 'Angola');
INSERT INTO paises (abreviatura, nombre) VALUES ('AIA', 'Anguila');
INSERT INTO paises (abreviatura, nombre) VALUES ('ATG', 'Antigua y Barbuda');
INSERT INTO paises (abreviatura, nombre) VALUES ('ANT', 'Antillas Neerlandesas');
INSERT INTO paises (abreviatura, nombre) VALUES ('KSA', 'Arabia Saudita');
INSERT INTO paises (abreviatura, nombre) VALUES ('ALG', 'Argelia');
INSERT INTO paises (abreviatura, nombre) VALUES ('ARG', 'Argentina');
INSERT INTO paises (abreviatura, nombre) VALUES ('ARM', 'Armenia');
INSERT INTO paises (abreviatura, nombre) VALUES ('ARU', 'Aruba');
INSERT INTO paises (abreviatura, nombre) VALUES ('AUS', 'Australia');
INSERT INTO paises (abreviatura, nombre) VALUES ('AUT', 'Austria');
INSERT INTO paises (abreviatura, nombre) VALUES ('AZE', 'Azerbaiyán');
INSERT INTO paises (abreviatura, nombre) VALUES ('BAH', 'Bahamas');
INSERT INTO paises (abreviatura, nombre) VALUES ('BHR', 'Baréin');
INSERT INTO paises (abreviatura, nombre) VALUES ('BAN', 'Bangladesh');
INSERT INTO paises (abreviatura, nombre) VALUES ('BEL', 'Bélgica');
INSERT INTO paises (abreviatura, nombre) VALUES ('BLZ', 'Bélice');
INSERT INTO paises (abreviatura, nombre) VALUES ('BEN', 'Benín');
INSERT INTO paises (abreviatura, nombre) VALUES ('BER', 'Bermudas');
INSERT INTO paises (abreviatura, nombre) VALUES ('BLR', 'Bielorrusia');
INSERT INTO paises (abreviatura, nombre) VALUES ('MYA', 'Birmania');
INSERT INTO paises (abreviatura, nombre) VALUES ('BOL', 'Bolivia');
INSERT INTO paises (abreviatura, nombre) VALUES ('BIH', 'Bosnia Herzegovina');
INSERT INTO paises (abreviatura, nombre) VALUES ('BOT', 'Botswana');
INSERT INTO paises (abreviatura, nombre) VALUES ('BRA', 'Brasil');
INSERT INTO paises (abreviatura, nombre) VALUES ('BRU', 'Brunei');
INSERT INTO paises (abreviatura, nombre) VALUES ('BUL', 'Bulgaria');
INSERT INTO paises (abreviatura, nombre) VALUES ('BFA', 'Burkina Faso');
INSERT INTO paises (abreviatura, nombre) VALUES ('BDI', 'Burundi');
INSERT INTO paises (abreviatura, nombre) VALUES ('BHU', 'Bután');
INSERT INTO paises (abreviatura, nombre) VALUES ('CPV', 'Cabo Verde');
INSERT INTO paises (abreviatura, nombre) VALUES ('CAM', 'Camboya');
INSERT INTO paises (abreviatura, nombre) VALUES ('CMR', 'Camerún');
INSERT INTO paises (abreviatura, nombre) VALUES ('CAN', 'Canadá');
INSERT INTO paises (abreviatura, nombre) VALUES ('QAT', 'Catar');
INSERT INTO paises (abreviatura, nombre) VALUES ('CHA', 'Chad');
INSERT INTO paises (abreviatura, nombre) VALUES ('CHI', 'Chile');
INSERT INTO paises (abreviatura, nombre) VALUES ('CHN', 'China');
INSERT INTO paises (abreviatura, nombre) VALUES ('CYP', 'Chipre');
INSERT INTO paises (abreviatura, nombre) VALUES ('COL', 'Colombia');
INSERT INTO paises (abreviatura, nombre) VALUES ('PRK', 'Corea del Norte');
INSERT INTO paises (abreviatura, nombre) VALUES ('KOR', 'Corea del Sur');
INSERT INTO paises (abreviatura, nombre) VALUES ('CIV', 'Costa de Marfil');
INSERT INTO paises (abreviatura, nombre) VALUES ('CRC', 'Costa Rica');
INSERT INTO paises (abreviatura, nombre) VALUES ('CRO', 'Croacia');
INSERT INTO paises (abreviatura, nombre) VALUES ('CUB', 'Cuba');
INSERT INTO paises (abreviatura, nombre) VALUES ('DEN', 'Dinamarca');
INSERT INTO paises (abreviatura, nombre) VALUES ('DMA', 'Dominica');
INSERT INTO paises (abreviatura, nombre) VALUES ('ECU', 'Ecuador');
INSERT INTO paises (abreviatura, nombre) VALUES ('EGY', 'Egipto');
INSERT INTO paises (abreviatura, nombre) VALUES ('SLV', 'El Salvador');
INSERT INTO paises (abreviatura, nombre) VALUES ('UAE', 'Emiratos Árabes Unidos');
INSERT INTO paises (abreviatura, nombre) VALUES ('ERI', 'Eritrea');
INSERT INTO paises (abreviatura, nombre) VALUES ('SCO', 'Escocia');
INSERT INTO paises (abreviatura, nombre) VALUES ('SVK', 'Eslovaquia');
INSERT INTO paises (abreviatura, nombre) VALUES ('SVN', 'Eslovenia');
INSERT INTO paises (abreviatura, nombre) VALUES ('ESP', 'España');
INSERT INTO paises (abreviatura, nombre) VALUES ('USA', 'Estados Unidos');
INSERT INTO paises (abreviatura, nombre) VALUES ('EST', 'Estonia');
INSERT INTO paises (abreviatura, nombre) VALUES ('ETH', 'Etiopía');
INSERT INTO paises (abreviatura, nombre) VALUES ('FIJ', 'Fiji');
INSERT INTO paises (abreviatura, nombre) VALUES ('PHI', 'Filipinas');
INSERT INTO paises (abreviatura, nombre) VALUES ('FIN', 'Finlandia');
INSERT INTO paises (abreviatura, nombre) VALUES ('FRA', 'Francia');
INSERT INTO paises (abreviatura, nombre) VALUES ('GAB', 'Gabón');
INSERT INTO paises (abreviatura, nombre) VALUES ('WAL', 'Gales');
INSERT INTO paises (abreviatura, nombre) VALUES ('GAM', 'Gambia');
INSERT INTO paises (abreviatura, nombre) VALUES ('GHA', 'Ghana');
INSERT INTO paises (abreviatura, nombre) VALUES ('GRN', 'Granada');
INSERT INTO paises (abreviatura, nombre) VALUES ('GRE', 'Grecia');
INSERT INTO paises (abreviatura, nombre) VALUES ('GUM', 'Guam');
INSERT INTO paises (abreviatura, nombre) VALUES ('GUA', 'Guatemala');
INSERT INTO paises (abreviatura, nombre) VALUES ('GUI', 'Guinea');
INSERT INTO paises (abreviatura, nombre) VALUES ('GNB', 'Guinea-Bissau');
INSERT INTO paises (abreviatura, nombre) VALUES ('EQG', 'Guinea Ecuatorial');
INSERT INTO paises (abreviatura, nombre) VALUES ('GUY', 'Guyana');
INSERT INTO paises (abreviatura, nombre) VALUES ('HAI', 'Haití');
INSERT INTO paises (abreviatura, nombre) VALUES ('HON', 'Honduras');
INSERT INTO paises (abreviatura, nombre) VALUES ('HKG', 'Hong Kong');
INSERT INTO paises (abreviatura, nombre) VALUES ('HUN', 'Hungría');
INSERT INTO paises (abreviatura, nombre) VALUES ('IND', 'India');
INSERT INTO paises (abreviatura, nombre) VALUES ('IDN', 'Indonesia');
INSERT INTO paises (abreviatura, nombre) VALUES ('ENG', 'Inglaterra');
INSERT INTO paises (abreviatura, nombre) VALUES ('IRN', 'Irán');
INSERT INTO paises (abreviatura, nombre) VALUES ('IRQ', 'Iraq');
INSERT INTO paises (abreviatura, nombre) VALUES ('IRL', 'Irlanda');
INSERT INTO paises (abreviatura, nombre) VALUES ('NIR', 'Irlanda del Norte');
INSERT INTO paises (abreviatura, nombre) VALUES ('ISL', 'Islandia');
INSERT INTO paises (abreviatura, nombre) VALUES ('CAY', 'Islas Caimán');
INSERT INTO paises (abreviatura, nombre) VALUES ('COK', 'Islas Cook');
INSERT INTO paises (abreviatura, nombre) VALUES ('FRO', 'Islas Feroe');
INSERT INTO paises (abreviatura, nombre) VALUES ('SOL', 'Islas Salomón');
INSERT INTO paises (abreviatura, nombre) VALUES ('TCA', 'Islas Turcas y Caicos');
INSERT INTO paises (abreviatura, nombre) VALUES ('VGB', 'Islas Vírgenes Británicas');
INSERT INTO paises (abreviatura, nombre) VALUES ('VIR', 'Islas Vírgenes Estadounidenses');
INSERT INTO paises (abreviatura, nombre) VALUES ('ISR', 'Israel');
INSERT INTO paises (abreviatura, nombre) VALUES ('ITA', 'Italia');
INSERT INTO paises (abreviatura, nombre) VALUES ('JAM', 'Jamaica');
INSERT INTO paises (abreviatura, nombre) VALUES ('JPN', 'Japón');
INSERT INTO paises (abreviatura, nombre) VALUES ('JOR', 'Jordania');
INSERT INTO paises (abreviatura, nombre) VALUES ('KAZ', 'Kazajistán');
INSERT INTO paises (abreviatura, nombre) VALUES ('KEN', 'Kenia');
INSERT INTO paises (abreviatura, nombre) VALUES ('KGZ', 'Kirguizistán');
INSERT INTO paises (abreviatura, nombre) VALUES ('KUW', 'Kuwait');
INSERT INTO paises (abreviatura, nombre) VALUES ('LAO', 'Laos');
INSERT INTO paises (abreviatura, nombre) VALUES ('LES', 'Lesoto');
INSERT INTO paises (abreviatura, nombre) VALUES ('LVA', 'Letonia');
INSERT INTO paises (abreviatura, nombre) VALUES ('LIB', 'Líbano');
INSERT INTO paises (abreviatura, nombre) VALUES ('LBR', 'Liberia');
INSERT INTO paises (abreviatura, nombre) VALUES ('LBY', 'Libia');
INSERT INTO paises (abreviatura, nombre) VALUES ('LIE', 'Liechtenstein');
INSERT INTO paises (abreviatura, nombre) VALUES ('LTU', 'Lituania');
INSERT INTO paises (abreviatura, nombre) VALUES ('LUX', 'Luxemburgo');
INSERT INTO paises (abreviatura, nombre) VALUES ('MAC', 'Macao');
INSERT INTO paises (abreviatura, nombre) VALUES ('MKD', 'Macedonia');
INSERT INTO paises (abreviatura, nombre) VALUES ('MAD', 'Madagascar');
INSERT INTO paises (abreviatura, nombre) VALUES ('MAS', 'Malasia');
INSERT INTO paises (abreviatura, nombre) VALUES ('MWI', 'Malaui');
INSERT INTO paises (abreviatura, nombre) VALUES ('MDV', 'Maldivas');
INSERT INTO paises (abreviatura, nombre) VALUES ('MLI', 'Malí');
INSERT INTO paises (abreviatura, nombre) VALUES ('MLT', 'Malta');
INSERT INTO paises (abreviatura, nombre) VALUES ('MAR', 'Marruecos');
INSERT INTO paises (abreviatura, nombre) VALUES ('MRI', 'Mauricio');
INSERT INTO paises (abreviatura, nombre) VALUES ('MTN', 'Mauritania');
INSERT INTO paises (abreviatura, nombre) VALUES ('MEX', 'Méjico');
INSERT INTO paises (abreviatura, nombre) VALUES ('MDA', 'Moldavia');
INSERT INTO paises (abreviatura, nombre) VALUES ('MGL', 'Mongolia');
INSERT INTO paises (abreviatura, nombre) VALUES ('MNE', 'Montenegro');
INSERT INTO paises (abreviatura, nombre) VALUES ('MSR', 'Montserrat');
INSERT INTO paises (abreviatura, nombre) VALUES ('MOZ', 'Mozambique');
INSERT INTO paises (abreviatura, nombre) VALUES ('NAM', 'Namibia');
INSERT INTO paises (abreviatura, nombre) VALUES ('NEP', 'Nepal');
INSERT INTO paises (abreviatura, nombre) VALUES ('NCA', 'Nicaragua');
INSERT INTO paises (abreviatura, nombre) VALUES ('NIG', 'Níger');
INSERT INTO paises (abreviatura, nombre) VALUES ('NGA', 'Nigeria');
INSERT INTO paises (abreviatura, nombre) VALUES ('NOR', 'Noruega');
INSERT INTO paises (abreviatura, nombre) VALUES ('NCL', 'Nueva Caledonia');
INSERT INTO paises (abreviatura, nombre) VALUES ('NZL', 'Nueva Zelanda');
INSERT INTO paises (abreviatura, nombre) VALUES ('OMA', 'Omán');
INSERT INTO paises (abreviatura, nombre) VALUES ('NED', 'Países Bajos');
INSERT INTO paises (abreviatura, nombre) VALUES ('PAK', 'Pakistán');
INSERT INTO paises (abreviatura, nombre) VALUES ('PLE', 'Palestina');
INSERT INTO paises (abreviatura, nombre) VALUES ('PAN', 'Panamá');
INSERT INTO paises (abreviatura, nombre) VALUES ('PNG', 'Papúa Nueva Guinea');
INSERT INTO paises (abreviatura, nombre) VALUES ('PAR', 'Paraguay');
INSERT INTO paises (abreviatura, nombre) VALUES ('PER', 'Perú');
INSERT INTO paises (abreviatura, nombre) VALUES ('POL', 'Polonia');
INSERT INTO paises (abreviatura, nombre) VALUES ('POR', 'Portugal');
INSERT INTO paises (abreviatura, nombre) VALUES ('PUR', 'Puerto Rico');
INSERT INTO paises (abreviatura, nombre) VALUES ('CTA', 'República Centroafricana');
INSERT INTO paises (abreviatura, nombre) VALUES ('CZE', 'República Checa');
INSERT INTO paises (abreviatura, nombre) VALUES ('CGO', 'República del Congo');
INSERT INTO paises (abreviatura, nombre) VALUES ('COD', 'República Democrática del Congo');
INSERT INTO paises (abreviatura, nombre) VALUES ('DOM', 'República Dominicana');
INSERT INTO paises (abreviatura, nombre) VALUES ('RWA', 'Ruanda');
INSERT INTO paises (abreviatura, nombre) VALUES ('ROU', 'Rumania');
INSERT INTO paises (abreviatura, nombre) VALUES ('RUS', 'Rusia');
INSERT INTO paises (abreviatura, nombre) VALUES ('SAM', 'Samoa');
INSERT INTO paises (abreviatura, nombre) VALUES ('ASA', 'Samoa Americana');
INSERT INTO paises (abreviatura, nombre) VALUES ('SKN', 'San Cristóbal y Nevis');
INSERT INTO paises (abreviatura, nombre) VALUES ('SMR', 'San Marino');
INSERT INTO paises (abreviatura, nombre) VALUES ('VIN', 'San Vicente y las Granadinas');
INSERT INTO paises (abreviatura, nombre) VALUES ('LCA', 'Santa Lucía');
INSERT INTO paises (abreviatura, nombre) VALUES ('STP', 'Santo Tomé y Príncipe');
INSERT INTO paises (abreviatura, nombre) VALUES ('SEN', 'Senegal');
INSERT INTO paises (abreviatura, nombre) VALUES ('SRB', 'Serbia');
INSERT INTO paises (abreviatura, nombre) VALUES ('SEY', 'Seychelles');
INSERT INTO paises (abreviatura, nombre) VALUES ('SLE', 'Sierra Leona');
INSERT INTO paises (abreviatura, nombre) VALUES ('SIN', 'Singapur');
INSERT INTO paises (abreviatura, nombre) VALUES ('SOM', 'Somalia');
INSERT INTO paises (abreviatura, nombre) VALUES ('SRI', 'Sri Lanka');
INSERT INTO paises (abreviatura, nombre) VALUES ('SWZ', 'Suazilandia');
INSERT INTO paises (abreviatura, nombre) VALUES ('RSA', 'Sudáfrica');
INSERT INTO paises (abreviatura, nombre) VALUES ('SUD', 'Sudán');
INSERT INTO paises (abreviatura, nombre) VALUES ('SWE', 'Suecia');
INSERT INTO paises (abreviatura, nombre) VALUES ('SUI', 'Suiza');
INSERT INTO paises (abreviatura, nombre) VALUES ('SUR', 'Surinam');
INSERT INTO paises (abreviatura, nombre) VALUES ('TAH', 'Tahiti');
INSERT INTO paises (abreviatura, nombre) VALUES ('THA', 'Tailandia');
INSERT INTO paises (abreviatura, nombre) VALUES ('TAN', 'Tanzania');
INSERT INTO paises (abreviatura, nombre) VALUES ('TJK', 'Tayikistán');
INSERT INTO paises (abreviatura, nombre) VALUES ('TLS', 'Timor Oriental');
INSERT INTO paises (abreviatura, nombre) VALUES ('TOG', 'Togo');
INSERT INTO paises (abreviatura, nombre) VALUES ('TGA', 'Tonga');
INSERT INTO paises (abreviatura, nombre) VALUES ('TRI', 'Trinidad y Tobago');
INSERT INTO paises (abreviatura, nombre) VALUES ('TUN', 'Túnez');
INSERT INTO paises (abreviatura, nombre) VALUES ('TKM', 'Turkmenistán');
INSERT INTO paises (abreviatura, nombre) VALUES ('TUR', 'Turquía');
INSERT INTO paises (abreviatura, nombre) VALUES ('UKR', 'Ucrania');
INSERT INTO paises (abreviatura, nombre) VALUES ('UGA', 'Uganda');
INSERT INTO paises (abreviatura, nombre) VALUES ('URU', 'Uruguay');
INSERT INTO paises (abreviatura, nombre) VALUES ('UZB', 'Uzbekistán');
INSERT INTO paises (abreviatura, nombre) VALUES ('VAN', 'Vanuatu');
INSERT INTO paises (abreviatura, nombre) VALUES ('VEN', 'Venezuela');
INSERT INTO paises (abreviatura, nombre) VALUES ('VIE', 'Vietnam');
INSERT INTO paises (abreviatura, nombre) VALUES ('YEM', 'Yemen');
INSERT INTO paises (abreviatura, nombre) VALUES ('DJI', 'Yibuti');
INSERT INTO paises (abreviatura, nombre) VALUES ('ZAM', 'Zambia');
INSERT INTO paises (abreviatura, nombre) VALUES ('ZIM', 'Zimbabue');


-- INSERCIÓN DE LOS DATOS DE LOS EQUIPOS
INSERT INTO equipos (nombre, pais) VALUES ('Real Zaragoza', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Real Madrid', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Real Sociedad', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Real Betis Balompié', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Fútbol Club Barcelona', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Atlético de Madrid', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Athletic Club de Bilbao', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Valencia Club de Fútbol', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Sevilla Fútbol Club', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Real Club Deportivo de la Coruña', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Real Club Deportivo Español de Barcelona', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Real Unión Club de Irún', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Arenas Club de Guecho', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Vizcaya', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Vigo Sporting', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Club Ciclista de San Sebastián', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Basconia', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Club Español de Madrid', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Gimnástica', 61);
INSERT INTO equipos (nombre, pais) VALUES ('España', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Europa', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Sabadell', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Racing Club de Ferrol', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Real Club Celta de Vigo', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Real Valladolid', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Granada', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Elche', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Club Deportivo Castellón', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Unión Deportiva Las Palmas', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Castilla Club de Fútbol', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Sporting de Gijón', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Real Club Deportivo Mallorca', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Real Club Recreativo de Huelva', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Club Atlético Osasuna', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Getafe Club de Fútbol', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Racing de Irún', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Racing de Santander', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Atlético Aviación', 61);
INSERT INTO equipos (nombre, pais) VALUES ('Villareal Club de Fútbol', 61);

INSERT INTO equipos (nombre, pais) VALUES ("A.C.F. Fiorentina", 101);
INSERT INTO equipos (nombre, pais) VALUES ("Glasgow Rangers", 58);
INSERT INTO equipos (nombre, pais) VALUES ("Sporting de Lisboa", 152);
INSERT INTO equipos (nombre, pais) VALUES ("MTK Budapest", 84);
INSERT INTO equipos (nombre, pais) VALUES ("West Ham United", 87);
INSERT INTO equipos (nombre, pais) VALUES ("TSV Munich", 3);
INSERT INTO equipos (nombre, pais) VALUES ("Borussia Dortmund", 3);
INSERT INTO equipos (nombre, pais) VALUES ("Liverpool", 87);
INSERT INTO equipos (nombre, pais) VALUES ("Bayern Munich", 3);
INSERT INTO equipos (nombre, pais) VALUES ("A.C. Milan", 101);
INSERT INTO equipos (nombre, pais) VALUES ("Hamburgo SV", 3);
INSERT INTO equipos (nombre, pais) VALUES ("Slovan Bratislava", 155);
INSERT INTO equipos (nombre, pais) VALUES ("Górnik Zabrze", 151);
INSERT INTO equipos (nombre, pais) VALUES ("Chelsea", 87);
INSERT INTO equipos (nombre, pais) VALUES ("Dinamo Moscú", 161);
INSERT INTO equipos (nombre, pais) VALUES ("Leeds United", 87);
INSERT INTO equipos (nombre, pais) VALUES ("F.C. Magdeburgo", 3);
INSERT INTO equipos (nombre, pais) VALUES ("Dinamo Kiev", 161);
INSERT INTO equipos (nombre, pais) VALUES ("Ferencváros", 84);
INSERT INTO equipos (nombre, pais) VALUES ("Anderlecht", 20);
INSERT INTO equipos (nombre, pais) VALUES ("Austria Viena", 15);
INSERT INTO equipos (nombre, pais) VALUES ("Fortuna Düsseldorf", 3);
INSERT INTO equipos (nombre, pais) VALUES ("Arsenal", 87);
INSERT INTO equipos (nombre, pais) VALUES ("Dinamo Tbilisi", 161);
INSERT INTO equipos (nombre, pais) VALUES ("Carl Zeiss Jena", 3);
INSERT INTO equipos (nombre, pais) VALUES ("Standard Lieja", 20);
INSERT INTO equipos (nombre, pais) VALUES ("Aberdeen", 58);
INSERT INTO equipos (nombre, pais) VALUES ("Juventus", 101);
INSERT INTO equipos (nombre, pais) VALUES ("Oporto", 152);
INSERT INTO equipos (nombre, pais) VALUES ("Everton", 87);
INSERT INTO equipos (nombre, pais) VALUES ("Rapid Viena", 15);
INSERT INTO equipos (nombre, pais) VALUES ("Ajax", 144);
INSERT INTO equipos (nombre, pais) VALUES ("Lokomotiv Leipzig", 3);
INSERT INTO equipos (nombre, pais) VALUES ("K.V. Mechelen", 20);
INSERT INTO equipos (nombre, pais) VALUES ("Sampdoria", 101);
INSERT INTO equipos (nombre, pais) VALUES ("Manchester United", 87);
INSERT INTO equipos (nombre, pais) VALUES ("Werder Bremen", 3);
INSERT INTO equipos (nombre, pais) VALUES ("A.S. Mónaco", 68);
INSERT INTO equipos (nombre, pais) VALUES ("Parma", 101);
INSERT INTO equipos (nombre, pais) VALUES ("Amberes", 20);
INSERT INTO equipos (nombre, pais) VALUES ("París Saint-Germain", 68);
INSERT INTO equipos (nombre, pais) VALUES ("VfB Stuttgart", 3);
INSERT INTO equipos (nombre, pais) VALUES ("Lazio", 101);



-- INSERCIÓN DE LOS DATOS DE LAS TEMPORADAS
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1904, 1905);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1905, 1906);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1906, 1907);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1907, 1908);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1908, 1909);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1909, 1910);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1910, 1911);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1911, 1912);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1912, 1913);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1913, 1914);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1914, 1915);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1915, 1916);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1916, 1917);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1917, 1918);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1918, 1919);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1919, 1920);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1920, 1921);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1921, 1922);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1922, 1923);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1923, 1924);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1924, 1925);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1925, 1926);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1926, 1927);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1927, 1928);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1928, 1929);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1929, 1930);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1930, 1931);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1931, 1932);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1932, 1933);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1933, 1934);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1934, 1935);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1935, 1936);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1938, 1939);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1939, 1940);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1940, 1941);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1941, 1942);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1942, 1943);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1943, 1944);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1944, 1945);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1945, 1946);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1946, 1947);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1947, 1948);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1948, 1949);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1949, 1950);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1950, 1951);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1951, 1952);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1952, 1953);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1953, 1954);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1954, 1955);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1955, 1956);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1956, 1957);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1957, 1958);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1958, 1959);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1959, 1960);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1960, 1961);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1961, 1962);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1962, 1963);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1963, 1964);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1964, 1965);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1965, 1966);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1966, 1967);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1967, 1968);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1968, 1969);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1969, 1970);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1970, 1971);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1971, 1972);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1972, 1973);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1973, 1974);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1974, 1975);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1975, 1976);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1976, 1977);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1977, 1978);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1978, 1979);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1979, 1980);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1980, 1981);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1981, 1982);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1982, 1983);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1983, 1984);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1984, 1985);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1985, 1986);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1986, 1987);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1987, 1988);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1988, 1989);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1989, 1990);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1990, 1991);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1991, 1992);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1992, 1993);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1993, 1994);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1994, 1995);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1995, 1996);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1996, 1997);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1997, 1998);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1998, 1999);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (1999, 2000);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (2000, 2001);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (2001, 2002);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (2002, 2003);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (2003, 2004);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (2004, 2005);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (2005, 2006);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (2006, 2007);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (2007, 2008);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (2008, 2009);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (2009, 2010);
INSERT INTO temporadas (anyo_inicio, anyo_fin) VALUES (2010, 2011);


-- INSERCIÓN DE LOS DATOS DE LOS CAMPEONATOS
INSERT INTO competiciones (nombre) VALUES ('Liga española');
INSERT INTO competiciones (nombre) VALUES ('Copa de la Coronación');
INSERT INTO competiciones (nombre) VALUES ('Copa de S. M. el Rey');
INSERT INTO competiciones (nombre) VALUES ('Copa del Presidente de la República');
INSERT INTO competiciones (nombre) VALUES ('Copa de S. E. El Generalísimo');
INSERT INTO competiciones (nombre) VALUES ('Supercopa de España');
INSERT INTO competiciones (nombre) VALUES ('Copa de Europa');
INSERT INTO competiciones (nombre) VALUES ('Recopa');
INSERT INTO competiciones (nombre) VALUES ('Copa de Ferias');
INSERT INTO competiciones (nombre) VALUES ('Copa de la UEFA');
INSERT INTO competiciones (nombre) VALUES ('Europa League');
INSERT INTO competiciones (nombre) VALUES ('Supercopa de Europa');
INSERT INTO competiciones (nombre) VALUES ('Copa Intercontinental');
INSERT INTO competiciones (nombre) VALUES ('Mundial de Clubes');

-- INSERCIÓN DE LOS DATOS DE LAS COMPETICIONES
-- Copa de la Coronación
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (2, 1, 14, 5);

-- Copa del Rey
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 2, 7, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon) VALUES (3, 3, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 4, 2, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 5, 2, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 6, 2, 14);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 7, 2, 15);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 8, 16, 11);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 9, 7, 17);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 9, 5, 18);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 10, 7, 11);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 11, 5, 19);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 12, 36, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 12, 5, 3);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 13, 7, 20);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 14, 7, 11);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 15, 7, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 16, 2, 13);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 17, 12, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 18, 13, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 19, 5, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 20, 7, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 21, 5, 12);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 22, 7, 21);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 23, 12, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 24, 5, 13);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 25, 5, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 26, 12, 13);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 27, 5, 3);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 28, 11, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 29, 7, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 30, 7, 4);

-- Copa del presidente de la República
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (4, 31, 7, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (4, 32, 7, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (4, 33, 2, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (4, 34, 9, 22);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (4, 35, 2, 5);

/* Copa del Generalísimo */
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 36, 9, 23);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 37, 11, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 38, 8, 11);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 39, 5, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 40, 7, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 41, 7, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 42, 7, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 43, 2, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 44, 2, 11);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 45, 9, 24);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 46, 8, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 47, 7, 25);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 48, 5, 3);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 49, 5, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 50, 5, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 51, 8, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 52, 7, 9);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 53, 7, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 54, 5, 11);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 55, 7, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 56, 5, 26);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 57, 6, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 58, 6, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 59, 2, 9);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 60, 5, 1);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 61, 1, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 62, 6, 1);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 63, 1, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 64, 8, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 65, 5, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 66, 7, 27);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 67, 2, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 68, 5, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 69, 6, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 70, 7, 28);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 71, 2, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 72, 2, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (5, 73, 6, 1);

-- Copa del Rey
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 74, 4, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 75, 5, 29);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 76, 8, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 77, 2, 30);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 78, 5, 31);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 79, 2, 31);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 80, 5, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 81, 7, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 82, 6, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 83, 1, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 84, 3, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 85, 5, 3);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 86, 2, 25);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 87, 5, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 88, 6, 32);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 89, 6, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 90, 2, 1);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 91, 1, 24);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 92, 10, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 93, 6, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 94, 5, 4);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 95, 5, 32);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 96, 8, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 97, 11, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 98, 1, 24);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 99, 10, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 100, 32, 33);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 101, 1, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 102, 4, 34);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 103, 11, 1);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 104, 9, 35);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 105, 8, 35);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 104, 5, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (3, 105, 9, 6);

-- Liga española
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 28, 5, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 29, 7, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 30, 7, 37);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 31, 2, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 32, 2, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 33, 7, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 34, 4, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 35, 7, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 37, 38, 9);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 38, 38, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 39, 8, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 40, 7, 9);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 41, 8, 38);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 42, 5, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 43, 9, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 44, 8, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 45, 5, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 46, 5, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 47, 6, 10);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 48, 6, 9);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 49, 5, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 50, 5, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 51, 2, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 52, 2, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 53, 7, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 54, 2, 9);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 55, 2, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 56, 5, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 57, 5, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 58, 2, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 59, 2, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 60, 2, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 61, 2, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 62, 2, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 63, 6, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 64, 2, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 65, 2, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 66, 2, 29);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 67, 6, 7);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 68, 8, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 69, 2, 8);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 70, 6, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 71, 5, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 72, 2, 1);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 73, 2, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 74, 6, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 75, 2, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 76, 2, 31);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 77, 2, 3);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 78, 3, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 79, 3, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 80, 7, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 81, 7, 2);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 82, 5, 6);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 83, 2, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 84, 2, 5);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 85, 2, 3);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 86, 2, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 87, 2, 8);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 88, 5, 6);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 89, 5, 2);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 90, 5, 2);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 91, 5, 10);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 92, 2, 10);
INSERT INTO campeonatos(competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 93, 6, 8);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 94, 2, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 95, 5, 7);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 96, 5, 2);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 97, 10, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 98, 2, 10);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 99, 8, 10);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 100, 2, 3);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 101, 8, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 102, 5, 2);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 103, 5, 2);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 104, 2, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 105, 2, 39);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 104, 5, 2);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (1, 105, 5, 2);

-- Supercopa española
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 80, 3, 2);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 81, 5, 7);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon) VALUES (6, 82, 7);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 83, 6, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 86, 2, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon) VALUES (6, 87, 2);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 88, 2, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 89, 5, 6);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 90, 5, 6);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 91, 2, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 92, 5, 1);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 93, 10, 2);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 94, 5, 6);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 95, 2, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 96, 32, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 97, 8, 5);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 98, 10, 11);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 99, 2, 1);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 100, 10, 8);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 101, 2, 32);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 102, 1, 8);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 103, 5, 4);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 104, 5, 11);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 105, 9, 2);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 104, 2, 8);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 105, 5, 7);
INSERT INTO campeonatos (competicion, temporada, equipo_campeon, equipo_subcampeon) VALUES (6, 105, 5, 9);