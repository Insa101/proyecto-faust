USE nombreBBDD;

CREATE TABLE IF NOT EXISTS clientes(
                                	id_cliente INT NOT NULL AUTO_INCREMENT,
                                    nombre VARCHAR (50) NOT NULL,
                                    apellido1 VARCHAR (100) NOT NULL,
                                    apellido2 VARCHAR (100) NOT NULL,
                                    fnac DATE NOT NULL,
                                    email VARCHAR (100) NOT NULL UNIQUE,
                                    PRIMARY KEY (id_cliente)
                                    ) ENGINE InnoDB;
CREATE TABLE IF NOT EXISTS libros(
                                	id_libro INT NOT NULL AUTO_INCREMENT,
                                    titulo VARCHAR (100) NOT NULL UNIQUE,
                                    id_autor INT NOT NULL,
                                    PRIMARY KEY (id_libro),
                                    CONSTRAINT fk_autores
										FOREIGN KEY (id_autor) REFERENCES autores (id_autor)
										ON DELETE NO ACTION
                                        ON UPDATE CASCADE
                                    ) ENGINE InnoDB;
CREATE TABLE IF NOT EXISTS reservas (
                                	id_reserva INT NOT NULL AUTO_INCREMENT,
                                    freserva DATE NOT NULL, 
                                    id_libro INT NOT NULL,
                                    id_cliente INT NOT NULL,
                                    PRIMARY KEY (id_reserva),
                                    CONSTRAINT fk_libros
										FOREIGN KEY (id_libro) REFERENCES libros (id_libro)
                                        ON DELETE NO ACTION
                                        ON UPDATE CASCADE,
                                    CONSTRAINT fk_clientes    
										FOREIGN KEY (id_cliente) REFERENCES clientes (id_cliente)
										ON DELETE NO ACTION
                                        ON UPDATE CASCADE
                                    ) ENGINE InnoDB;