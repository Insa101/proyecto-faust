-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Servidor: localhost
-- Tiempo de generación: 10-10-2023 a las 17:35:45
-- Versión del servidor: 10.4.27-MariaDB
-- Versión de PHP: 8.2.0

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de datos: `database_examen`
--
CREATE DATABASE IF NOT EXISTS database_examen CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;

USE database_examen;
-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `categorias_productos`
--

CREATE TABLE `categorias_productos` (
  `id_categoria` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `descripcion` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `categorias_productos`
--

INSERT INTO `categorias_productos` (`id_categoria`, `nombre`, `descripcion`) VALUES
(1, 'Electrónica', 'Productos electrónicos y dispositivos'),
(2, 'Ropa', 'Ropa y accesorios de moda'),
(3, 'Menaje del Hogar', 'Artículos para el hogar y decoración'),
(4, 'Deportes', 'Equipamiento deportivo y ropa deportiva'),
(5, 'Juguetes', 'Juguetes y juegos para niños'),
(6, 'Alimentos', 'Alimentos y productos de despensa'),
(7, 'Belleza', 'Productos de belleza y cuidado personal'),
(8, 'Electrodomésticos', 'Electrodomésticos para el hogar'),
(9, 'Hogar y Jardín', 'Productos para el hogar y jardín.'),
(10, 'Muebles', 'Muebles y mobiliario'),
(11, 'Libros', 'Libros y literatura'),
(12, 'Joyería', 'Joyería y accesorios'),
(13, 'Automóviles', 'Piezas y accesorios para automóviles'),
(14, 'Música', 'Instrumentos musicales y música'),
(15, 'Herramientas', 'Herramientas y equipos de bricolaje'),
(16, 'Salud', 'Productos de salud y bienestar'),
(17, 'Mascotas', 'Productos para mascotas'),
(18, 'Fotografía', 'Equipo y accesorios para fotografía.'),
(19, 'Arte y Manualidades', 'Suministros de arte y manualidades'),
(20, 'Cine y TV', 'Películas y programas de televisión'),
(21, 'Calzado', 'Calzado y zapatos');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `clientes`
--

CREATE TABLE `clientes` (
  `id_cliente` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `apellidos` varchar(255) NOT NULL,
  `doc_identidad` varchar(20) NOT NULL,
  `telefono` varchar(20) NOT NULL,
  `email` varchar(255) NOT NULL,
  `fecha_nacimiento` date NOT NULL,
  `direccion` varchar(255) NOT NULL,
  `provincia` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `clientes`
--

INSERT INTO `clientes` (`id_cliente`, `nombre`, `apellidos`, `doc_identidad`, `telefono`, `email`, `fecha_nacimiento`, `direccion`, `provincia`) VALUES
(1, 'Carlos', 'Gómez', '123456789A', '123-456-7890', 'carlos_go@email.com', '1990-03-12', 'Calle A, Madrid', 1),
(2, 'Ana', 'Martínez', '987654321B', '987-654-3210', 'ana_m@email.com', '1988-07-25', 'Calle B, Barcelona', 2),
(3, 'Franz', 'Schmidt', '234567890C', '555-123-4567', 'franz@email.com', '1992-05-08', 'Hauptstrasse 1, Berlín', 4),
(4, 'Giulia', 'Rossi', '345678901D', '123-789-4560', 'giulia@email.com', '1989-12-20', 'Via Roma 12, Milán', 3),
(5, 'José', 'Santos', '456789012E', '555-987-6543', 'jose@email.com', '1985-10-15', 'Rua A, Lisboa', 5),
(6, 'Emily', 'Smith', '567890123F', '123-456-7890', 'emily_s@email.com', '1993-02-28', 'High Street 20, Londres', 6),
(7, 'Mia', 'Johansson', '678901234G', '987-654-3210', 'mia@email.com', '1987-09-03', 'Drottninggatan 15, Estocolmo', 7),
(8, 'Olav', 'Olsen', '789012345H', '555-123-4567', 'olav@email.com', '1991-04-17', 'Karl Johans gate 8, Oslo', 8),
(9, 'Hans', 'Hansen', '890123456I', '123-789-4560', 'hans@email.com', '1994-08-22', 'Strøget 7, Copenhague', 9),
(10, 'Laura', 'Bianchi', '901234567J', '555-987-6543', 'laura@email.com', '1996-06-11', 'Keskuskatu 3, Helsinki', 10),
(11, 'Lucas', 'de Jong', '012345678K', '123-456-7890', 'lucas@email.com', '1986-11-05', 'Damstraat 10, Ámsterdam', 11),
(12, 'Sophie', 'Dubois', '123456789L', '987-654-3210', 'sophie@email.com', '1990-01-30', 'Rue de la Paix 5, París', 12),
(13, 'Max', 'Müller', '234567890M', '555-123-4567', 'max@email.com', '1984-03-18', 'Bahnhofstrasse 15, Zúrich', 13),
(14, 'Eva', 'Weber', '345678901N', '123-789-4560', 'eva@email.com', '1995-07-07', 'Kärntnertor 3, Viena', 14),
(15, 'Nikos', 'Papadopoulos', '456789012O', '555-987-6543', 'nikos@email.com', '1988-09-14', 'Syntagma Square 1, Atenas', 15),
(16, 'Isabella', 'Ricci', '123456789M', '123-456-7890', 'isabella@email.com', '1991-09-18', 'Via Venezia 8, Milán', 3),
(17, 'Pablo', 'Rodríguez', '987654321N', '987-654-3210', 'pablo@email.com', '1989-02-13', 'Rua B, Lisboa', 5),
(18, 'Sophia', 'Schneider', '234567890O', '555-123-4567', 'sophia@email.com', '1996-05-27', 'Schlossallee 6, Berlín', 4),
(19, 'Lukas', 'Novak', '345678901P', '123-789-4560', 'lukas@email.com', '1987-03-01', 'Václavské náměstí 12, Praga', 16),
(20, 'Laura', 'Santos', '456789012Q', '555-987-6543', 'laura_s@email.com', '1993-12-24', 'Calle C, Madrid', 1),
(21, 'Martina', 'Bianchi', '567890123R', '123-456-7890', 'martina@email.com', '1988-08-08', 'Viale Roma 7, Roma', 3),
(22, 'Sebastian', 'Müller', '678901234S', '987-654-3210', 'sebastian@email.com', '1992-01-04', 'Lindengasse 2, Viena', 14),
(23, 'Katarina', 'Larsen', '789012345T', '555-123-4567', 'katarina@email.com', '1990-04-14', 'Bakken 5, Copenhague', 9),
(24, 'Gustav', 'Eriksson', '890123456U', '123-789-4560', 'gustav@email.com', '1986-07-30', 'Sveavägen 15, Estocolmo', 7),
(25, 'Lea', 'Andersson', '901234567V', '555-987-6543', 'lea@email.com', '1994-03-12', 'Kungsgatan 9, Gotemburgo', 7),
(26, 'Manuel', 'Hernández', '012345678W', '123-456-7890', 'manuel@email.com', '1997-11-11', 'Calle D, Madrid', 1),
(27, 'Emma', 'Johansen', '123456789X', '987-654-3210', 'emma@email.com', '1989-06-26', 'Norre Kystvej 7, Copenhague', 9),
(28, 'Lena', 'Sørensen', '234567890Y', '555-123-4567', 'lena@email.com', '1991-02-19', 'Kungsportsavenyen 10, Gotemburgo', 7),
(29, 'Ricardo', 'Gutiérrez', '345678901Z', '123-789-4560', 'ricardo@email.com', '1996-09-03', 'Passeig de Gràcia 14, Barcelona', 2),
(30, 'Marta', 'López', '456789012AA', '555-987-6543', 'marta_l@email.com', '1985-05-14', 'Avenida da Liberdade 18, Lisboa', 5),
(31, 'Antonio', 'Martí', '567890123AB', '123-456-7890', 'antonio@email.com', '1990-08-22', 'Calle E, Madrid', 1),
(32, 'Julia', 'Müller', '678901234AC', '987-654-3210', 'julia@email.com', '1988-12-01', 'Praterstraße 5, Viena', 14),
(33, 'Giovanni', 'Ferrari', '789012345AD', '555-123-4567', 'giovanni@email.com', '1993-04-09', 'Via Garibaldi 3, Génova', 3),
(34, 'Catarina', 'Larsson', '901234567AE', '123-789-4560', 'catarina@email.com', '1987-01-15', 'Drottninggatan 5, Estocolmo', 7),
(35, 'Pietro', 'Ricci', '012345678AF', '555-987-6543', 'pietro@email.com', '1994-07-28', 'Via Roma 10, Milán', 3),
(36, 'Chen', 'Wang', '234567890AI', '555-123-4567', 'chen@email.com', '1990-09-21', 'Nanjing West Road, Shanghái', 32),
(37, 'Liping', 'Zhao', '1122334455AA', '123-456-7890', 'liping@email.com', '1990-03-12', 'Beijing Road 123, Pekín', 31),
(38, 'Cheng', 'Wu', '2233445566BB', '555-123-4567', 'cheng@email.com', '1988-07-25', 'Nanjing East Road 456, Shanghái', 32),
(39, 'Wei', 'Liu', '3344556677CC', '123-789-4560', 'wei@email.com', '1992-05-08', 'Guangzhou Street 789, Guangdong', 33),
(40, 'Xin', 'Chen', '4455667788DD', '555-987-6543', 'xin@email.com', '1989-12-20', 'Nanjing West Road 321, Jiangsu', 34),
(41, 'Jing', 'Zhou', '5566778899EE', '123-456-7890', 'jing@email.com', '1993-02-28', 'Hangzhou Road 654, Zhejiang', 35),
(42, 'Min', 'Li', '6677889900FF', '987-654-3210', 'min@email.com', '1987-09-03', 'Guangzhou Street 7, Guangdong', 33),
(43, 'Yun', 'Wang', '7788990011GG', '555-123-4567', 'yun@email.com', '1990-01-30', 'Nanjing East Road 1, Shanghái', 32),
(44, 'Xia', 'Zhang', '8899001122HH', '123-789-4560', 'xia@email.com', '1996-06-11', 'Beijing Road 123, Pekín', 31),
(45, 'Wei', 'Wu', '9900112233II', '555-987-6543', 'weiw@email.com', '1986-11-05', 'Nanjing West Road 321, Jiangsu', 34),
(46, 'Ling', 'Wang', '0011223344JJ', '123-456-7890', 'lingw@email.com', '1991-09-18', 'Hangzhou Road 654, Zhejiang', 35),
(47, 'Ravi', 'Kumar', '345678901AJ', '123-789-4560', 'ravi@email.com', '1985-11-13', 'Andheri East, Mumbai', 47),
(48, 'Amit', 'Sharma', '345672901AJ', '555-123-4567', 'amit@email.com', '1990-08-15', 'Andheri East, Mumbai', 47),
(49, 'Yuki', 'Nakamura', '567890123AL', '123-456-7890', 'yuki@email.com', '1989-02-09', 'Chuo, Tokio', 46),
(50, 'Akiko', 'Sato', '987654321AH', '987-654-3210', 'akiko@email.com', '1992-03-08', 'Shibuya, Tokio', 46),
(51, 'Luis', 'González', '1122334455PP', '123-456-7890', 'luis@email.com', '1990-04-15', 'Reforma 123, Ciudad de México', 40),
(52, 'Sofía', 'Hernández', '2233445566QQ', '555-123-4567', 'sofia@email.com', '1988-07-25', 'Guadalajara 456, Jalisco', 41),
(53, 'Carlos', 'Martínez', '3344556677RR', '123-789-4560', 'carlos_m@email.com', '1992-05-08', 'Polanco 789, Ciudad de México', 40),
(54, 'Ana', 'López', '4455667788SS', '555-987-6543', 'ana_l@email.com', '1989-12-20', 'Zapopan 321, Jalisco', 41),
(55, 'Fernando', 'Sánchez', '5566778899TT', '123-456-7890', 'fernando@email.com', '1993-02-28', 'Coyoacán 654, Ciudad de México', 40),
(56, 'Marta', 'Vargas', '6677889900UU', '555-123-4567', 'marta_v@email.com', '1990-01-30', 'Tlaquepaque 555, Jalisco', 41),
(57, 'Javier', 'Rodríguez', '7788990011VV', '123-789-4560', 'javier@email.com', '1987-09-03', 'Tlalpan 888, Ciudad de México', 40),
(58, 'Carlos', 'García', '567890123AP', '123-789-4560', 'carlos_ga@email.com', '1988-09-10', '789 Oak St, Ciudad de México', 40),
(59, 'Ana', 'Silva', '1122334455KK', '123-456-7890', 'ana_s@email.com', '1990-04-15', 'Avenida Paulista 123, Sao Paulo', 42),
(60, 'Pedro', 'Santos', '2233445566LL', '555-123-4567', 'pedro@email.com', '1988-07-25', 'Copacabana 456, Río de Janeiro', 43),
(61, 'María', 'Oliveira', '3344556677MM', '123-789-4560', 'maria@email.com', '1992-05-08', 'Rua Augusta 789, Sao Paulo', 42),
(62, 'Carlos', 'Lima', '4455667788NN', '555-987-6543', 'carlos_l@email.com', '1989-12-20', 'Ipanema 321, Río de Janeiro', 43),
(63, 'Lucia', 'Pereira', '5566778899OO', '123-456-7890', 'lucia@email.com', '1993-02-28', 'Avenida Brasil 654, Sao Paulo', 42),
(64, 'John', 'Smith', '123456789AN', '123-456-7890', 'john@email.com', '1986-04-15', '123 Main St, Nueva York', 37),
(65, 'Emily', 'Johnson', '987654321AO', '555-123-4567', 'emily_j@email.com', '1990-02-22', '456 Elm St, Los Ángeles', 36);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `clientes_idiomas`
--

CREATE TABLE `clientes_idiomas` (
  `id` int(11) NOT NULL,
  `id_cliente` int(11) NOT NULL,
  `id_idioma` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `clientes_idiomas`
--

INSERT INTO `clientes_idiomas` (`id`, `id_cliente`, `id_idioma`) VALUES
(1, 1, 1),
(2, 20, 1),
(3, 26, 1),
(4, 31, 1),
(5, 2, 1),
(6, 29, 1),
(7, 51, 1),
(8, 53, 1),
(9, 55, 1),
(10, 57, 1),
(11, 58, 1),
(12, 52, 1),
(13, 54, 1),
(14, 56, 1),
(15, 11, 2),
(16, 12, 2),
(17, 4, 3),
(18, 16, 3),
(19, 21, 3),
(20, 33, 3),
(21, 35, 3),
(22, 3, 3),
(23, 18, 3),
(24, 5, 4),
(25, 17, 4),
(26, 30, 4),
(27, 6, 4),
(28, 13, 4),
(29, 14, 4),
(30, 22, 4),
(31, 32, 4),
(32, 15, 4),
(33, 19, 4),
(34, 7, 5),
(35, 24, 5),
(36, 25, 5),
(37, 28, 5),
(38, 34, 5),
(39, 8, 5),
(40, 59, 5),
(41, 61, 5),
(42, 63, 5),
(43, 60, 5),
(44, 62, 5),
(45, 9, 6),
(46, 23, 6),
(47, 27, 6),
(48, 10, 6),
(49, 65, 6),
(50, 64, 6),
(51, 37, 13),
(52, 44, 13),
(53, 36, 13),
(54, 38, 13),
(55, 43, 13),
(56, 39, 13),
(57, 42, 13),
(58, 40, 13),
(59, 45, 13),
(60, 41, 13),
(61, 46, 13),
(62, 49, 14),
(63, 50, 14),
(64, 47, 16),
(65, 48, 16),
(128, 30, 12),
(129, 57, 17),
(130, 40, 12),
(131, 18, 5),
(132, 32, 12),
(133, 36, 14),
(134, 50, 16),
(135, 49, 15),
(136, 62, 12),
(137, 59, 11),
(138, 3, 6),
(139, 37, 11),
(140, 8, 3),
(141, 17, 16),
(142, 61, 12),
(143, 44, 6),
(144, 10, 16),
(145, 5, 15),
(146, 34, 13),
(147, 38, 3),
(148, 25, 1),
(149, 31, 15),
(150, 41, 2),
(151, 20, 8),
(152, 51, 8),
(153, 45, 8),
(154, 14, 8),
(155, 52, 2),
(156, 19, 2),
(157, 1, 8),
(158, 35, 12),
(159, 24, 1),
(160, 29, 13),
(161, 15, 9),
(162, 2, 7);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `continentes`
--

CREATE TABLE `continentes` (
  `id_continente` int(11) NOT NULL,
  `nombre` varchar(25) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `continentes`
--

INSERT INTO `continentes` (`id_continente`, `nombre`) VALUES
(1, 'África'),
(2, 'América'),
(3, 'Asia'),
(4, 'Europa'),
(5, 'Oceanía');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `descuentos`
--

CREATE TABLE `descuentos` (
  `codigo_descuento` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `valor_porcentual` decimal(5,2) NOT NULL,
  `descripcion` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `descuentos`
--

INSERT INTO `descuentos` (`codigo_descuento`, `nombre`, `valor_porcentual`, `descripcion`) VALUES
(1, 'Sin descuento', '0.00', 'En estos casos no aplica ningún descuento'),
(2, 'Cumpleaños', '5.00', 'Descuento del 5% en tu cumpleaños'),
(3, 'San Valentín', '15.00', 'Descuento del 14% para el Día de San Valentín'),
(4, 'Aniversario', '30.00', 'Descuento del 30% en el aniversario de la tienda'),
(5, 'Vuelta al Cole', '10.00', 'Descuento del 10% en artículos escolares'),
(6, 'Halloween', '10.00', 'Descuento del 18% para la noche de Halloween'),
(7, 'Black Friday', '25.00', 'Descuento del 25% en el evento Black Friday'),
(8, 'Navidad', '20.00', 'Descuento del 20% para celebrar la Navidad');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `detalle_factura`
--

CREATE TABLE `detalle_factura` (
  `id_detalle` int(11) NOT NULL,
  `id_factura` int(11) NOT NULL,
  `id_producto` int(11) NOT NULL,
  `precio_producto` decimal(10,2) NOT NULL,
  `unidades_compradas` int(11) NOT NULL,
  `codigo_descuento` int(11) NOT NULL,
  `codigo_iva` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `detalle_factura`
--

INSERT INTO `detalle_factura` (`id_detalle`, `id_factura`, `id_producto`, `precio_producto`, `unidades_compradas`, `codigo_descuento`, `codigo_iva`) VALUES
(1, 1, 1, '599.99', 2, 8, 1),
(2, 2, 2, '299.99', 1, 4, 2),
(3, 3, 3, '99.99', 4, 8, 3),
(4, 4, 4, '199.99', 1, 6, 1),
(5, 5, 5, '49.99', 3, 7, 2),
(6, 6, 6, '29.99', 2, 1, 3),
(7, 7, 7, '79.99', 1, 4, 1),
(8, 8, 8, '149.99', 2, 3, 2),
(9, 9, 9, '39.99', 3, 5, 3),
(10, 10, 10, '299.99', 1, 1, 1),
(11, 11, 11, '9.99', 10, 5, 2),
(12, 12, 12, '149.99', 1, 3, 3),
(13, 13, 13, '499.99', 1, 4, 1),
(14, 14, 14, '59.99', 2, 2, 2),
(15, 15, 15, '19.99', 5, 8, 3),
(16, 16, 16, '199.99', 2, 6, 1),
(17, 17, 17, '29.99', 1, 1, 2),
(18, 18, 18, '24.99', 3, 6, 3),
(19, 19, 19, '14.99', 5, 4, 1),
(20, 20, 20, '49.99', 1, 1, 2),
(21, 21, 21, '2.99', 20, 1, 3),
(22, 22, 22, '9.99', 2, 8, 1),
(23, 23, 23, '3.99', 10, 7, 2),
(24, 24, 24, '1.99', 30, 1, 3),
(25, 25, 25, '7.99', 4, 4, 1),
(26, 26, 26, '19.99', 1, 1, 2),
(27, 27, 27, '49.99', 2, 1, 3),
(28, 28, 28, '29.99', 1, 1, 1),
(29, 29, 29, '199.99', 1, 1, 2),
(30, 30, 30, '12.99', 3, 1, 3),
(31, 31, 31, '449.99', 1, 1, 1),
(32, 32, 32, '249.99', 2, 1, 2),
(33, 33, 33, '349.99', 1, 8, 3),
(34, 34, 34, '39.99', 2, 1, 1),
(35, 35, 35, '199.99', 1, 1, 2),
(36, 36, 36, '79.99', 1, 2, 3),
(37, 37, 37, '149.99', 2, 1, 1),
(38, 38, 38, '34.99', 3, 1, 2),
(39, 39, 39, '9.99', 5, 1, 3),
(40, 40, 40, '19.99', 1, 6, 1),
(41, 41, 41, '59.99', 2, 1, 1),
(42, 42, 42, '14.99', 3, 2, 2),
(43, 43, 43, '299.99', 1, 1, 3),
(44, 44, 44, '89.99', 1, 1, 1),
(45, 45, 45, '9.99', 10, 1, 2),
(46, 46, 46, '129.99', 1, 1, 3),
(47, 47, 47, '59.99', 2, 6, 1),
(48, 48, 48, '39.99', 1, 6, 2),
(49, 49, 49, '24.99', 4, 1, 3),
(50, 50, 50, '119.99', 1, 1, 1),
(51, 51, 51, '34.99', 2, 6, 2),
(52, 52, 52, '149.99', 1, 1, 3),
(53, 53, 53, '79.99', 1, 1, 1),
(54, 54, 54, '49.99', 2, 1, 2),
(55, 55, 55, '199.99', 1, 7, 3),
(56, 56, 56, '999.99', 1, 1, 1),
(57, 57, 57, '29.99', 4, 7, 2),
(58, 58, 58, '599.99', 1, 5, 3),
(59, 59, 59, '69.99', 1, 2, 1),
(60, 60, 60, '14.99', 2, 1, 2),
(61, 61, 61, '199.99', 1, 7, 3),
(62, 62, 62, '129.99', 1, 1, 1),
(63, 63, 63, '49.99', 3, 3, 2),
(64, 64, 64, '79.99', 1, 3, 3),
(65, 65, 65, '19.99', 2, 4, 1),
(66, 66, 66, '49.99', 1, 1, 2),
(67, 67, 67, '29.99', 2, 1, 3),
(68, 68, 68, '9.99', 3, 1, 1),
(69, 69, 69, '39.99', 1, 2, 2),
(70, 70, 70, '6.99', 10, 8, 3),
(71, 71, 71, '599.99', 1, 7, 1),
(72, 72, 72, '24.99', 2, 1, 2),
(73, 73, 8, '349.99', 1, 7, 1),
(74, 74, 11, '249.99', 2, 1, 2),
(75, 75, 14, '199.99', 1, 1, 3),
(76, 76, 15, '599.99', 1, 1, 1),
(77, 77, 18, '34.99', 4, 5, 2),
(78, 78, 21, '149.99', 1, 4, 3),
(79, 79, 25, '79.99', 1, 1, 1),
(80, 80, 27, '29.99', 2, 5, 2),
(81, 81, 31, '99.99', 1, 6, 3),
(82, 82, 33, '8.99', 10, 5, 1),
(83, 83, 36, '199.99', 1, 1, 2),
(84, 84, 39, '49.99', 2, 1, 3),
(85, 85, 42, '89.99', 1, 2, 1),
(86, 86, 46, '14.99', 3, 7, 2),
(87, 87, 48, '299.99', 1, 4, 3),
(88, 88, 50, '59.99', 2, 4, 1),
(89, 89, 54, '49.99', 1, 1, 2),
(90, 90, 57, '599.99', 1, 5, 3),
(91, 91, 60, '24.99', 2, 7, 1),
(92, 92, 63, '9.99', 3, 2, 2),
(93, 93, 66, '79.99', 1, 2, 3),
(94, 94, 68, '19.99', 2, 1, 1),
(95, 95, 71, '199.99', 1, 1, 2),
(96, 96, 74, '29.99', 2, 1, 3),
(97, 97, 77, '59.99', 1, 8, 1),
(98, 98, 80, '12.99', 4, 4, 2),
(99, 99, 6, '39.99', 3, 1, 2),
(100, 100, 9, '29.99', 2, 3, 1);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `facturas`
--

CREATE TABLE `facturas` (
  `id_factura` int(11) NOT NULL,
  `id_cliente` int(11) NOT NULL,
  `fecha` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `facturas`
--

INSERT INTO `facturas` (`id_factura`, `id_cliente`, `fecha`) VALUES
(1, 1, '2020-12-04'),
(2, 1, '2022-05-07'),
(3, 1, '2020-12-16'),
(4, 1, '2021-10-02'),
(5, 20, '2021-11-22'),
(6, 20, '2020-03-16'),
(7, 20, '2023-05-11'),
(8, 20, '2020-02-28'),
(9, 26, '2022-09-25'),
(10, 26, '2021-03-09'),
(11, 26, '2021-09-26'),
(12, 26, '2021-02-13'),
(13, 31, '2020-05-25'),
(14, 31, '2022-08-17'),
(15, 31, '2023-12-09'),
(16, 31, '2023-10-25'),
(17, 2, '2023-04-04'),
(18, 2, '2020-10-31'),
(19, 2, '2022-05-26'),
(20, 2, '2021-06-30'),
(21, 29, '2020-04-15'),
(22, 29, '2020-12-13'),
(23, 29, '2023-11-23'),
(24, 29, '2020-08-10'),
(25, 4, '2023-05-16'),
(26, 4, '2023-01-11'),
(27, 4, '2021-01-12'),
(28, 4, '2020-01-29'),
(29, 16, '2021-04-16'),
(30, 16, '2022-03-23'),
(31, 16, '2023-03-30'),
(32, 16, '2021-07-17'),
(33, 21, '2021-12-25'),
(34, 21, '2021-04-15'),
(35, 21, '2020-06-27'),
(36, 21, '2022-08-01'),
(37, 33, '2023-06-12'),
(38, 33, '2021-06-23'),
(39, 33, '2021-01-19'),
(40, 33, '2020-10-31'),
(41, 35, '2021-01-02'),
(42, 35, '2022-07-14'),
(43, 35, '2021-08-23'),
(44, 35, '2020-08-16'),
(45, 3, '2022-03-10'),
(46, 3, '2021-01-27'),
(47, 3, '2022-10-21'),
(48, 3, '2022-10-19'),
(49, 18, '2021-07-31'),
(50, 18, '2023-07-06'),
(51, 18, '2020-10-23'),
(52, 18, '2021-07-09'),
(53, 5, '2021-03-01'),
(54, 5, '2021-04-09'),
(55, 5, '2022-11-10'),
(56, 5, '2022-06-25'),
(57, 17, '2023-11-01'),
(58, 17, '2023-09-23'),
(59, 17, '2023-02-16'),
(60, 17, '2020-06-16'),
(61, 30, '2020-11-28'),
(62, 30, '2023-03-04'),
(63, 30, '2021-02-17'),
(64, 30, '2020-02-23'),
(65, 6, '2021-05-04'),
(66, 6, '2022-04-03'),
(67, 6, '2023-04-01'),
(68, 6, '2021-06-27'),
(69, 7, '2021-09-08'),
(70, 7, '2023-12-25'),
(71, 7, '2022-11-06'),
(72, 7, '2022-04-15'),
(73, 24, '2022-11-25'),
(74, 24, '2023-08-19'),
(75, 24, '2021-06-18'),
(76, 24, '2020-06-03'),
(77, 25, '2021-09-20'),
(78, 25, '2023-05-05'),
(79, 25, '2023-07-16'),
(80, 25, '2023-09-04'),
(81, 28, '2023-10-03'),
(82, 28, '2023-09-29'),
(83, 28, '2023-06-19'),
(84, 28, '2022-01-31'),
(85, 34, '2020-01-12'),
(86, 34, '2021-11-25'),
(87, 34, '2021-05-30'),
(88, 34, '2021-05-11'),
(89, 8, '2022-07-23'),
(90, 8, '2020-09-16'),
(91, 8, '2023-11-16'),
(92, 8, '2021-04-03'),
(93, 9, '2022-08-22'),
(94, 9, '2021-06-12'),
(95, 9, '2023-04-21'),
(96, 9, '2020-03-06'),
(97, 23, '2022-12-27'),
(98, 23, '2022-05-28'),
(99, 23, '2023-01-22'),
(100, 23, '2020-02-01');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `ficha_producto`
--

CREATE TABLE `ficha_producto` (
  `id_producto` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `categoria` int(11) NOT NULL,
  `fecha_lanzamiento` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `ficha_producto`
--

INSERT INTO `ficha_producto` (`id_producto`, `nombre`, `categoria`, `fecha_lanzamiento`) VALUES
(1, 'Smartphone Modelo X', 1, '2023-10-20'),
(2, 'Tablet Avanzada', 1, '2023-10-18'),
(3, 'Auriculares Inalámbricos', 1, '2023-10-15'),
(4, 'Cámara de Seguridad', 1, '2023-10-12'),
(5, 'Reproductor de Streaming', 1, '2023-10-10'),
(6, 'Camiseta de Algodón', 2, '2023-10-20'),
(7, 'Pantalones Vaqueros', 2, '2023-10-18'),
(8, 'Vestido de Noche', 2, '2023-10-15'),
(9, 'Zapatos Deportivos', 2, '2023-10-12'),
(10, 'Abrigo de Invierno', 2, '2023-10-10'),
(11, 'Juego de Tazas de Café', 3, '2023-10-20'),
(12, 'Manta de Sofá', 3, '2023-10-18'),
(13, 'Cuadro Decorativo', 3, '2023-10-15'),
(14, 'Cubiertos de Acero Inoxidable', 3, '2023-10-12'),
(15, 'Lámpara de Pie', 3, '2023-10-10'),
(16, 'Balón de Fútbol', 4, '2023-10-20'),
(17, 'Raqueta de Tenis', 4, '2023-10-18'),
(18, 'Bicicleta de Montaña', 4, '2023-10-15'),
(19, 'Ropa Deportiva', 4, '2023-10-12'),
(20, 'Cintas de Correr', 4, '2023-10-10'),
(21, 'Set de Construcción', 5, '2023-10-20'),
(22, 'Muñeca Interactiva', 5, '2023-10-18'),
(23, 'Juego de Mesa', 5, '2023-10-15'),
(24, 'Puzzle Infantil', 5, '2023-10-12'),
(25, 'Coche de Control Remoto', 5, '2023-10-10'),
(26, 'Caja de Cereales', 6, '2023-10-20'),
(27, 'Aceite de Oliva Extra Virgen', 6, '2023-10-18'),
(28, 'Chocolate Amargo', 6, '2023-10-15'),
(29, 'Sopa Instantánea', 6, '2023-10-12'),
(30, 'Café Orgánico', 6, '2023-10-10'),
(31, 'Crema Hidratante', 7, '2023-10-20'),
(32, 'Perfume Floral', 7, '2023-10-18'),
(33, 'Maquillaje de Lujo', 7, '2023-10-15'),
(34, 'Cepillo Eléctrico para el Cabello', 7, '2023-10-12'),
(35, 'Kit de Manicura', 7, '2023-10-10'),
(36, 'Lavadora de Carga Frontal', 8, '2023-10-20'),
(37, 'Horno de Convección', 8, '2023-10-18'),
(38, 'Frigorífico de Dos Puertas', 8, '2023-10-15'),
(39, 'Cafetera de Goteo', 8, '2023-10-12'),
(40, 'Aspiradora Robot', 8, '2023-10-10'),
(41, 'Set de Muebles de Patio', 9, '2023-10-20'),
(42, 'Cortadora de Césped', 9, '2023-10-18'),
(43, 'Juego de Sábanas', 9, '2023-10-15'),
(44, 'Manguera de Jardín', 9, '2023-10-12'),
(45, 'Cubiertos de Parrilla', 9, '2023-10-10'),
(46, 'Sofá de Tela', 10, '2023-10-20'),
(47, 'Mesa de Comedor', 10, '2023-10-18'),
(48, 'Silla de Oficina', 10, '2023-10-15'),
(49, 'Estantería de Madera', 10, '2023-10-12'),
(50, 'Cama King Size', 10, '2023-10-10'),
(51, 'Novela de Ciencia Ficción', 11, '2023-10-20'),
(52, 'Libro de Cocina', 11, '2023-10-18'),
(53, 'Libro de Poesía', 11, '2023-10-15'),
(54, 'Libro de Historia', 11, '2023-10-12'),
(55, 'Libro de Autoayuda', 11, '2023-10-10'),
(56, 'Anillo de Diamantes', 12, '2023-10-20'),
(57, 'Collar de Perlas', 12, '2023-10-18'),
(58, 'Reloj de Pulsera', 12, '2023-10-15'),
(59, 'Pendientes de Oro', 12, '2023-10-12'),
(60, 'Pulsera de Plata', 12, '2023-10-10'),
(61, 'Neumáticos Deportivos', 13, '2023-10-20'),
(62, 'Aceite de Motor Sintético', 13, '2023-10-18'),
(63, 'Asientos de Cuero', 13, '2023-10-15'),
(64, 'Sistema de Sonido Mejorado', 13, '2023-10-12'),
(65, 'Llantas de Aleación', 13, '2023-10-10'),
(66, 'Guitarra Acústica', 14, '2023-10-20'),
(67, 'Teclado Digital', 14, '2023-10-18'),
(68, 'Micrófono de Estudio', 14, '2023-10-15'),
(69, 'Amplificador de Guitarra', 14, '2023-10-12'),
(70, 'Discos de Vinilo', 14, '2023-10-10'),
(71, 'Juego de Destornilladores', 15, '2023-10-20'),
(72, 'Taladro Inalámbrico', 15, '2023-10-18'),
(73, 'Sierra Circular', 15, '2023-10-15'),
(74, 'Soldador Eléctrico', 15, '2023-10-12'),
(75, 'Lijadora de Banda', 15, '2023-10-10'),
(76, 'Báscula Digital', 16, '2023-10-20'),
(77, 'Masajeador de Espalda', 16, '2023-10-18'),
(78, 'Monitor de Presión Arterial', 16, '2023-10-15'),
(79, 'Suplementos Vitamínicos', 16, '2023-10-12'),
(80, 'Equipo de Ejercicio en Casa', 16, '2023-10-10'),
(81, 'Comida para Perros', 17, '2023-10-20'),
(82, 'Juguetes para Gatos', 17, '2023-10-18'),
(83, 'Collar y Correa para Perros', 17, '2023-10-15'),
(84, 'Cama para Mascotas', 17, '2023-10-12'),
(85, 'Comida para Gatos', 17, '2023-10-10'),
(86, 'Cámara DSLR Profesional', 18, '2023-10-20'),
(87, 'Trípode de Fotografía', 18, '2023-10-18'),
(88, 'Lente Gran Angular', 18, '2023-10-15'),
(89, 'Estudio de Fotografía Portátil', 18, '2023-10-12'),
(90, 'Álbum de Fotos', 18, '2023-10-10'),
(91, 'Set de Pinturas al Óleo', 19, '2023-10-20'),
(92, 'Kit de Scrapbooking', 19, '2023-10-18'),
(93, 'Tela de Lienzo', 19, '2023-10-15'),
(94, 'Herramientas de Escultura', 19, '2023-10-12'),
(95, 'Cajas de Colores', 19, '2023-10-10'),
(96, 'Televisor 4K', 20, '2023-10-20'),
(97, 'Blu-ray Discs', 20, '2023-10-18'),
(98, 'Proyector de Cine en Casa', 20, '2023-10-15'),
(99, 'Sistema de Sonido Envolvente', 20, '2023-10-12'),
(100, 'Películas Clásicas en DVD', 20, '2023-10-10'),
(101, 'Zapatillas Deportivas', 21, '2023-10-20'),
(102, 'Botas de Invierno', 21, '2023-10-18'),
(103, 'Zapatos de Vestir', 21, '2023-10-15'),
(104, 'Sandalias de Playa', 21, '2023-10-12'),
(105, 'Zapatos de Running', 21, '2023-10-10');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `idiomas`
--

CREATE TABLE `idiomas` (
  `id_idioma` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `descripcion` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `idiomas`
--

INSERT INTO `idiomas` (`id_idioma`, `nombre`, `descripcion`) VALUES
(1, 'Español', 'Idioma español'),
(2, 'Inglés', 'Idioma inglés'),
(3, 'Francés', 'Idioma francés'),
(4, 'Alemán', 'Idioma alemán'),
(5, 'Italiano', 'Idioma italiano'),
(6, 'Portugués', 'Idioma portugués'),
(7, 'Holandés', 'Idioma holandés'),
(8, 'Sueco', 'Idioma sueco'),
(9, 'Noruego', 'Idioma noruego'),
(10, 'Danés', 'Idioma danés'),
(11, 'Finlandés', 'Idioma finlandés'),
(12, 'Ruso', 'Idioma ruso'),
(13, 'Chino', 'Idioma chino'),
(14, 'Japonés', 'Idioma japonés'),
(15, 'Coreano', 'Idioma coreano'),
(16, 'Hindi', 'Idioma de la India'),
(17, 'Árabe', 'Idioma arabe');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `iva`
--

CREATE TABLE `iva` (
  `codigo_iva` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `valor_porcentual` decimal(5,2) NOT NULL,
  `descripcion` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `iva`
--

INSERT INTO `iva` (`codigo_iva`, `nombre`, `valor_porcentual`, `descripcion`) VALUES
(1, 'IVA General (21%)', '21.00', 'Tipo de IVA general aplicable a la mayoría de bienes y servicios.'),
(2, 'IVA Reducido (10%)', '10.00', 'Tipo de IVA reducido aplicable a bienes y servicios específicos, como alimentos, productos culturales, etc.'),
(3, 'IVA Superreducido (4%)', '4.00', 'Tipo de IVA superreducido aplicable a bienes y servicios esenciales, como alimentos básicos, libros, medicamentos, etc.'),
(4, 'IVA Exento (0%)', '0.00', 'Tipo de IVA exento, que implica que no se aplica IVA a ciertos bienes o servicios.'),
(5, 'IVA Recargo de Equivalencia (5.2%)', '5.20', 'Tipo de IVA con recargo de equivalencia aplicable a comerciantes minoristas.');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `paises`
--

CREATE TABLE `paises` (
  `id_pais` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `bandera` text NOT NULL,
  `id_idioma_principal` int(11) NOT NULL,
  `num_habitantes` int(11) NOT NULL,
  `id_continente` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `paises`
--

INSERT INTO `paises` (`id_pais`, `nombre`, `bandera`, `id_idioma_principal`, `num_habitantes`, `id_continente`) VALUES
(1, 'España', 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Flag_of_Spain.svg/40px-Flag_of_Spain.svg.png', 1, 46754778, 4),
(2, 'Francia', 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Flag_of_France_%281794%E2%80%931815%2C_1830%E2%80%931974%29.svg/40px-Flag_of_France_%281794%E2%80%931815%2C_1830%E2%80%931974%29.svg.png', 3, 65273511, 4),
(3, 'Alemania', 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Flag_of_Germany.svg/40px-Flag_of_Germany.svg.png', 4, 83783942, 4),
(4, 'Italia', 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Flag_of_Italy.svg/40px-Flag_of_Italy.svg.png', 5, 60461826, 4),
(5, 'Portugal', 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Flag_of_Portugal.svg/40px-Flag_of_Portugal.svg.png', 6, 10291196, 4),
(6, 'Inglaterra', 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Flag_of_England.svg/40px-Flag_of_England.svg.png', 2, 68207116, 4),
(7, 'Suecia', 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Flag_of_Sweden.svg/40px-Flag_of_Sweden.svg.png', 4, 10230185, 4),
(8, 'Noruega', 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Flag_of_Norway.svg/40px-Flag_of_Norway.svg.png', 4, 5421240, 4),
(9, 'Dinamarca', 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Flag_of_Denmark.svg/40px-Flag_of_Denmark.svg.png', 4, 5806015, 4),
(10, 'Finlandia', 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Flag_of_Finland.svg/40px-Flag_of_Finland.svg.png', 4, 5540720, 4),
(11, 'Países Bajos', 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Flag_of_the_Netherlands.svg/40px-Flag_of_the_Netherlands.svg.png', 7, 17134872, 4),
(12, 'Bélgica', 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Flag_of_Belgium_%28civil%29.svg/40px-Flag_of_Belgium_%28civil%29.svg.png', 2, 11589623, 4),
(13, 'Suiza', 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Flag_of_Switzerland.svg/40px-Flag_of_Switzerland.svg.png', 1, 8717539, 4),
(14, 'Austria', 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_Austria.svg/40px-Flag_of_Austria.svg.png', 1, 9006398, 4),
(15, 'Grecia', 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Flag_of_Greece.svg/40px-Flag_of_Greece.svg.png', 1, 10715549, 4),
(16, 'Estados Unidos', 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Flag_of_the_United_States.svg/40px-Flag_of_the_United_States.svg.png', 6, 331915073, 2),
(17, 'Canadá', 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Flag_of_Canada.svg/40px-Flag_of_Canada.svg.png', 2, 38041000, 2),
(18, 'México', 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/Flag_of_Mexico.svg/40px-Flag_of_Mexico.svg.png', 1, 126190788, 2),
(19, 'Brasil', 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Flag_of_Brazil.svg/40px-Flag_of_Brazil.svg.png', 5, 212559417, 2),
(20, 'Australia', 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Flag_of_Australia_%28converted%29.svg/40px-Flag_of_Australia_%28converted%29.svg.png', 6, 25499884, 5),
(21, 'China', 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Flag_of_the_People%27s_Republic_of_China.svg/40px-Flag_of_the_People%27s_Republic_of_China.svg.png', 13, 1444216107, 3),
(22, 'Japón', 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Flag_of_Japan.svg/40px-Flag_of_Japan.svg.png', 14, 125964511, 3),
(23, 'India', 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/40px-Flag_of_India.svg.png', 16, 1380004385, 3),
(24, 'Egipto', 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Flag_of_Egypt.svg/40px-Flag_of_Egypt.svg.png', 17, 100388073, 1);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `provincias`
--

CREATE TABLE `provincias` (
  `id_provincia` int(11) NOT NULL,
  `nombre` varchar(255) NOT NULL,
  `pais` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Volcado de datos para la tabla `provincias`
--

INSERT INTO `provincias` (`id_provincia`, `nombre`, `pais`) VALUES
(1, 'Madrid', 1),
(2, 'Barcelona', 1),
(3, 'Île-de-France', 2),
(4, "Provence-Alpes-Côte d\'Azur", 2),
(5, 'Berlín', 3),
(6, 'Hamburgo', 3),
(7, 'Lombardía', 4),
(8, 'Lacio', 4),
(9, 'Lisboa', 5),
(10, 'Oporto', 5),
(11, 'Inglaterra', 6),
(12, 'Escocia', 6),
(13, 'Estocolmo', 7),
(14, 'Gotemburgo', 7),
(15, 'Oslo', 8),
(16, 'Trondheim', 8),
(17, 'Copenhague', 9),
(18, 'Aarhus', 9),
(19, 'Helsinki', 10),
(20, 'Tampere', 10),
(21, 'Ámsterdam', 11),
(22, 'Róterdam', 11),
(23, 'Bruselas', 12),
(24, 'Amberes', 12),
(25, 'Zúrich', 13),
(26, 'Ginebra', 13),
(27, 'Viena', 14),
(28, 'Salzburgo', 14),
(29, 'Atenas', 15),
(30, 'Salónica', 15),
(31, 'Pekín', 21),
(32, 'Shanghái', 21),
(33, 'Guangdong', 21),
(34, 'Jiangsu', 21),
(35, 'Zhejiang', 21),
(36, 'Los Ángeles', 16),
(37, 'Nueva York', 16),
(38, 'Quebec', 17),
(39, 'Ontario', 17),
(40, 'Ciudad de México', 18),
(41, 'Jalisco', 18),
(42, 'Sao Paulo', 19),
(43, 'Río de Janeiro', 19),
(44, 'Nueva Gales del Sur', 20),
(45, 'Victoria', 20),
(46, 'Tokio', 22),
(47, 'Mumbai', 23);

--
-- Índices para tablas volcadas
--

--
-- Indices de la tabla `categorias_productos`
--
ALTER TABLE `categorias_productos`
  ADD PRIMARY KEY (`id_categoria`),
  ADD UNIQUE KEY `nombre` (`nombre`);

--
-- Indices de la tabla `clientes`
--
ALTER TABLE `clientes`
  ADD PRIMARY KEY (`id_cliente`),
  ADD UNIQUE KEY `doc_identidad` (`doc_identidad`),
  ADD UNIQUE KEY `email` (`email`),
  ADD KEY `provincia` (`provincia`);

--
-- Indices de la tabla `clientes_idiomas`
--
ALTER TABLE `clientes_idiomas`
  ADD PRIMARY KEY (`id`),
  ADD KEY `id_cliente` (`id_cliente`),
  ADD KEY `id_idioma` (`id_idioma`);

--
-- Indices de la tabla `continentes`
--
ALTER TABLE `continentes`
  ADD PRIMARY KEY (`id_continente`),
  ADD UNIQUE KEY `nombre` (`nombre`);

--
-- Indices de la tabla `descuentos`
--
ALTER TABLE `descuentos`
  ADD PRIMARY KEY (`codigo_descuento`),
  ADD UNIQUE KEY `nombre` (`nombre`);

--
-- Indices de la tabla `detalle_factura`
--
ALTER TABLE `detalle_factura`
  ADD PRIMARY KEY (`id_detalle`),
  ADD KEY `id_factura` (`id_factura`),
  ADD KEY `id_producto` (`id_producto`),
  ADD KEY `codigo_descuento` (`codigo_descuento`),
  ADD KEY `codigo_iva` (`codigo_iva`);

--
-- Indices de la tabla `facturas`
--
ALTER TABLE `facturas`
  ADD PRIMARY KEY (`id_factura`),
  ADD KEY `id_cliente` (`id_cliente`);

--
-- Indices de la tabla `ficha_producto`
--
ALTER TABLE `ficha_producto`
  ADD PRIMARY KEY (`id_producto`),
  ADD UNIQUE KEY `nombre` (`nombre`),
  ADD KEY `categoria` (`categoria`);

--
-- Indices de la tabla `idiomas`
--
ALTER TABLE `idiomas`
  ADD PRIMARY KEY (`id_idioma`),
  ADD UNIQUE KEY `nombre` (`nombre`);

--
-- Indices de la tabla `iva`
--
ALTER TABLE `iva`
  ADD PRIMARY KEY (`codigo_iva`),
  ADD UNIQUE KEY `nombre` (`nombre`);

--
-- Indices de la tabla `paises`
--
ALTER TABLE `paises`
  ADD PRIMARY KEY (`id_pais`),
  ADD UNIQUE KEY `nombre` (`nombre`),
  ADD KEY `id_continente` (`id_continente`),
  ADD KEY `id_idioma_principal` (`id_idioma_principal`);

--
-- Indices de la tabla `provincias`
--
ALTER TABLE `provincias`
  ADD PRIMARY KEY (`id_provincia`),
  ADD UNIQUE KEY `nombre` (`nombre`),
  ADD KEY `pais` (`pais`);

--
-- AUTO_INCREMENT de las tablas volcadas
--

--
-- AUTO_INCREMENT de la tabla `categorias_productos`
--
ALTER TABLE `categorias_productos`
  MODIFY `id_categoria` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=22;

--
-- AUTO_INCREMENT de la tabla `clientes`
--
ALTER TABLE `clientes`
  MODIFY `id_cliente` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=66;

--
-- AUTO_INCREMENT de la tabla `clientes_idiomas`
--
ALTER TABLE `clientes_idiomas`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=163;

--
-- AUTO_INCREMENT de la tabla `continentes`
--
ALTER TABLE `continentes`
  MODIFY `id_continente` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT de la tabla `descuentos`
--
ALTER TABLE `descuentos`
  MODIFY `codigo_descuento` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT de la tabla `detalle_factura`
--
ALTER TABLE `detalle_factura`
  MODIFY `id_detalle` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=101;

--
-- AUTO_INCREMENT de la tabla `facturas`
--
ALTER TABLE `facturas`
  MODIFY `id_factura` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=101;

--
-- AUTO_INCREMENT de la tabla `ficha_producto`
--
ALTER TABLE `ficha_producto`
  MODIFY `id_producto` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=106;

--
-- AUTO_INCREMENT de la tabla `idiomas`
--
ALTER TABLE `idiomas`
  MODIFY `id_idioma` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=18;

--
-- AUTO_INCREMENT de la tabla `iva`
--
ALTER TABLE `iva`
  MODIFY `codigo_iva` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT de la tabla `paises`
--
ALTER TABLE `paises`
  MODIFY `id_pais` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=25;

--
-- AUTO_INCREMENT de la tabla `provincias`
--
ALTER TABLE `provincias`
  MODIFY `id_provincia` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=48;

--
-- Restricciones para tablas volcadas
--

--
-- Filtros para la tabla `clientes`
--
ALTER TABLE `clientes`
  ADD CONSTRAINT `clientes_ibfk_1` FOREIGN KEY (`provincia`) REFERENCES `provincias` (`id_provincia`);

--
-- Filtros para la tabla `clientes_idiomas`
--
ALTER TABLE `clientes_idiomas`
  ADD CONSTRAINT `clientes_idiomas_ibfk_1` FOREIGN KEY (`id_cliente`) REFERENCES `clientes` (`id_cliente`),
  ADD CONSTRAINT `clientes_idiomas_ibfk_2` FOREIGN KEY (`id_idioma`) REFERENCES `idiomas` (`id_idioma`);

--
-- Filtros para la tabla `detalle_factura`
--
ALTER TABLE `detalle_factura`
  ADD CONSTRAINT `detalle_factura_ibfk_1` FOREIGN KEY (`id_factura`) REFERENCES `facturas` (`id_factura`),
  ADD CONSTRAINT `detalle_factura_ibfk_2` FOREIGN KEY (`id_producto`) REFERENCES `ficha_producto` (`id_producto`),
  ADD CONSTRAINT `detalle_factura_ibfk_3` FOREIGN KEY (`codigo_descuento`) REFERENCES `descuentos` (`codigo_descuento`),
  ADD CONSTRAINT `detalle_factura_ibfk_4` FOREIGN KEY (`codigo_iva`) REFERENCES `iva` (`codigo_iva`);

--
-- Filtros para la tabla `facturas`
--
ALTER TABLE `facturas`
  ADD CONSTRAINT `facturas_ibfk_1` FOREIGN KEY (`id_cliente`) REFERENCES `clientes` (`id_cliente`);

--
-- Filtros para la tabla `ficha_producto`
--
ALTER TABLE `ficha_producto`
  ADD CONSTRAINT `ficha_producto_ibfk_1` FOREIGN KEY (`categoria`) REFERENCES `categorias_productos` (`id_categoria`);

--
-- Filtros para la tabla `paises`
--
ALTER TABLE `paises`
  ADD CONSTRAINT `paises_ibfk_1` FOREIGN KEY (`id_continente`) REFERENCES `continentes` (`id_continente`),
  ADD CONSTRAINT `paises_ibfk_2` FOREIGN KEY (`id_idioma_principal`) REFERENCES `idiomas` (`id_idioma`);

--
-- Filtros para la tabla `provincias`
--
ALTER TABLE `provincias`
  ADD CONSTRAINT `provincias_ibfk_1` FOREIGN KEY (`pais`) REFERENCES `paises` (`id_pais`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
