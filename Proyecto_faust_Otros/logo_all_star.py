from PIL import Image, ImageDraw, ImageColor
from collections import Counter
import os # NUEVO: Importamos os para validar rutas

def generar_bandera_grande(ruta_logo, salida="bandera_logo_grande.png"):
    try:
        if not os.path.exists(ruta_logo): # NUEVO: Validación rápida de la ruta
            raise FileNotFoundError(f"El archivo no existe: {ruta_logo}")

        # 1. Cargar el logo
        print(f"Cargando logo desde: {ruta_logo}")
        logo = Image.open(ruta_logo).convert("RGBA")
        ancho_logo, alto_logo = logo.size
        print(f"Logo cargado: {ancho_logo}x{alto_logo} px")

        # 2. Configuración de la Bandera
        alto_bandera = alto_logo * 5
        ancho_bandera = int(alto_bandera * 1.5)
        alto_franja = alto_bandera // 5

        # 3. Detección de Colores
        logo_small = logo.resize((150, 150))
        colores = logo_small.getcolors(maxcolors=25000)
        colores_ordenados = sorted(colores, key=lambda x: x[0], reverse=True)
        colores_solidos = [c[1] for c in colores_ordenados if c[1][3] == 255]

        color_negro = (0, 0, 0) 
        color_rojo = (255, 0, 0)
        
        for c in colores_solidos:
            r, g, b, a = c
            if r < 40 and g < 40 and b < 40: color_negro = (r, g, b)
            if r > 100 and g < 100 and b < 100: color_rojo = (r, g, b)
        
        print(f"Colores -> Negro: {color_negro}, Rojo: {color_rojo}")

        # 4. Crear el lienzo de la bandera
        bandera = Image.new("RGB", (ancho_bandera, alto_bandera), color_negro)
        draw = ImageDraw.Draw(bandera)

        # Dibujar franjas rojas (índices 1 y 3)
        draw.rectangle([(0, alto_franja), (ancho_bandera, alto_franja * 2)], fill=color_rojo)
        draw.rectangle([(0, alto_franja * 3), (ancho_bandera, alto_franja * 4)], fill=color_rojo)

        # 5. Colocar el Logo GRANDE (NUEVO: Cambios en esta sección)
        # Queremos que ocupe las 3 franjas centrales (Roja-Negra-Roja)
        # Altura total disponible = 3 franjas
        margen = int(alto_franja * 0.1) # Mantenemos un pequeño margen de seguridad
        
        # NUEVO: La altura objetivo es ahora de 3 franjas menos el margen
        alto_objetivo = (alto_franja * 3) - (margen * 2)
        
        factor_escala = alto_objetivo / alto_logo
        ancho_objetivo = int(ancho_logo * factor_escala)

        logo_redimensionado = logo.resize((ancho_objetivo, alto_objetivo), Image.Resampling.LANCZOS)

        # NUEVO: Posición vertical para que empiece en la 2da franja
        # La posición Y inicial es el fin de la 1ra franja + margen
        x_pos = (ancho_bandera - ancho_objetivo) // 2
        y_pos = (alto_franja * 1) + margen 

        bandera.paste(logo_redimensionado, (x_pos, y_pos), logo_redimensionado)

        # 6. Guardar
        bandera.save(salida, dpi=(300, 300))
        print(f"\n¡Éxito! Imagen generada con el logo GRANDE como: {salida}")
        # NUEVO: Intenta abrir la imagen automáticamente para que veas el resultado
        os.startfile(salida) 

    except FileNotFoundError as fe:
        print(f"\nERROR RUTA: {fe}")
    except Exception as e:
        print(f"\nERROR GENERAL: {e}")

# --- LLAMADA A LA FUNCIÓN ---
if __name__ == "__main__":
    # ¡IMPORTANTE! Asegúrate de que esta ruta sea correcta antes de ejecutar
    ruta_del_escudo = "C:/Users/insau/Downloads/Escudo vectorizado.png"
    
    generar_bandera_grande(ruta_del_escudo)