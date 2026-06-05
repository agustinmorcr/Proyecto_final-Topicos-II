import os
import cv2 as cv, cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from skimage import  io, filters, color
from skimage.measure import euler_number

# =====================================================
# 1. CARGAR IMAGEN 2D
# =====================================================
def cargar_imagen():
    """
    Carga una imagen desde el disco y la transforma en una matriz binaria estricta de ceros (fondo) y unos (objeto)
    """
    root = tk.Tk()
    root.withdraw() 
    ruta = filedialog.askopenfilename(
        title="Seleccionar Imagen", 
        filetypes=[("Imagen(*.png *.jpg *.gif *.tif)", "*.png *.jpg *.gif *.tif")]
    )
    
    print("\nCargando modelo...")
    if ruta:
        nombre_base = os.path.splitext(os.path.basename(ruta))[0]
        print(f"\nNombre del Modelo: {nombre_base}\n")
        # Cargar imagen en escala de grises
        imagen = io.imread(ruta)
        #Aplanar dimensiones
        imagen = np.squeeze(imagen)
        # Si es imagen a color, pasar a escala de grises
        if imagen.ndim == 3:
            # Si tiene canal alfa, pasar a RGB
            if imagen.shape[-1] == 4:
                imagen = color.rgba2rgb(imagen)
            # Convertir a escala de grises
            imagen = color.rgb2gray(imagen)
        # Encontrar el umbral de separación (Otsu)
        umbral = filters.threshold_otsu(imagen)
        # Binarizar la imagen con Otsu (True/False convertido a 0/1)
        imagen_binaria = (imagen > umbral).astype(np.uint8)
        return imagen, imagen_binaria, nombre_base
    return None, None, None

# =====================================================
# 2. CALCULO DE SIMPLEJOS
# =====================================================
def calcular_simplejos(M):
    """
    Calcula de forma apropiada los simplejos en 2D (n0, n1, n2) utilizando el enfoque de Perímetro de Contacto
    """
    # ------------------------------------------------------
    # DIMENSIÓN 2 (n2): CARAS / ÁREA (A)
    # ------------------------------------------------------
    # Suma todos los elementos de la matriz. Como el objeto vale 1 y el fondo 0, el resultado es exactamente el número total de píxeles activos (Área 'A').
    n2 = int(np.sum(M))
    A = n2 
    
    # ------------------------------------------------------
    # DIMENSIÓN 1 (n1): ARISTAS
    # ------------------------------------------------------
    # Evaluamos las adyacencias utilizando operaciones lógicas AND (&) entre píxeles vecinos.
    
    # M[:, :-1] toma toda la matriz excepto la última columna.
    # M[:, 1:] toma toda la matriz excepto la primera columna (desplazada a la derecha).
    # Al aplicar & entre ambas, solo da 1 donde dos píxeles horizontales adyacentes están activos.
    contactos_horizontales = np.sum(M[:, :-1] & M[:, 1:])

    # M[:-1, :] toma toda la matriz excepto la última fila.
    # M[1:, :] toma toda la matriz excepto la primera fila (desplazada hacia abajo).
    # Al aplicar & entre ambas, encuentra los píxeles adyacentes de forma vertical.
    contactos_verticales = np.sum(M[:-1, :] & M[1:, :])

    # El Perímetro de Contacto (Pc) es la suma de todas las fronteras internas compartidas.
    Pc = contactos_horizontales + contactos_verticales
    
    # Ecuación fundamental: Cada píxel aislado aportaría 4 aristas (4*A), pero cada contacto interno une dos píxeles compartiendo una sola arista, por lo que restamos Pc
    n1 = 4 * A - Pc
    
    # ------------------------------------------------------
    # DIMENSIÓN 0 (n0): VÉRTICES
    # ------------------------------------------------------
    # Obtenemos las dimensiones (alto y ancho) de la matriz de la imagen.
    H, W = M.shape
    n0 = 0 # Inicializamos el contador de vértices en cero

    # Para contar los vértices en una rejilla discreta, analizamos las "cruces" o esquinas.
    # Una rejilla de píxeles de H x W tiene exactamente (H + 1) x (W + 1) intersecciones de esquinas.
    for r in range(H + 1):
        for c in range(W + 1):
            # Para cada intersección (r, c), identificamos las 4 celdas que confluyen en ese punto.
            # Se usan condiciones 'if' para asegurar que si estamos en los bordes exteriores de la imagen, no intente leer fuera de la matriz (asumiendo que fuera hay fondo = 0).
            p1 = M[r-1, c-1] if (r > 0 and c > 0) else 0  # Píxel arriba a la izquierda
            p2 = M[r-1, c]   if (r > 0 and c < W) else 0  # Píxel arriba a la derecha
            p3 = M[r, c-1]   if (r < H and c > 0) else 0  # Píxel abajo a la izquierda
            p4 = M[r, c]     if (r < H and c < W) else 0  # Píxel abajo a la derecha
            
            # Si al menos uno de los 4 píxeles alrededor de esta intersección pertenece al objeto (1), significa que este punto de la cuadrícula es un vértice real del objeto unificado.
            if (p1 or p2 or p3 or p4):
                n0 += 1 # Registramos este vértice único en el contador
                
    return n0, n1, n2

# =====================================================
# 3. Betti
# =====================================================
def calcular_betti(matriz):
    contours, hierarchy = cv2.findContours(matriz, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    num_objetos = 0
    num_agujeros = 0
    if hierarchy is not None:
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1: # No tiene padre, es objeto
                num_objetos += 1
            else: # Tiene padre, es agujero
                num_agujeros += 1
    return num_objetos, num_agujeros

# =====================================================
# 4. Euler
# =====================================================
def calcular_euler(imagen_binaria):
    # Calcular la característica de Euler
    euler_2d = euler_number(imagen_binaria, connectivity=2)
    # Imprimir resultado
    return euler_2d

# =====================================================
# 5. EULER EN VCC Y 3OT
# =====================================================
# -----------------------------
# F4
# -----------------------------
def encontrar_inicio(img):
    # Busca un pixel de borde:
    # uno que tenga fondo arriba o a la izquierda

    padded = np.pad(img, 1, 'constant', constant_values=0)

    for y in range(1, padded.shape[0] - 1):
        for x in range(1, padded.shape[1] - 1):
            if padded[y, x] == 255:
                if padded[y - 1, x] == 0 or padded[y, x - 1] == 0:
                    return x, y
    return None

def cadena_f4(img_binaria):
    # Genera el código de Freeman en 4 direcciones
    # siguiendo el contorno pixel a pixel

    # Direcciones: 0→, 1↓, 2←, 3↑
    inv = {
        0: (1, 0),
        1: (0, 1),
        2: (-1, 0),
        3: (0, -1)
    }

    padded = np.pad(img_binaria, 1, 'constant', constant_values=0)

    start = encontrar_inicio(img_binaria)
    if start is None:
        return []

    x0, y0 = start
    x, y = x0, y0
    d = 0  # dirección inicial

    cadena = []

    # Recorrido del contorno
    for _ in range(10000):
        dx, dy = inv[d]
        x += dx
        y += dy

        cadena.append(d)

        # Si regresa al inicio, termina
        if (x, y) == (x0, y0):
            break

        # Intentar girar a la izquierda
        d = (d + 3) % 4

        # Buscar siguiente dirección válida
        for _ in range(4):
            dx, dy = inv[d]

            # Selección del pixel a revisar según dirección
            if d == 0:
                px, py = x, y
            elif d == 1:
                px, py = x - 1, y
            elif d == 2:
                px, py = x - 1, y - 1
            else:
                px, py = x, y - 1

            # Si es parte del objeto, continuar
            if padded[py, px] == 255:
                break

            # Si no, girar a la derecha
            d = (d + 1) % 4

    return cadena

# -----------------------------
# VCC
# -----------------------------
def cadena_vcc(f4):
    # Codifica cambios de dirección en F4 (vertical, horizontal, giro)

    tabla = {
        (0, 0): 0, (0, 1): 1, (0, 3): 2,
        (1, 0): 2, (1, 1): 0, (1, 2): 1,
        (2, 1): 2, (2, 2): 0, (2, 3): 1,
        (3, 0): 1, (3, 2): 2, (3, 3): 0
    }

    return [tabla.get((f4[i - 1], f4[i]), 0) for i in range(len(f4))]

# -----------------------------
# 3OT
# -----------------------------
def cadena_3ot(f4):
    # Clasifica cambios respecto a una dirección de referencia

    if len(f4) < 2:
        return []

    c = []
    n2_convexo = 0
    n2_concavo = 0
    ref = f4[0]
    sup = f4[0]
    cambio = False

    for i in range(1, len(f4)):
        x = f4[i]

        if x == sup:
            c.append(0)
        else:
            if not cambio:
                c.append(2)
                cambio = True
                giro = (x - sup) % 4
                if giro == 1:
                    n2_convexo += 1
                elif giro == 3:
                    n2_concavo += 1
            elif x == ref:
                c.append(1)
                ref = sup
            elif (x - ref) % 4 == 2:
                c.append(2)
                giro = (sup - ref) % 4
                if giro == 1:
                    n2_convexo += 1
                elif giro == 3:
                    n2_concavo += 1
                ref = sup
            else:
                c.append(1)
                ref = sup

        sup = x

    # cierre circular
    x = f4[0]

    if x == sup:
        c.append(0)
    elif not cambio:
        c.append(2)
        giro = (x - sup) % 4
        if giro == 1:
            n2_convexo += 1
        elif giro == 3:
            n2_concavo += 1
    elif x == ref:
        c.append(1)
    elif (x - ref) % 4 == 2:
        c.append(2)
        giro = (sup - ref) % 4
        if giro == 1:
            n2_convexo += 1
        elif giro == 3:
            n2_concavo += 1
    else:
        c.append(1)

    return (c, n2_convexo, n2_concavo)

# --------------------------------
# Relación entre VCC, 3OT y Euler
# --------------------------------
def relacion_euler_cadenas(vcc, n2_convexo, n2_concavo):
    
    # Cálculo de euler a través de VCC
    euler_vcc = (vcc.count(1) - vcc.count(2))/4

    # Cálculo de euler a través de 3OT
    
    euler_ot3 = (n2_convexo - n2_concavo)/4

    return euler_vcc, euler_ot3

# ======================================================
# 6. VISUALIZACIÓN
# ======================================================
def visualizar(nom, img, v, e, f, cc, h, euler, euler_vcc, euler_ot3):
    """ Genera la ventana de resultados con gráficas y métricas """

    # Creamos la figura con un poco más de espacio abajo para el texto
    fig, (ax_img, ax_txt) = plt.subplots(
        1, 2,
        figsize=(12, 6),
        gridspec_kw={'width_ratios': [3, 1]}
    )

    # Imagen
    ax_img.imshow(img, cmap='gray')
    ax_img.set_title(f"Nombre: {nom}", fontsize=25)
    ax_img.axis('off')

    # Texto
    texto_reporte = (
        f"Vértices (n0): {v}\n"
        f"Aristas (n1): {e}\n"
        f"Caras (n2): {f}\n"
        "──────────────────────────────\n"
        f"Componente conectado (B0): {cc}\n"
        f"Agujeros / Hoyos (B1): {h}\n"
        "──────────────────────────────\n"
        f"Euler (X): {euler}\n"
        f"Euler Total (VCC): {euler_vcc}\n"
        f"Euler Total (3OT): {euler_ot3}"
    )

    ax_txt.axis('off')
    ax_txt.text(
        0.05, 0.95,
        texto_reporte,
        transform=ax_txt.transAxes,
        va='top',
        fontsize=20,
        family='monospace',
        bbox=dict(facecolor='lightgray', alpha=0.8)
    )

    plt.tight_layout()
    plt.show()

# =====================================================
# 7. EJECUCIÓN
# =====================================================
if __name__ == "__main__":
    img, img_binaria, nombre = cargar_imagen()

    if img is not None:
        v, e, f = calcular_simplejos(img_binaria)
        euler = calcular_euler(img_binaria)

        cc, h = calcular_betti(img_binaria)

        contornos, jerarquia = cv.findContours(img_binaria, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        
        euler_vcc_total = 0.0
        euler_ot3_total = 0.0
        for i, contorno in enumerate(contornos):
            lienzo = np.zeros_like(img_binaria)
            cv.drawContours(lienzo, contornos, i, 255, -1)
            f4 = cadena_f4(lienzo)
            vcc = cadena_vcc(f4)
            ot3, n2_conv, n2_conc = cadena_3ot(f4)
            euler_vcc_ind, euler_ot3_ind = relacion_euler_cadenas(vcc, n2_conv, n2_conc)
            padre = jerarquia[0][i][3]
            if padre == -1:
                euler_vcc_total += euler_vcc_ind
                euler_ot3_total += euler_ot3_ind
            else:
                euler_vcc_total -= euler_vcc_ind
                euler_ot3_total -= euler_ot3_ind

        visualizar(nombre, img, v, e, f, cc, h, euler, euler_vcc_total, euler_ot3_total)