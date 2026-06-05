import os
import trimesh
import numpy as np
import pyvista as pv
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import label
from skimage.measure import euler_number

# =====================================================
# 1. CARGAR MODELO 3D
# =====================================================
def cargar_modelo3D():
    root = tk.Tk()
    root.withdraw() 
    ruta = filedialog.askopenfilename(
        title="Seleccionar Modelo 3D", 
        filetypes=[("Archivos(*.stl *.obj *.ply)", "*.stl *.obj *.ply")]
    )
    
    print("Cargando modelo...")
    if ruta:
        nombre_base = os.path.splitext(os.path.basename(ruta))[0]
        modelo = trimesh.load(ruta)

        if isinstance(modelo, trimesh.Scene):
            modelo = trimesh.util.concatenate(tuple(modelo.geometry.values()))

        if not hasattr(modelo, 'vertices'):
            raise ValueError("El archivo no contiene una malla válida.")

        print("\n====== INFORMACIÓN DEL MODELO ======")
        print(f" Nombre del Modelo: {nombre_base}")
        print(f" ¿Malla cerrada?: {modelo.is_watertight}")
        print(f"====================================")

        return modelo, nombre_base
    return None, None

# =====================================================
# 2. VOXELIZACIÓN
# =====================================================
def voxalizar(modelo, nombre):
    print("\nVoxelizando modelo...")
    min_bound, max_bound = modelo.bounds
    tamano = np.max(max_bound - min_bound)

    # Rejilla adaptativa cúbica uniforme para estabilidad analítica
    pitch = tamano / 100
    print(f"Pitch automático: {pitch:.4f}")

    vox = modelo.voxelized(pitch)
    
    directorio = filedialog.askdirectory(title="Selecciona la carpeta para guardar tus archivos")
    if directorio:
        nombre_archivo = f"Voxel_{nombre}.obj"
        ruta = os.path.join(directorio, nombre_archivo)
        malla_voxelizada = vox.as_boxes()
        malla_voxelizada.export(ruta)

        malla = trimesh.load(ruta)
        # Extraer las coordenadas espaciales de los puntos
        puntos = malla.vertices
        # Desplazar todos los puntos para que empiecen en (0,0,0) 
        puntos_desplazados = puntos - puntos.min(axis=0)
        indices = np.round(puntos_desplazados).astype(int)
        # Calcular el tamaño exacto del contenedor
        dimensiones = indices.max(axis=0) + 1
        # Crear una matriz 3D llena de False (ceros)
        matriz_3d = np.zeros(dimensiones, dtype=bool)
        # Encender a True las posiciones exactas
        matriz_3d[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        return matriz_3d, vox
    return None, None

# =====================================================
# 3. CALCULO DE SIMPLEJOS
# =====================================================
def calcular_simplejos_3d(matriz):
    """
    Calcula apropiadamente los simplejos (Vértices, Aristas, Caras, Volúmenes)
    de un objeto binario voxelizado en 3D basándose en las propiedades locales
    de una rejilla cúbica.
    """
    # Asegurar que la matriz sea de tipo booleano o entero binario
    M = matriz.astype(bool)
    
    # -----------------------------------------------------------------
    # 3-SIMPLEJO (c3): Son simplemente los voxeles activos (Volúmenes)
    # -----------------------------------------------------------------
    c3 = np.sum(M)
    
    # Para evaluar los elementos compartidos de manera eficiente, extendemos la matriz
    # con un acolchado (padding) para no perder las fronteras exteriores del objeto.
    pad_M = np.pad(M, 1, mode='constant', constant_values=False)
    
    # -----------------------------------------------------------------
    # 2-SIMPLEJO (c2): Caras
    # Cada voxel activo aporta inicialmente 6 caras unitarias orientadas.
    # Las caras en las tres direcciones principales se encuentran desplazando la matriz.
    # -----------------------------------------------------------------
    # Filtramos para contar solo las caras que pertenecen o tocan a la estructura activa
    # Una cara existe si al menos uno de los dos voxeles adyacentes que divide está activo
    c2 = (np.sum(pad_M[1:, 1:-1, 1:-1] | pad_M[:-1, 1:-1, 1:-1]) +  # Dirección X
          np.sum(pad_M[1:-1, 1:, 1:-1] | pad_M[1:-1, :-1, 1:-1]) +  # Dirección Y
          np.sum(pad_M[1:-1, 1:-1, 1:] | pad_M[1:-1, 1:-1, :-1]))   # Dirección Z

    # -----------------------------------------------------------------
    # 1-SIMPLEJO (c1): Aristas
    # Una arista existe en el espacio si al menos uno de los 4 voxeles
    # que convergen alrededor de ella está activo.
    # -----------------------------------------------------------------
    # Aristas paralelas al eje X (compartidas por combinaciones en Y y Z)
    aristas_x = (pad_M[1:-1, :-1, :-1] | pad_M[1:-1, 1:, :-1] | 
                 pad_M[1:-1, :-1, 1:]  | pad_M[1:-1, 1:, 1:])
    
    # Aristas paralelas al eje Y (compartidas por combinaciones en X y Z)
    aristas_y = (pad_M[:-1, 1:-1, :-1] | pad_M[1:, 1:-1, :-1] | 
                 pad_M[:-1, 1:-1, 1:]  | pad_M[1:, 1:-1, 1:])
    
    # Aristas paralelas al eje Z (compartidas por combinaciones en X y Y)
    aristas_z = (pad_M[:-1, :-1, 1:-1] | pad_M[1:, :-1, 1:-1] | 
                 pad_M[:-1, 1:, 1:-1]  | pad_M[1:, 1:, 1:-1])
    
    c1 = np.sum(aristas_x) + np.sum(aristas_y) + np.sum(aristas_z)

    # -----------------------------------------------------------------
    # 0-SIMPLEJO (c0): Vértices
    # Un vértice existe en la malla si al menos uno de los 8 voxeles 
    # que se intersectan en esa esquina está activo.
    # -----------------------------------------------------------------
    vertices = (
        pad_M[:-1, :-1, :-1] | pad_M[1:, :-1, :-1] | pad_M[:-1, 1:, :-1] | pad_M[1:, 1:, :-1] |
        pad_M[:-1, :-1, 1:]  | pad_M[1:, :-1, 1:]  | pad_M[:-1, 1:, 1:]  | pad_M[1:, 1:, 1:]
    )
    c0 = np.sum(vertices)
    
    return c0, c1, c2, c3

# =====================================================
# 4. CALCULO DE TUNELES Y CAVIDADES
# =====================================================
def analizar_topologia_3d(matriz):
    """
    Analiza un objeto voxelizado 3D y devuelve de forma exacta y desagregada:
    b0 (Componentes), b1 (Túneles) y b2 (Cavidades).
    """
    # Definimos la estructura estándar de 6-conectividad ortogonal (cara compartida)
    estructura_6 = np.zeros((3, 3, 3), dtype=bool)
    estructura_6[1, 1, :] = True
    estructura_6[1, :, 1] = True
    estructura_6[:, 1, 1] = True
    
    # -------------------------------------------------------------------------
    # CALCULAR b0: Componentes conexas del objeto sólido
    # -------------------------------------------------------------------------
    _, b0 = label(matriz, structure=estructura_6)
    
    # -------------------------------------------------------------------------
    # CALCULAR b2: Cavidades herméticas (Teorema de Dualidad de Alexander)
    # -------------------------------------------------------------------------
    # Añadimos un padding generoso de 2 capas de aire (0) para que el fondo
    # pueda fluir libremente a través de cualquier túnel hacia el exterior.
    matriz_expandida = np.pad(matriz, 2, mode='constant', constant_values=0)
    
    # Invertimos la matriz expandida: Objeto = False, Fondo = True
    matriz_inversa = ~matriz_expandida.astype(bool)
    
    # Etiquetamos todas las regiones vacías (de fondo) independientes
    fondo_etiquetado, num_componentes_fondo = label(matriz_inversa, structure=estructura_6)
    
    # La coordenada [0, 0, 0] es obligatoriamente el "océano de aire" exterior
    etiqueta_exterior = fondo_etiquetado[0, 0, 0]
    
    # Una cavidad verdadera es cualquier burbuja de fondo que NO logró 
    # conectarse con la etiqueta exterior en [0,0,0]
    b2 = 0
    for idx in range(1, num_componentes_fondo + 1):
        if idx != etiqueta_exterior:
            b2 += 1
            
    # -------------------------------------------------------------------------
    # CALCULAR b1: Túneles mediante la Característica de Euler Global
    # -------------------------------------------------------------------------
    # Calculamos los simplejos locales sobre la matriz original
    c0, c1, c2, c3 = calcular_simplejos_3d(matriz)
    chi = c0 - c1 + c2 - c3
    
    # Aplicamos la fórmula de Euler-Poincaré extendida: Chi = b0 - b1 + b2
    # Despejando b1: b1 = b0 + b2 - Chi
    b1 = b0 + b2 - chi
    
    return b0, b1, b2

# =====================================================
# 5. EULER
# =====================================================
def calcular_euler(matriz):
    # Calcular la Característica de Euler (Objetos - Túneles + Cavidades)
    # connectivity=3 evalúa los 26 vecinos posibles de un vóxel
    euler_3d = euler_number(matriz, connectivity=3)
    # Imprimir resultado
    return euler_3d

# =====================================================
# 6. SIMPLEJOS TETRAVOXEL Y OCTOVOXEL
# =====================================================
def simplejos_3d(matriz_3d):
    # Asegurar que la matriz sea booleana
    m = matriz_3d.astype(bool)
    print ("\n---Conteo de simplejos---\n")
    
    # Voxeles
    vox_num = np.sum(m)
    
    # Bivoxeles
    n_b_x = np.sum(m[:-1, :, :] & m[1:, :, :])
    n_b_y = np.sum(m[:, :-1, :] & m[:, 1:, :])
    n_b_z = np.sum(m[:, :, :-1] & m[:, :, 1:])
    num_biv = n_b_x + n_b_y + n_b_z
    
    # Tetravoxeles
    n_t_xy = np.sum(m[:-1, :-1, :] & m[1:, :-1, :] & m[:-1, 1:, :] & m[1:, 1:, :])
    n_t_xz = np.sum(m[:-1, :, :-1] & m[1:, :, :-1] & m[:-1, :, 1:] & m[1:, :, 1:])
    n_t_yz = np.sum(m[:, :-1, :-1] & m[:, 1:, :-1] & m[:, :-1, 1:] & m[:, 1:, 1:])
    num_tetra = n_t_xy + n_t_xz + n_t_yz
    
    # Octovoxeles
    num_octo = np.sum(m[:-1, :-1, :-1] & m[1:, :-1, :-1] & m[:-1, 1:, :-1] & m[:-1, :-1, 1:] & m[1:, 1:, :-1] & m[1:, :-1, 1:] & m[:-1, 1:, 1:] & m[1:, 1:, 1:])
    
    return vox_num, num_biv, num_tetra, num_octo

# =====================================================
# 7. Visualización
# =====================================================
def visualizar(nombre, original, voxel, c0, c1, c2, c3, b0, b1, b2, euler, v, b, t, o):
    plotter = pv.Plotter(shape="2|1", window_size=[1600, 900])

    # Imagen izquierda
    plotter.subplot(0)
    plotter.add_mesh(
        pv.wrap(original),
        color="lightblue"
    )
    plotter.camera.zoom(1.4)
    plotter.add_text("Modelo Original")

    # Imagen derecha
    plotter.subplot(1)
    plotter.add_mesh(
        pv.wrap(voxel.as_boxes()),
        color="orange"
    )
    plotter.camera.zoom(1.4)
    plotter.add_text("Voxelizado")

    # Panel inferior
    plotter.subplot(2)

    texto = (
        "------------------------------------------------------\n"
        f"{nombre}\n"
        "------------------------------------------------------\n"
        f"c0-Simplejos (Vértices): {c0}\n"
        f"c1-Simplejos (Aristas): {c1}\n"
        f"c2-Simplejos (Caras): {c2}\n"
        f"c3-Simplejos (Volúmenes): {c3}\n"
        "------------------------------------------------------\n"
        f"Componentes conectadas (β0): {b0}\n"
        f"Túneles (β1): {b1}\n"
        f"Cavidades (β2): {b2}\n"
        f"Euler (X): {euler}\n"
        "------------------------------------------------------\n"
        f"Voxeles: {v}\n"
        f"Bivoxeles: {b}\n"
        f"Tetravoxeles: {t}\n"
        f"Octovoxeles: {o}\n"
        "------------------------------------------------------\n"
    )

    plotter.add_text(
        texto,
        position="upper_edge",
        font_size=14
    )

    plotter.show()

# =====================================================
# 8. EJECUCIÓN
# =====================================================
if __name__ == "__main__":
    modelo, nombre = cargar_modelo3D()
    if modelo is not None:
        matriz_vox, vox = voxalizar(modelo, nombre)

        c0, c1, c2, c3 = calcular_simplejos_3d(matriz_vox)
        b0, b1, b2 = analizar_topologia_3d(matriz_vox)
        euler =  calcular_euler(matriz_vox)
        v, b, t, o = simplejos_3d(matriz_vox)

        visualizar(nombre, modelo, vox, c0, c1, c2, c3, b0, b1, b2, euler, v, b, t, o)

        print("\nAnálisis finalizado.")