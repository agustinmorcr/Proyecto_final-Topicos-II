# Proyecto_final-Tópicos-II Viso.on Nuevos Paradigmas
En este proyecto se explora la característica de Euler, tanto para 2D como para 3D. Dentro del repositorio se encuentran dos scripts de Python, uno de ellos para realizar los cálculos del número de Euler, números de Betti y cálculo de simplejos para imágenes 2D. El segundo script permite voxelizar objetos 3D, además de calcular el número de Euler, números de Betti, simplejos y simplejos cúbicos.
También se incluyen dos carpetas con los archivos necesarios para realizar pruebas. En una de ellas se incluyen 3 imágenes 2D en formato .gif, y en la otra se incluyen 3 objetos 3D en formato .stl. Dependiendo del script que se ejecute, EulerFinal2D o EulerFinal3D, son los archivos que se deben cargar para realizar los cálculos.

## Manual de usuario
Para iniciar, se deben descargar los dos scripts .py y las carpetas con las imágenes y objetos que se encuentran dentro de este repositorio.
Posteriormente se ejecuta el script que se desee probar primero con algún un IDE, el recomendado es Visual Studio Code (VSC) y la versión de Python 3.14.2.
Para ejecutar los scripts, serán necesarias algunas librerías para que funcionen correctamente. 

- En el caso de EulerFianl2D.py, las librerías necesarias son:
  - **os**           Instalada por defecto en VSC.
  - **cv2**          Comando para instalación: *pip install opencv-python*.
  - **numpy**        Comando para instalación: *pip install numpy*.
  - **tkinter**      Instalada por defecto en VSC. 
  - **matplotlib**   Comando para instalación: *pip install matplotlib*.
  - **skimage**      Comando para instalación: *pip install scikit-image*.
        
- En el caso de EulerFianl3D.py, las librerías necesarias son:
  - **os**           Instalada por defecto en VSC. 
  - **trimesh**      Comando para instalación: *pip install trimesh*.
  - **numpy**        Comando para instalación: *pip install numpy*.
  - **pyvista**      Comando para instalación: *pip install pyvista*.
  - **tkinter**      Instalada por defecto en VSC.
  - **scipy**        Comando para instalación: *pip install scipy*.
  - **skimage**      Comando para instalación: *pip install scikit-image*.

__*Nota:__ En caso de que el comando *pip install* no funcione, puede internar con el comando *pip3 install*. Esto normalmente es necesario en sistemas operativos mac o distribuciones de linux.

Una vez instaladas las librerías necesarias, para correr el programa puede ejecutar el comando *python EulerFinal2D.py* o *python EulerFinal3D.py* dependiendo del programa que desee probar, seguido de eso podrá observar la ventana de explorador de archivos, ahí busca la ruta en la que haya descargado el repositorio y seleccione el archivo que desee analizar (.gif para 2D y .stl para 3D). 
En el caso de 3D, le aparecerá otra ventana de explorador de archivos pidiendole la ruta en la que desee guardar el nuevo objeto voxelizado creado, simplemente seleccione su ruta y de click en aceptar, en esa misma ruta podrá consultar ese archivo creado.

Posteriormente a la ejecución del script, podrá observar una ventana en donde se muestre el objeto seleccionado (en el caso del 3D podrá observar tanto el objeto original como el voxelizado, además de que podrá escalarlos, rotarlos y trasladarlos para analizar mejor su morfología) además de los cálculos del número de euler, simplejos, simplejos cúbicos y números de Betti.  

__*Nota:__ En caso de que el comando *python* no funcione para ejecutar el script, puede internar con el comando *python3*. Esto normalmente es necesario en sistemas operativos mac o distribuciones de linux.
