import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

#1.1
# Directorio que contiene las imágenes
directorio_imagenes = ['img00.jpeg', 'img01.jpeg', 'img02.jpeg', 'img03.jpeg', 'img04.jpeg', 'img05.jpeg', 'img06.jpeg', 'img07.jpeg', 'img08.jpeg',
                       'img09.jpeg', 'img10.jpeg', 'img11.jpeg', 'img12.jpeg', 'img13.jpeg', 'img14.jpeg', 'img15.jpeg']

# Cargar y procesar las imágenes
imagenes = []
for img_path in directorio_imagenes:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Leer la imagen en escala de grises
    if img is not None:
        imagenes.append(img)

# Convertir imágenes a matrices y apilarlas
matrices_imagenes = [img.flatten() for img in imagenes]
matriz_datos = np.stack(matrices_imagenes)  # Asegurarse de usar np.stack para crear una matriz bidimensional

# Aplicar la SVD a la matriz original
U, S, VT = np.linalg.svd(matriz_datos, full_matrices=False)

# Nueva representación de baja dimensión (por ejemplo, considerando las primeras k columnas)
k = 25  # Número de componentes principales a mantener (ajusta según tu necesidad)
representacion_baja_dimension = U[:, :k] @ np.diag(S[:k])

# Reconstruir las imágenes a partir de la representación de baja dimensión
imagen_reconstruida = representacion_baja_dimension @ VT[:k, :]

# Mostrar las imágenes originales y las reconstrucciones lado a lado
num_imagenes = len(imagenes)
fig, axs = plt.subplots(2, num_imagenes, figsize=(20, 8))

# Título para la fila de imágenes originales
titulo_original = axs[0, 0].set_title('Imágenes Originales', fontsize=12)
titulo_original.set_position([1.1, 2.5])
axs[0, 0].axis('off')

# Título para la fila de imágenes reconstruidas
titulo_reconstruido = axs[1, 0].set_title('Reconstruidas (k = {})'.format(k), fontsize=12)
titulo_reconstruido.set_position([1.1, 2.5])
axs[1, 0].axis('off')

for i in range(num_imagenes):
    axs[0, i].imshow(imagenes[i], cmap='gray')
    axs[0, i].axis('off')
    
    # Reajustar la forma de la imagen reconstruida a la original y mostrarla
    reconstruccion = imagen_reconstruida[i, :].reshape(imagenes[i].shape)
    axs[1, i].imshow(reconstruccion, cmap='gray')
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()

# Visualizar solo la primera imagen y su reconstrucción en más detalle
fig_individual, axs_individual = plt.subplots(1, 2, figsize=(8, 4))

# Título para la imagen original individual
titulo_original_individual = axs_individual[0].set_title('Imagen Original', fontsize=10)
axs_individual[0].imshow(imagenes[0], cmap='gray')
axs_individual[0].axis('off')

# Reajustar la forma de la imagen reconstruida a la original y mostrarla
reconstruccion_individual = imagen_reconstruida[0, :].reshape(imagenes[0].shape)
titulo_reconstruido_individual = axs_individual[1].set_title('Reconstrucción (k = {})'.format(k), fontsize=10)
axs_individual[1].imshow(reconstruccion_individual, cmap='gray')
axs_individual[1].axis('off')

plt.tight_layout()
plt.show()

#1.2
# Mapa de calor para la matriz S (valores singulares)
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz S')
plt.imshow(np.diag(S), cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Componentes')
plt.show()

# Mapa de calor para la matriz VT (transpuesta de V)
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz VT')
plt.imshow(VT, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Características')
plt.show()

# Mapa de calor para la matriz U
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz U')
plt.imshow(U, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Imágenes')
plt.show()

U_reconstruido, S_reconstruido, VT_reconstruido = np.linalg.svd(representacion_baja_dimension, full_matrices=False)


# Mapa de calor para la matriz U (baja dimensión)
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz U (baja dimensión)')
plt.imshow(U_reconstruido, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Imágenes')
plt.show()

# Mapa de calor para la matriz S (baja dimensión)
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz S (baja dimensión)')
plt.imshow(np.diag(S_reconstruido), cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Componentes')
plt.show()

# Mapa de calor para la matriz VT (baja dimensión)
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz VT (baja dimensión)')
plt.imshow(VT_reconstruido, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Características')
plt.show()


#1.3
# Crear una lista para almacenar las similitudes para diferentes valores de d
similaridades = []

# Rango de valores de d que quieres probar
valores_d = [5, 10, 25]

for d in valores_d:
    # Aplicar la SVD con el valor de d actual
    U, S, VT = np.linalg.svd(matriz_datos, full_matrices=False)
    representacion_baja_dimension = U[:, :d] @ np.diag(S[:d])

    # Calcular la similaridad entre pares de imágenes en el espacio de baja dimensión d
    sim_matrix = cosine_similarity(representacion_baja_dimension)
    similaridades.append(sim_matrix)

# Visualizar la similaridad entre imágenes para distintos valores de d
fig, axs = plt.subplots(1, len(valores_d), figsize=(18, 6))

for i, sim_matrix in enumerate(similaridades):
    ax = axs[i]
    im = ax.imshow(sim_matrix, cmap='viridis')
    ax.set_title(f'Similaridad (d = {valores_d[i]})')
    ax.set_xlabel("Imágenes")
    ax.set_ylabel("Imágenes")
    ax.grid(False)
    
plt.subplots_adjust(wspace=0.5)  # Ajusta el espacio entre subtramas
plt.show()

#1.4?
# Cargar la imagen
imagen_elegida = cv2.imread('img00.jpeg', cv2.IMREAD_GRAYSCALE)

# Descomposición SVD para la imagen elegida
U, S, VT = np.linalg.svd(imagen_elegida, full_matrices=False)

# Calcular el error para distintos valores de d
for d in range(1, len(S) + 1):
    # Representación de baja dimensión
    representacion_baja_dimension = U[:, :d] @ np.diag(S[:d]) @ VT[:d, :]

    # Calcular el error entre la imagen original y la reconstruida
    error = np.linalg.norm(imagen_elegida - representacion_baja_dimension, 'fro') / np.linalg.norm(imagen_elegida, 'fro')

    # Si el error es menor al 10%, se ha alcanzado el valor mínimo de d
    if error < 0.1:
        print(f"El valor de d que mantiene el error por debajo del 10% es: {d}")
        break
    