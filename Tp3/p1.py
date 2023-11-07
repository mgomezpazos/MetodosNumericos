import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

#Ejercicio 1.1/2?-------------------------------------------------------------------------------------------------------------------------------------------------------------------
directorio_imagenes = ['img00.jpeg', 'img01.jpeg', 'img02.jpeg', 'img03.jpeg', 'img04.jpeg', 'img05.jpeg', 'img06.jpeg', 'img07.jpeg', 'img08.jpeg',
                       'img09.jpeg', 'img10.jpeg', 'img11.jpeg', 'img12.jpeg', 'img13.jpeg', 'img14.jpeg', 'img15.jpeg']

# procesamos las fotos
imagenes = []
for img_path in directorio_imagenes:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Leer la imagen en escala de grises
    if img is not None:
        imagenes.append(img)

# Convertimos las imágenes a matrices y las apilamos
matrices_imagenes = [img.flatten() for img in imagenes]
matriz_datos = np.stack(matrices_imagenes) 

# Aplicamos SVD a la matriz original
U, S, VT = np.linalg.svd(matriz_datos, full_matrices=False)

#Representación considerando las primeras k columnas----------------------------------------------------------------
k_values = [5, 10, 15]

num_imagenes = len(imagenes)  # Definir el número de imágenes

fig, axs = plt.subplots(len(k_values), num_imagenes, figsize=(20, 8))

for i, k in enumerate(k_values):
    representacion_baja_dimension_primeras_dimensiones_k = U[:, :k] @ np.diag(S[:k])
    imagen_reconstruida_primeras_k = representacion_baja_dimension_primeras_dimensiones_k @ VT[:k, :]

    for j in range(num_imagenes):
        axs[i, j].imshow(imagenes[j], cmap='gray')
        axs[i, j].axis('off')
        reconstruccion_k = imagen_reconstruida_primeras_k[j, :].reshape(imagenes[j].shape)
        axs[i, j].imshow(reconstruccion_k, cmap='gray')
        axs[i, j].axis('off')

plt.show()

#Representación considerando las ultimas k columnas-----------------------------------------------------------------
k_values = [5, 10, 15]

num_imagenes = len(imagenes)

fig, axs = plt.subplots(len(k_values), num_imagenes, figsize=(20, 8))

for i, k in enumerate(k_values):
    representacion_baja_dimension_ultimas_dimensiones_k = U[:, -k:] @ np.diag(S[:k])
    imagen_reconstruida_ultimas_k = representacion_baja_dimension_ultimas_dimensiones_k @ VT[-k:, :]

    for j in range(num_imagenes):
        axs[i, j].imshow(imagenes[j], cmap='gray')
        axs[i, j].axis('off')
        reconstruccion_k = imagen_reconstruida_ultimas_k[j, :].reshape(imagenes[j].shape)
        axs[i, j].imshow(reconstruccion_k, cmap='gray')
        axs[i, j].axis('off')

plt.show()

# Comparación entre la foto original y la reconstruida con los primeros k=5, 10 y 15 solo para la primera imagen
fig, axs = plt.subplots(1, len(k_values) + 1, figsize=(20, 6))

# Mostrar la imagen original
axs[0].imshow(imagenes[0], cmap='gray')
axs[0].axis('off')

for i, k in enumerate(k_values):
    representacion_baja_dimension_primeras_k = U[:, :k] @ np.diag(S[:k])
    imagen_reconstruida_k = representacion_baja_dimension_primeras_k @ VT[:k, :]
    reconstruccion_k = imagen_reconstruida_k[0, :].reshape(imagenes[0].shape)

    axs[i+1].imshow(reconstruccion_k, cmap='gray')
    axs[i+1].axis('off')

plt.show()

# Comparación entre la foto original y la reconstruida con los últimos k=5, 10 y 15 solo para la primera imagen
fig, axs = plt.subplots(1, len(k_values) + 1, figsize=(20, 6))

# Mostrar la imagen original
axs[0].imshow(imagenes[0], cmap='gray')
axs[0].axis('off')

for i, k in enumerate(k_values):
    representacion_baja_dimension_ultimas_k = U[:, -k:] @ np.diag(S[-k:])
    imagen_reconstruida_k = representacion_baja_dimension_ultimas_k @ VT[-k:, :]
    reconstruccion_k = imagen_reconstruida_k[0, :].reshape(imagenes[0].shape)

    axs[i+1].imshow(reconstruccion_k, cmap='gray')
    axs[i+1].axis('off')

plt.show()

#Ejercicio 1.2?------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Mapa de calor para la matriz S original (valores singulares)
axs[0].imshow(np.diag(S), cmap='viridis', aspect='auto')
axs[0].set_xlabel('Componentes')
axs[0].set_ylabel('Componentes')
axs[0].grid(False)
axs[0].axis('on')

# Mapa de calor para la matriz VT original (transpuesta de V)
axs[1].imshow(VT, cmap='viridis', aspect='auto')
axs[1].set_xlabel('Componentes')
axs[1].set_ylabel('Características')
axs[1].grid(False)
axs[1].axis('on')

# Mapa de calor para la matriz U original
axs[2].imshow(U, cmap='viridis', aspect='auto')
axs[2].set_xlabel('Componentes')
axs[2].set_ylabel('Imágenes')
axs[2].grid(False)
axs[2].axis('on')

plt.tight_layout()
plt.show()

U_reconstruido, S_reconstruido, VT_reconstruido = np.linalg.svd(representacion_baja_dimension_ultimas_k, full_matrices=False)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Mapa de calor para la matriz S reconstruida (baja dimensión)
axs[0].imshow(np.diag(S_reconstruido), cmap='viridis', aspect='auto')
axs[0].set_xlabel('Componentes')
axs[0].set_ylabel('Componentes')
axs[0].grid(False)
axs[0].axis('on')

# Mapa de calor para la matriz U reconstruida (baja dimensión)
axs[2].imshow(U_reconstruido, cmap='viridis', aspect='auto')
axs[2].set_xlabel('Componentes')
axs[2].set_ylabel('Imágenes')
axs[2].grid(False)
axs[2].axis('on')

# Mapa de calor para la matriz VT reconstruida (baja dimensión)
axs[1].imshow(VT_reconstruido, cmap='viridis', aspect='auto')
axs[1].set_xlabel('Componentes')
axs[1].set_ylabel('Características')
axs[1].grid(False)
axs[1].axis('on')

plt.tight_layout()
plt.show()

#Ejercicio 1.3 ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Crear una lista para almacenar las similitudes para diferentes valores de d
similaridades = []

# Rango de valores de d que quieres probar
valores_d = [5, 10, 25]

for d in valores_d:
    # Aplicar la SVD con el valor de d actual
    U, S, VT = np.linalg.svd(matriz_datos, full_matrices=False)
    representacion_baja_dimension_ultimas_dimensiones = U[:, :d] @ np.diag(S[:d])

    # Calcular la similaridad entre pares de imágenes en el espacio de baja dimensión d
    sim_matrix = cosine_similarity(representacion_baja_dimension_ultimas_dimensiones)
    similaridades.append(sim_matrix)

# Visualizar la similaridad entre imágenes para distintos valores de d
fig, axs = plt.subplots(1, len(valores_d), figsize=(18, 6))

for i, sim_matrix in enumerate(similaridades):
    ax = axs[i]
    im = ax.imshow(sim_matrix, cmap='viridis')
    ax.set_xlabel("Imágenes")
    ax.set_ylabel("Imágenes")
    ax.grid(False)
    
plt.subplots_adjust(wspace=0.5)  # Ajusta el espacio entre subtramas
plt.show()


#1.4---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Cargar una imagen del conjunto (por ejemplo, la primera imagen)
imagen_original = cv2.imread('img00.jpeg', cv2.IMREAD_GRAYSCALE)

# Apilar la misma imagen 16 veces para formar un conjunto de imágenes
n_imagenes = 16
imagenes_apiladas = np.stack([imagen_original] * n_imagenes)

# Calcular la compresión a distintas dimensiones y encontrar la mínima 'd' con un error inferior al 10%
error_umbral = 0.1  # 10% de error

U, S, VT = np.linalg.svd(imagenes[0], full_matrices=False)
d_optimo = 0
V_optimo=[]
Vt_optimo_a=[]
for d in range(1, 29):
    representacion_baja_dim = U[:, :d] @ np.diag(S[:d]) @ VT[:d, :]
    error = np.linalg.norm(imagenes[0] - representacion_baja_dim, 'fro') / np.linalg.norm(imagenes[0], 'fro')
    if error < error_umbral and d_optimo == 0:
        d_optimo = d
        VT_optimo = VT[:d, :]
        Vt_optimo_a=VT_optimo
        V_optimo = VT_optimo.T

print(V_optimo.shape)
# multiplicar por V_optimo cada imagen
representaciones_baja_dim = []

for imagen_original in imagenes:
    representacion_baja_dim = imagen_original @ V_optimo
    representacion_baja_dim = representacion_baja_dim @ Vt_optimo_a
    representaciones_baja_dim.append(representacion_baja_dim)

#plot all img_V
fig, axs = plt.subplots(4, 4, figsize=(20, 10))
for i in range(4):
    for j in range(4):
        axs[i, j].imshow(representaciones_baja_dim[i*4+j], cmap='gray')
        print(representaciones_baja_dim[i*4+j].shape)
        axs[i, j].axis('off')
        
plt.show()