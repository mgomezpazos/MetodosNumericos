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
k = 5  # Número de autovalores que se mantendrán
representacion_baja_dimension_primeras_dimensiones = U[:, :k] @ np.diag(S[:k])
# Reconstruimos las imágenes
imagen_reconstruida_primeras = representacion_baja_dimension_primeras_dimensiones @ VT[:k, :]

# Ploteos de las imágenes reconstruidas con las primeras k columnas
num_imagenes = len(imagenes)
fig, axs = plt.subplots(2, num_imagenes, figsize=(20, 8))

for i in range(num_imagenes):
    axs[0, i].imshow(imagenes[i], cmap='gray')
    axs[0, i].axis('off')
    reconstruccion = imagen_reconstruida_primeras[i, :].reshape(imagenes[i].shape)
    axs[1, i].imshow(reconstruccion, cmap='gray')
    axs[1, i].axis('off')

# Añadir títulos a las subtramas
fig.suptitle("Imágenes Originales y Reconstruidas con Primeras Dimensiones k=5", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#Representación considerando las ultimas k columnas-----------------------------------------------------------------
k = 5  # Número de autovalores que se mantendrán
representacion_baja_dimension_ultimas_dimensiones = U[:, -k:] @ np.diag(S[-k:])
# Recontruimos las imagenes
imagen_reconstruida_ultimas = representacion_baja_dimension_ultimas_dimensiones @ VT[-k:, :]
# Ploteos de las imágenes reconstruidas con las últimas k columnas
fig2, axs2 = plt.subplots(2, num_imagenes, figsize=(20, 8))

for i in range(num_imagenes):
    axs2[0, i].imshow(imagenes[i], cmap='gray')
    axs2[0, i].axis('off')
    reconstruccion = imagen_reconstruida_ultimas[i, :].reshape(imagenes[i].shape)
    axs2[1, i].imshow(reconstruccion, cmap='gray')
    axs2[1, i].axis('off')

# Añadir títulos a las subtramas
fig2.suptitle("Imágenes Originales y Reconstruidas con Últimas Dimensiones", fontsize=16)
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


#visualización individual de la reconstrucción para las primeras dimensiones (primera foto)----------------------------------------------
fig_individual, axs_individual = plt.subplots(1, 2, figsize=(8, 4))
titulo_original_individual = axs_individual[0].set_title('Imagen Original', fontsize=10)
axs_individual[0].imshow(imagenes[0], cmap='gray')
axs_individual[0].axis('off')
reconstruccion_individual = imagen_reconstruida_primeras[0, :].reshape(imagenes[0].shape)
titulo_reconstruido_individual = axs_individual[1].set_title('Reconstrucción con primeras dimensiones(k = {})'.format(k), fontsize=10)
axs_individual[1].imshow(reconstruccion_individual, cmap='gray')
axs_individual[1].axis('off')
plt.tight_layout()
plt.show()


#visualización individual de la reconstrucción para las ultimas dimensiones (primera foto)---------------------------------------------
fig_individual, axs_individual = plt.subplots(1, 2, figsize=(8, 4))
titulo_original_individual = axs_individual[0].set_title('Imagen Original', fontsize=10)
axs_individual[0].imshow(imagenes[0], cmap='gray')
axs_individual[0].axis('off')
reconstruccion_individual = imagen_reconstruida_ultimas[0, :].reshape(imagenes[0].shape)
titulo_reconstruido_individual = axs_individual[1].set_title('Reconstrucción con ultimas dimensiones(k = {})'.format(k), fontsize=10)
axs_individual[1].imshow(reconstruccion_individual, cmap='gray')
axs_individual[1].axis('off')
plt.tight_layout()
plt.show()


#Ejercicio 1.2?------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Mapa de calor para la matriz S original (valores singulares)
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz S')
plt.imshow(np.diag(S), cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Componentes')
plt.show()

# Mapa de calor para la matriz VT original(transpuesta de V)
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz VT')
plt.imshow(VT, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Características')
plt.show()

# Mapa de calor para la matriz U original
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz U')
plt.imshow(U, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Imágenes')
plt.show()

U_reconstruido, S_reconstruido, VT_reconstruido = np.linalg.svd(representacion_baja_dimension_ultimas_dimensiones, full_matrices=False)

# Mapa de calor para la matriz U reconstruido(baja dimensión)
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz U (baja dimensión)')
plt.imshow(U_reconstruido, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Imágenes')
plt.show()

# Mapa de calor para la matriz S reconstruida(baja dimensión)
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz S (baja dimensión)')
plt.imshow(np.diag(S_reconstruido), cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Componentes')
plt.show()

# Mapa de calor para la matriz VT reconstruida (baja dimensión)
plt.figure(figsize=(6, 4))
plt.title('Mapa de calor de la matriz VT (baja dimensión)')
plt.imshow(VT_reconstruido, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Componentes')
plt.ylabel('Características')
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
    ax.set_title(f'Similaridad (d = {valores_d[i]})')
    ax.set_xlabel("Imágenes")
    ax.set_ylabel("Imágenes")
    ax.grid(False)
    
plt.subplots_adjust(wspace=0.5)  # Ajusta el espacio entre subtramas
plt.show()


#1.4---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Cargar una imagen
imagen = cv2.imread('img00.jpeg', cv2.IMREAD_GRAYSCALE)

# Realizar la Descomposición de Valores Singulares (SVD) de la imagen
U, S, VT = np.linalg.svd(imagen, full_matrices=False)

# Valor deseado para d (número de componentes principales)
d = 8

# Tomar los primeros 'd' autovalores y componentes principales para comprimir la imagen
U_d = U[:, :d]
S_d = np.diag(S[:d])
VT_d = VT[:d, :]

# Proyectar la imagen original en un espacio de baja dimensión utilizando los 'd' componentes principales
imagen_comprimida = np.dot(np.dot(U_d, S_d), VT_d)

# Reconstruir la imagen comprimida
imagen_comprimida = imagen_comprimida.reshape(imagen.shape)

# Mostrar la imagen original y la imagen comprimida
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagen_comprimida, cmap='gray')
plt.title(f'Imagen Comprimida (d={d})')
plt.axis('off')

plt.tight_layout()
plt.show()

# Suponiendo que 'imagenes' es el conjunto de imágenes
n_imagenes = len(imagenes)

# Crear una lista para almacenar las representaciones de baja dimensión de cada imagen
representaciones_baja_dim = []

# Calcular la compresión a distintas dimensiones y encontrar la mínima 'd' con un error inferior al 10%
error_umbral = 0.1  # 10% de error

U, S, VT = np.linalg.svd(imagenes[0], full_matrices=False)
d_optimo = 0
V_optimo=[]
for d in range(1, 29):
    representacion_baja_dim = U[:, :d] @ np.diag(S[:d]) @ VT[:d, :]
    error = np.linalg.norm(imagenes[0] - representacion_baja_dim, 'fro') / np.linalg.norm(imagenes[0], 'fro')
    if error < error_umbral and d_optimo == 0:
        d_optimo = d
        VT_optimo = VT[:d, :]
        V_optimo = VT_optimo.T



print(V_optimo.shape)
# multiplicar por V_optimo cada imagen
representaciones_baja_dim = []
for imagen_original in imagenes:
    representacion_baja_dim = imagen_original @ V_optimo
    representaciones_baja_dim.append(representacion_baja_dim)

#plot all img_V
fig, axs = plt.subplots(4, 4, figsize=(20, 10))
for i in range(4):
    for j in range(4):
        axs[i, j].imshow(representaciones_baja_dim[i*4+j], cmap='gray')
        print(representaciones_baja_dim[i*4+j].shape)
        #title d=i
        axs[i, j].set_title(f'img{i*4+j}')
        axs[i, j].axis('off')
        
plt.show()