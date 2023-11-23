import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

n = 5
d = 100

A = np.random.rand(n, d)
b = np.random.rand(n)

# Función de costo F(x)
def cost_function_F(x):
    return np.linalg.norm(np.dot(A, x) - b)**2

# Función de costo F2(x) con regularización L2
def cost_function_F2(x, delta_squared):
    return cost_function_F(x) + delta_squared * np.linalg.norm(x)**2

def gradient_descent(cost_function, initial_guess, learning_rate, iterations, delta_squared=None):
    x = initial_guess.copy()
    for i in range(iterations):
        gradient = 2 * np.dot(A.T, (np.dot(A, x) - b))
        if delta_squared is not None:
            gradient += 2 * delta_squared * x
        gradient /= np.linalg.norm(gradient)
        x -= learning_rate * gradient
    return x

# Obtención de la solución mediante SVD
U, S, Vt = svd(A, full_matrices=False)
V = Vt.T
pseudoinv = np.dot(V, np.dot(np.diag(1/S), U.T))
solution_svd = np.dot(pseudoinv, b)

# Parámetros
learning_rate = 1 / S[0]  # Usando el inverso del mayor valor singular para el tamaño de paso
iterations = 1000
delta_squared = 1e-2 * np.max(S)  # Valor delta^2 para F2

# Condición inicial aleatoria
initial_guess = np.random.rand(d)

solution_F = gradient_descent(cost_function_F, initial_guess, learning_rate, iterations)
solution_F2 = gradient_descent(cost_function_F2, initial_guess, learning_rate, iterations, delta_squared)


print("Solución mediante SVD:", solution_svd)
print("Solución mediante gradiente descendiente (F):", solution_F)
print("Solución mediante gradiente descendiente (F2):", solution_F2)


def gradient_descent_tracking(cost_function, initial_guess, learning_rate, iterations, delta_squared=None):
    x = initial_guess.copy()
    solutions = [x.copy()]  # Almacenar la solución inicial
    for i in range(iterations):
        gradient = 2 * np.dot(A.T, (np.dot(A, x) - b))
        if delta_squared is not None:
            gradient += 2 * delta_squared * x
        gradient /= np.linalg.norm(gradient)  # Normalizar el gradiente
        x -= learning_rate * gradient
        solutions.append(x.copy())  # Almacenar la solución actual
    return solutions

# Obtener soluciones a lo largo de las iteraciones
solutions_F = gradient_descent_tracking(cost_function_F, initial_guess, learning_rate, iterations)
solutions_F2 = gradient_descent_tracking(cost_function_F2, initial_guess, learning_rate, iterations, delta_squared)


# Crear la variable num_iterations para el rango de iteraciones
num_iterations = range(iterations + 1)  # Iteraciones + solución inicial

# Graficar la evolución de las primeras 5 dimensiones de la solución para F
num_dimensions_to_plot = 5
plt.figure(figsize=(10, 6))
plt.title("Evolución de las primeras 5 dimensiones de la solución (F)")
plt.xlabel("Iteración")
plt.ylabel("Valor de la solución")

for i in range(num_dimensions_to_plot):
    sol_values = [sol[i] for sol in solutions_F]
    plt.plot(num_iterations, sol_values, label=f"x[{i}] (F)", alpha=0.7)
plt.legend()
plt.show()

# Graficar la evolución de las primeras 5 dimensiones de la solución para F2
plt.figure(figsize=(10, 6))
plt.title("Evolución de las primeras 5 dimensiones de la solución con regularización L2 (F2)")
plt.xlabel("Iteración")
plt.ylabel("Valor de la solución")

for i in range(num_dimensions_to_plot):
    sol_values = [sol[i] for sol in solutions_F2]
    plt.plot(num_iterations, sol_values, label=f"x[{i}] (F2)", alpha=0.7)
plt.legend()
plt.show()


# Obtener los valores de la función de costo a lo largo de las iteraciones para F y F2
cost_values_F = [cost_function_F(sol) for sol in solutions_F]
cost_values_F2 = [cost_function_F2(sol, delta_squared) for sol in solutions_F2]

# Graficar la función de costo a lo largo de las iteraciones para F
plt.figure(figsize=(10, 6))
plt.title("Valor de la función de costo (F) a lo largo de las iteraciones")
plt.xlabel("Iteración")
plt.ylabel("Valor de la función de costo")
plt.plot(num_iterations, cost_values_F, label="Costo (F)", alpha=0.7)
plt.legend()
plt.show()

# Graficar la función de costo a lo largo de las iteraciones para F2
plt.figure(figsize=(10, 6))
plt.title("Valor de la función de costo con regularización L2 (F2) a lo largo de las iteraciones")
plt.xlabel("Iteración")
plt.ylabel("Valor de la función de costo")
plt.plot(num_iterations, cost_values_F2, label="Costo (F2)", alpha=0.7)
plt.legend()
plt.show()


# Calcular el error respecto al mínimo encontrado por el método
error_F = np.array(cost_values_F) - cost_function_F(solution_F)
error_F2 = np.array(cost_values_F2) - cost_function_F2(solution_F2, delta_squared)

# Graficar la convergencia del error para F y F2
plt.figure(figsize=(10, 6))
plt.title("Convergencia del error (F)")
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.plot(num_iterations, error_F, label="Error (F)", alpha=0.7)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.title("Convergencia del error con regularización L2 (F2)")
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.plot(num_iterations, error_F2, label="Error (F2)", alpha=0.7)
plt.legend()
plt.show()

# Realizar un estudio variando delta_squared
delta_squared_values = [1e-4, 1e-2, 1,50,100,1000,15000]  # Valores de delta_squared a probar

errors_F2_varied = []  # Almacenar los errores para cada valor de delta_squared
solutions_F2_varied = []  # Almacenar las soluciones para cada valor de delta_squared

for delta_squared_val in delta_squared_values:
    solution_F2_varied = gradient_descent(cost_function_F2, initial_guess, learning_rate, iterations, delta_squared_val)
    solutions_F2_varied.append(solution_F2_varied)

    # Calcular los errores respecto al mínimo encontrado por el método para F2 variado
    cost_values_F2_varied = [cost_function_F2(sol, delta_squared_val) for sol in gradient_descent_tracking(
        cost_function_F2, initial_guess, learning_rate, iterations, delta_squared_val)]
    errors_F2_varied.append(np.array(cost_values_F2_varied) - cost_function_F2(solution_F2_varied, delta_squared_val))

# Graficar la convergencia del error para F2 con diferentes delta_squared
plt.figure(figsize=(10, 6))
plt.title("Convergencia del error con diferentes delta_squared (F2)")
plt.xlabel("Iteración")
plt.ylabel("Error")

for i, delta_squared_val in enumerate(delta_squared_values):
    plt.plot(range(iterations + 1), errors_F2_varied[i], label=f"delta_squared = {delta_squared_val}", alpha=0.7)

plt.legend()
plt.show()