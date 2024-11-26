# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Cargar los datos
data = pd.read_csv("mall_customers.csv")

# Visualizar los primeros registros
print(data.head())

# Análisis exploratorio de los datos
print(data.info())
print(data.describe())

# Gráfica de la distribución de género
sns.countplot(data['Gender'])
plt.title("Distribución de género")
plt.show()

# Gráficas de la distribución de ingresos y puntaje de gasto
sns.histplot(data['Annual Income (k$)'], kde=True, bins=20, color='blue')
plt.title("Distribución de ingresos anuales")
plt.show()

sns.histplot(data['Spending Score (1-100)'], kde=True, bins=20, color='green')
plt.title("Distribución de puntajes de gasto")
plt.show()

# Gráfica de dispersión para explorar relaciones
sns.scatterplot(x=data['Annual Income (k$)'], y=data['Spending Score (1-100)'], hue=data['Gender'])
plt.title("Ingreso anual vs. Puntaje de gasto")
plt.show()

# Preprocesamiento de los datos
# Codificar la columna de género
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Seleccionar las características relevantes
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Escalamiento de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método del codo para determinar el número óptimo de clústeres
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Método del codo")
plt.xlabel("Número de clústeres")
plt.ylabel("Inercia")
plt.show()

# Entrenamiento del modelo KMeans
k_optimal = 5  # Supongamos que el óptimo es 5 según el método del codo
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Agregar las etiquetas de los clústeres al conjunto de datos original
data['Cluster'] = labels

# Evaluación del modelo
silhouette_avg = silhouette_score(X_scaled, labels)
calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

print(f"Coeficiente de Silhouette: {silhouette_avg}")
print(f"Índice de Calinski-Harabasz: {calinski_harabasz}")

# Visualización de los clústeres
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_scaled[:, 1], y=X_scaled[:, 2], 
    hue=labels, palette="tab10", s=100
)
plt.title("Clústeres de clientes (K-Means)")
plt.xlabel("Ingreso anual (normalizado)")
plt.ylabel("Puntaje de gasto (normalizado)")
plt.show()

# Guardar los resultados en un archivo
data.to_csv("customer_segmentation_results.csv", index=False)
print("Resultados guardados en 'customer_segmentation_results.csv'")
