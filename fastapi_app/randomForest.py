from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# Cargar el conjunto de datos Iris
print("Cargando el conjunto de datos Iris...")
iris = load_iris()
X = iris.data
y = iris.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

# Crear y entrenar el modelo de Random Forest
print("\nEntrenando el modelo de Random Forest...")
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.4f}")

# Mostrar informe de clasificación
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Guardar el modelo entrenado usando joblib
print("\nGuardando el modelo en formato joblib...")
joblib.dump(modelo_rf, './model.joblib')