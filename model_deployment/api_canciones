# api_canciones.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# 1. Cargar el modelo entrenado
modelo = joblib.load('modelo_popularidad_canciones.pkl')

# 2. Inicializar la aplicación de Flask
app = Flask(__name__)

# 3. Crear una ruta principal para confirmar que la API funciona
@app.route('/')
def home():
    return "API de Predicción de Popularidad de Canciones funcionando correctamente."

# 4. Crear la ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del request en formato JSON
        data = request.get_json()

        # Convertir los datos en un DataFrame para el modelo
        input_features = pd.DataFrame([data])

        # Realizar la predicción
        prediccion = modelo.predict(input_features)

        # Devolver la predicción
        return jsonify({'popularidad_predicha': float(prediccion[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

# 5. Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
