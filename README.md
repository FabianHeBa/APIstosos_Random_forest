# APIstosos_Random_forest

# APIstosos Random Forest 🌳

Un API simple construida con **FastAPI** para servir un modelo de predicción de Scikit-Learn (Random Forest) entrenado con el dataset Iris.

Este proyecto fue desarrollado por el equipo **"APIstosos"** y está configurado para despliegue automático en **Render** usando un archivo `render.yaml` (Blueprint).

## 🚀 API en Vivo (Desplegada en Render)

La API está desplegada y puedes interactuar con ella:

* **URL Base:** `https://apistosos-random-forest-1.onrender.com/docs`
* **Health Check:** `https://apistosos-random-forest-1.onrender.com/health`
* **Info del Modelo:** `https://apistosos-random-forest-1.onrender.com/info`

## 🛠️ Requerimeintos básicos

* **Python 3**
* **FastAPI:** Para construir el API.
* **Uvicorn:** Como servidor ASGI.
* **Scikit-Learn / Joblib:** Para cargar y usar el modelo `.pkl`.
* **Render:** Para el despliegue y hosting.

## 📁 Estructura del Proyecto

El proyecto está contenido dentro de la carpeta `app/` para mantener la raíz limpia para los archivos de configuración de despliegue.

## 💻 Cómo ejecutarlo localmente

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/FabianHeBa/APIstosos_Random_forest.git](https://github.com/FabianHeBa/APIstosos_Random_forest.git)
    cd APIstosos_Random_forest/app
    ```

2.  **Crear un entorno virtual y activarlo:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación:**
    El servidor se iniciará en `http://127.0.0.1:8000`.
    ```bash
    uvicorn main:app --reload
    ```
5.  Abre tu navegador en `http://127.0.0.1:8000/docs` para ver la documentación interactiva.

## 🔌 Endpoints del API

### GET /health
Endpoint simple para verificar que la API está viva.
* **Respuesta:** `{"status": "ok"}`

### GET /info
Muestra información sobre el modelo y el equipo.
* **Respuesta:**
    ```json
    {
      "team": "APIstosos",
      "Model": "RandomForestClassifier",
      "n_estimators": 100,
      "max_depth": 8
    }
    ```

### POST /predict
Recibe 4 features de la flor Iris (Sepal Length, Sepal Width, Petal Length, Petal Width) y retorna la clase predicha.

* **Body (JSON):**
    ```json
    {
      "features": [5.1, 3.5, 1.4, 0.2]
    }
    ```

* **Ejemplo de uso con `curl` (en Windows CMD):**
    ```bash
    curl -X "POST" "[https://apistosos-random-forest-1.onrender.com/predict](https://apistosos-random-forest-1.onrender.com/predict)" -H "Content-Type: application/json" -d "{\"features\": [5.1, 3.5, 1.4, 0.2]}"
    ```

* **Respuesta Exitosa:**
    ```json
    {
      "prediction": "setosa"
    }
    ```
