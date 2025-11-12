# Neurona_inteligente

Red neuronal simple para clasificar el dataset Iris.

Descripción
-----------
Red neuronal feedforward escrita desde cero (sin frameworks de alto nivel) para clasificar flores Iris en 3 clases.

Estructura del proyecto
-----------------------
```
Neurona_inteligente/
├── Data/                   # Datos y carga
├── Model/                  # Neurona y red neuronal
├── Preprocessing/          # Normalización y encoding
├── Tests/                  # Tests unitarios
└── Main.py                 # Entrenador/evaluador principal
```

Requisitos
----------
Se recomienda usar un entorno virtual. Dependencias principales están en `requirements.txt`.

Instalación (Windows PowerShell)
-------------------------------
```powershell
# Activar/crear entorno virtual (si no existe)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

Ejecutar tests
---------------
```powershell
# Desde la raíz del proyecto (entorno activado)
python .\Tests\test_neurona.py
```

Ejecutar programa principal
---------------------------
```powershell
# Ejecuta el entrenamiento y la evaluación
python .\Main.py
```

Autores
----------------
Viviana De la Cruz
