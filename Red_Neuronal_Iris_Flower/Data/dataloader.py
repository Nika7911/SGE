"""
* dataloader.py
*
* Última fecha de modificación: 12/11/2025
* Viviana De la Cruz
* 
* Clase para cargar y dividir el dataset Iris desde archivo CSV.
* Separa características (X) de etiquetas (y) y divide en train/test.
"""

import pandas as pd 
from sklearn.model_selection import train_test_split  

class DataLoader:
    
    def __init__(self, ruta_csv):
        """
        * Inicializa el DataLoader con la ruta del archivo CSV.
        *
        * Pre: ruta_csv debe ser una ruta válida a un archivo CSV
        * Post: self.ruta_csv almacena la ruta del archivo
        """
        self.ruta_csv = ruta_csv
        
    def cargar_datos(self):
        """
        * Lee el archivo CSV y retorna un DataFrame de pandas.
        *
        * Pre: El archivo CSV debe existir en self.ruta_csv
        * Post: Retorna DataFrame con todos los datos del CSV
        """
        df = pd.read_csv(self.ruta_csv)
        return df
        
    def preparar_datos(self, test_size=0.2, random_state=42):
        """
        * Carga datos, separa características de etiquetas y divide en train/test.
        *
        * Pre: El CSV debe tener columnas: SepalLengthCm, SepalWidthCm, 
        *      PetalLengthCm, PetalWidthCm, Species
        * Post: Retorna X_train, X_test, y_train, y_test como arrays numpy
        *       test_size: porcentaje de datos para prueba (20% por defecto)
        """
        # Cargar datos del CSV
        df = self.cargar_datos()
        
        # Eliminar columna 'Id', no es útil para clasificación
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
        
        # Separar características (X) de etiquetas (y)
        # X: las 4 medidas de la flor
        X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
        
        # y: la especie de la flor
        y = df['Species'].values
        
        # Dividir en conjunto de entrenamiento (80%) y prueba (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Mantiene proporción de clases en ambos conjuntos
        )
        
        return X_train, X_test, y_train, y_test
    
    def obtener_info_dataset(self):
        """
        * Muestra información básica del dataset.
        *
        * Pre: El archivo CSV debe existir
        * Post: Imprime resumen del dataset
        """
        df = self.cargar_datos()
        print(f"\nDataset: {len(df)} muestras, {len(df.columns)} columnas")
        print(f"Especies: {df['Species'].nunique()} tipos\n")

