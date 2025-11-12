"""
* processor.py
*
* Última fecha de modificación: 12/11/2025
* Viviana De la Cruz
* 
* Preprocesa datos: normaliza características y codifica etiquetas.
* Convierte texto a números y escala valores entre 0 y 1.
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler 


class Processor:
    """
    * Normaliza características y codifica etiquetas para entrenar la red.
    """
    
    def __init__(self):
        """
        * Inicializa los transformadores de sklearn.
        *
        * Post: self.scaler y self.label_encoder inicializados
        """
        self.scaler = MinMaxScaler()  # Escala valores entre 0 y 1
        self.label_encoder = LabelEncoder()  # Convierte texto a números
        
    def normalizar_caracteristicas(self, X, entrenar=True):
        """
        * Normaliza características al rango [0, 1].
        *
        * Pre: X es un array numpy de características
        * Post: Retorna X normalizado entre 0 y 1
        *       Si entrenar=True, ajusta el scaler
        *       Si entrenar=False, usa scaler ya ajustado
        """
        if entrenar:
            # Aprende los valores min/max y transforma
            return self.scaler.fit_transform(X)
        else:
            # Solo transforma con valores ya aprendidos
            return self.scaler.transform(X)
        
    def codificar_etiquetas(self, y, entrenar=True):
        """
        * Convierte etiquetas de texto a números.
        * Ejemplo: ['Iris-setosa', 'Iris-versicolor'] → [0, 1]
        *
        * Pre: y es un array de strings (nombres de especies)
        * Post: Retorna y codificado como números (0, 1, 2)
        *       Si entrenar=True, aprende las clases
        *       Si entrenar=False, usa clases ya aprendidas
        """
        if entrenar:
            return self.label_encoder.fit_transform(y)
        else:
            return self.label_encoder.transform(y)
        
    def one_hot_encode(self, y, num_clases):
        """
        * Convierte etiquetas numéricas a formato one-hot.
        * Ejemplo: [0, 1, 2] → [[1,0,0], [0,1,0], [0,0,1]]
        *
        * Pre: y es un array de enteros (0, 1, 2, ...)
        *      num_clases es el número total de clases
        * Post: Retorna matriz one-hot de tamaño (len(y), num_clases)
        """
        # Crear matriz de ceros
        one_hot = np.zeros((len(y), num_clases))
        
        # Poner 1 en la posición correspondiente a cada clase
        for i, clase in enumerate(y):
            one_hot[i, clase] = 1
            
        return one_hot
    
    def decodificar_etiquetas(self, y_codificado):
        """
        * Convierte números de vuelta a nombres de especies.
        * Ejemplo: [0, 1, 2] → ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        *
        * Pre: y_codificado es un array de enteros
        *      label_encoder debe estar entrenado
        * Post: Retorna array de strings con nombres de especies
        """
        return self.label_encoder.inverse_transform(y_codificado)
