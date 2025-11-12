"""
* red_neuronal.py
*
* Última fecha de modificación: 12/11/2025
* Viviana De la Cruz
*
* Implementa una red neuronal multicapa con backpropagation.
* Combina múltiples neuronas en capas para clasificación.
"""
import numpy as np
from Model.neurona import Neurona 

class RedNeuronal:
    """
    * Red neuronal feedforward con múltiples capas.
    """
    
    def __init__(self, arquitectura):
        """
        * Crea la red con la arquitectura especificada.
        * 
        * Ejemplo: arquitectura = [4, 8, 3]
        *   - Capa entrada: 4 neuronas (características de Iris)
        *   - Capa oculta: 8 neuronas
        *   - Capa salida: 3 neuronas (3 especies)
        *
        * Pre: arquitectura es una lista con al menos 2 elementos
        * Post: self.capas contiene listas de neuronas para cada capa
        *       self.arquitectura almacena la estructura de la red
        """
        self.arquitectura = arquitectura
        self.num_capas = len(arquitectura) - 1
        self.capas = []
        
        # Crear capas de neuronas
        for i in range(self.num_capas):
            num_entradas = arquitectura[i]
            num_neuronas = arquitectura[i + 1]
            
            # Crear lista de neuronas para esta capa
            capa = [Neurona(num_entradas) for _ in range(num_neuronas)]
            self.capas.append(capa)
        
    def forward(self, X):
        """
        * Propagación hacia adelante a través de todas las capas.
        *
        * Pre: X es un array numpy con características de entrada
        *      X.shape[0] debe coincidir con arquitectura[0]
        * Post: Retorna la salida de la última capa
        *       self.activaciones guarda las salidas de cada capa
        """
        self.activaciones = [X]  # Guardar entrada inicial
        salida = X
        
        # Pasar por cada capa
        for capa in self.capas:
            salidas_capa = []
            for neurona in capa:
                salidas_capa.append(neurona.forward(salida))
            
            salida = np.array(salidas_capa)
            self.activaciones.append(salida)
        
        return salida
    
    def calcular_perdida(self, y_pred, y_real):
        """
        * Calcula el error cuadrático medio (MSE).
        *
        * Pre: y_pred y y_real son arrays numpy del mismo tamaño
        * Post: Retorna un float con el error promedio
        """
        return np.mean((y_real - y_pred) ** 2)
    
    def backpropagation(self, X, y, learning_rate):
        """
        * Ajusta los pesos de la red usando backpropagation.
        *
        * Pre: X son las entradas, y las salidas esperadas
        *      learning_rate > 0
        *      forward() debe haberse ejecutado antes
        * Post: Actualiza pesos y bias de todas las neuronas
        """
        # Forward para obtener predicción
        y_pred = self.forward(X)
        
        # Calcular error de salida
        error = y - y_pred
        
        # Backpropagation de atrás hacia adelante
        for i in reversed(range(self.num_capas)):
            capa = self.capas[i]
            entrada_capa = self.activaciones[i]
            
            # Calcular gradientes para esta capa
            gradientes = []
            for j, neurona in enumerate(capa):
                salida = self.activaciones[i + 1][j]
                
                # Derivada de sigmoide: σ'(x) = σ(x) * (1 - σ(x))
                derivada_activacion = salida * (1 - salida)
                gradiente = error[j] * derivada_activacion
                gradientes.append(gradiente)
                
                # Actualizar pesos y bias
                neurona.pesos += learning_rate * gradiente * entrada_capa
                neurona.bias += learning_rate * gradiente
            
            # Propagar error a capa anterior
            if i > 0:
                error_anterior = np.zeros(len(self.capas[i - 1]))
                for j, gradiente in enumerate(gradientes):
                    error_anterior += gradiente * capa[j].pesos
                error = error_anterior
    
    def entrenar(self, X_train, y_train, epochs=1000, learning_rate=0.01):
        """
        * Entrena la red durante un número determinado de épocas.
        *
        * Pre: X_train y y_train son arrays numpy
        *      y_train debe estar en formato one-hot
        *      epochs > 0, learning_rate > 0
        * Post: Los pesos de la red están ajustados
        *       Imprime el progreso del entrenamiento
        """
        print(f"\nIniciando entrenamiento: {epochs} épocas, lr={learning_rate}")
        print("=" * 60)
        
        for epoca in range(epochs):
            errores = []
            
            # Entrenar con cada muestra
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                
                # Forward + Backprop
                y_pred = self.forward(X)
                self.backpropagation(X, y, learning_rate)
                
                # Calcular error
                error = self.calcular_perdida(y_pred, y)
                errores.append(error)
            
            # Mostrar progreso cada 100 épocas
            if (epoca + 1) % 100 == 0:
                error_promedio = np.mean(errores)
                print(f"Época {epoca + 1}/{epochs} - Error: {error_promedio:.4f}")
        
        print("=" * 60)
        print("Entrenamiento completado ✓\n")
    
    def predecir(self, X):
        """
        * Realiza predicciones sobre nuevos datos.
        *
        * Pre: X puede ser un array 1D (una muestra) o 2D (múltiples muestras)
        * Post: Retorna predicciones en formato one-hot
        """
        # Si es una sola muestra, predecir directamente
        if X.ndim == 1:
            return self.forward(X)
        
        # Si son múltiples muestras, predecir cada una
        predicciones = []
        for muestra in X:
            pred = self.forward(muestra)
            predicciones.append(pred)
        
        return np.array(predicciones)
