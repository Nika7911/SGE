"""
* neurona.py
*
* Última fecha de modificación: 12/11/2025
* Viviana De la Cruz
* 
* Creamos una neurona artificial con función de activación sigmoide.
* La neurona realiza una suma de las entradas más un bias,
"""
import numpy as np

class Neurona:
    
    def __init__(self, num_entradas):
        """
        * Inicializa pesos aleatorios pequeños y bias en 0.
        *
        * Pre: num_entradas > 0
        * Post: self.pesos inicializado aleatoriamente
        *       self.bias = 0.0
        """
        # Pesos aleatorios pequeños para evitar saturación inicial
        self.pesos = np.random.randn(num_entradas) * 0.1
        self.bias = 0.0
        
    def activacion(self, x):
        """
        * Función sigmoide: σ(x) = 1 / (1 + e^(-x))
        *
        * Pre: x número real o array numpy
        * Post: Retorna valor en rango (0, 1)
        """
        x = np.clip(x, -500, 500)  # Evitar overflow
        return 1 / (1 + np.exp(-x))
    
    def forward(self, entradas):
        """
        * Propagación hacia adelante: calcula salida de la neurona.
        *
        * Pre: len(entradas) == len(self.pesos)
        * Post: Retorna valor entre 0 y 1
        *       No modifica el estado de la neurona
        """
        suma_ponderada = np.dot(entradas, self.pesos) + self.bias
        return self.activacion(suma_ponderada)
    
    def __repr__(self):
        return f"Neurona(pesos={self.pesos}, bias={self.bias:.3f})"

