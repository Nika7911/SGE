"""
* test_neurona.py
*
* Última fecha de modificación: 12/11/2025
* Viviana De la Cruz
* 
* Script de prueba para verificar el funcionamiento de la clase Neurona.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Model.neurona import Neurona


def test_inicializacion():
    """
    * Prueba que la neurona se inicializa correctamente.
    """
    neurona = Neurona(num_entradas=4)
    assert len(neurona.pesos) == 4, "Error: Número de pesos incorrecto"
    assert neurona.bias == 0.0, "Error: Bias debe ser 0.0"
    print("✓ Inicialización correcta")


def test_funcion_activacion():
    """
    * Prueba la función sigmoide.
    """
    neurona = Neurona(num_entradas=4)
    assert abs(neurona.activacion(0) - 0.5) < 0.001, "Error: sigmoide(0) debe ser ~0.5"
    assert neurona.activacion(10) > 0.9, "Error: sigmoide(10) debe ser > 0.9"
    assert neurona.activacion(-10) < 0.1, "Error: sigmoide(-10) debe ser < 0.1"
    print("✓ Función sigmoide correcta")


def test_forward():
    """
    * Prueba la propagación hacia adelante.
    """
    neurona = Neurona(num_entradas=4)
    entradas = np.array([5.1, 3.5, 1.4, 0.2])
    salida = neurona.forward(entradas)
    assert 0 < salida < 1, "Error: La salida debe estar entre 0 y 1"
    print("✓ Forward propagation correcta")


def ejecutar_todos_los_tests():
    """
    * Ejecuta todos los tests de la neurona.
    """
    print("\nTESTS DE LA CLASE NEURONA")
    print("-" * 40)
    
    try:
        test_inicializacion()
        test_funcion_activacion()
        test_forward()
        print("-" * 40)
        print("✓ Todos los tests pasaron\n")
        
    except AssertionError as e:
        print(f"✗ Test fallido: {e}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")


if __name__ == "__main__":
    ejecutar_todos_los_tests()
