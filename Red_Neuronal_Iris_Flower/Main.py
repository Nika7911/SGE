"""
* Main.py
*
* Última fecha de modificación: 12/11/2025
* Viviana De la Cruz
* 
* Programa principal que entrena y evalúa la red neuronal
* para clasificar flores del dataset Iris.
"""
from Data.dataloader import DataLoader
from Preprocessing.processor import Processor
from Model.red_neuronal import RedNeuronal
import numpy as np


def calcular_precision(predicciones, y_real, y_real_codificado):
    """
    * Calcula la precisión de las predicciones.
    *
    * Pre: predicciones es array de salidas one-hot
    *      y_real y y_real_codificado son arrays de etiquetas
    * Post: Retorna porcentaje de aciertos
    """
    # Convertir predicciones one-hot a clases (índice del máximo)
    clases_pred = np.argmax(predicciones, axis=1)
    
    # Comparar con las clases reales
    aciertos = np.sum(clases_pred == y_real_codificado)
    precision = (aciertos / len(y_real)) * 100
    
    return precision, clases_pred


def main():
    """
    * Función principal que ejecuta todo el pipeline.
    """
    print("\n" + "="*60)
    print("RED NEURONAL - CLASIFICACIÓN DE FLORES IRIS")
    print("="*60)
    
    # 1. Cargar datos
    dataloader = DataLoader("Data/Datasets/Iris.csv")
    X_train, X_test, y_train, y_test = dataloader.preparar_datos()
    
    # 2. Preprocesar
    processor = Processor()
    X_train_norm = processor.normalizar_caracteristicas(X_train, entrenar=True)
    X_test_norm = processor.normalizar_caracteristicas(X_test, entrenar=False)
    y_train_cod = processor.codificar_etiquetas(y_train, entrenar=True)
    y_test_cod = processor.codificar_etiquetas(y_test, entrenar=False)
    y_train_onehot = processor.one_hot_encode(y_train_cod, num_clases=3)
    
    # 3. Crear y entrenar red neuronal
    red = RedNeuronal(arquitectura=[4, 8, 3])
    red.entrenar(X_train_norm, y_train_onehot, epochs=1000, learning_rate=0.01)
    
    # 4. Evaluar en TRAIN y TEST
    print("\nEvaluando...")
    
    # Precisión en entrenamiento
    predicciones_train = red.predecir(X_train_norm)
    precision_train, _ = calcular_precision(predicciones_train, y_train, y_train_cod)
    
    # Precisión en test (datos nuevos)
    predicciones_test = red.predecir(X_test_norm)
    precision_test, clases_pred = calcular_precision(predicciones_test, y_test, y_test_cod)
    
    # Mostrar resultados
    print(f"\n{'='*60}")
    print(f"RESULTADOS")
    print(f"{'='*60}")
    print(f"Precisión en TRAIN: {precision_train:.2f}% (datos vistos)")
    print(f"Precisión en TEST:  {precision_test:.2f}% (datos nuevos)")
    
    # Análisis de aprendizaje
    diferencia = abs(precision_train - precision_test)
    if diferencia < 10:
        print(f"\n✓ La red APRENDIÓ correctamente (diferencia: {diferencia:.1f}%)")
    else:
        print(f"\n✗ Posible memorización (diferencia: {diferencia:.1f}%)")
    
    print(f"\nEjemplos de predicciones en TEST:")
    print(f"{'Real':<20} {'Predicción':<20} {'Confianza'}")
    print("-" * 60)
    
    nombres_especies = processor.decodificar_etiquetas([0, 1, 2])
    
    for i in range(min(10, len(y_test))):
        nombre_real = y_test[i]
        nombre_pred = nombres_especies[clases_pred[i]]
        confianza = np.max(predicciones_test[i]) * 100
        simbolo = "✓" if nombre_real == nombre_pred else "✗"
        
        print(f"{nombre_real:<20} {nombre_pred:<20} {confianza:>5.1f}% {simbolo}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
