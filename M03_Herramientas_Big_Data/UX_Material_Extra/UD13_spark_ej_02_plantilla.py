from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import sys

# DEFINICION DE FUNCIONES

def print_fitted_model_results(model, feature_names, fout = sys.stdout):
    """Imprimir resumen de parametros principales del modelo ajustado"""
    
    print("Fitted model summary\n", file=fout)

    if (len(feature_names) < len(model.summary.pValues)) and (feature_names[0].lower() != 'intercept'):
        feature_names = ['Intercept'] + feature_names
    
    feature_names = [ feat.ljust(20) for feat in feature_names ]
    
    coefs = np.append(model.intercept, model.coefficients.values)
    p_values = model.summary.pValues
    t_values = model.summary.tValues
    
    print("Variable            Coefficient\tp-Value.\tt-Value.", file=fout)
    for model_res in zip(feature_names, coefs, 
                         p_values, t_values):
        print("%s%f\t%e\t%f" % model_res, file=fout)
    
    print("", file=fout)
    print("RMSE : %f" % model.summary.rootMeanSquaredError, file=fout)
    print("r2 : %f" % model.summary.r2, file=fout)
    print("", file=fout)


def eval_model_prediction(pred, predictionCol, labelCol, fout = sys.stdout):
    """
    Evaluar la calidad de los valores previstos por el modelo
    frente a los valores reales, usando RMSE y R2
    """
    rmse_evaluator = RegressionEvaluator(
                      predictionCol = predictionCol, 
                      labelCol = labelCol, 
                      metricName = "rmse")

    r2_evaluator = RegressionEvaluator(
                      predictionCol = predictionCol, 
                      labelCol = labelCol, 
                      metricName = "r2")

    print("Fitted model - Test data summary\n", file=fout)

    print("RMSE on test data = %g" % 
          rmse_evaluator.evaluate(lr_predict), file=fout)
    print("R2 on test data = %g" % 
          r2_evaluator.evaluate(lr_predict), file=fout)



#############################
# CUERPO PRINCIPAL
#############################

# Crear la sesión de Spark
spark = 

# Leer datos de fichero local
auto_file = r'file:///home/bigdata/Curso/U13_Spark/datos/Auto.csv'

auto = spark.read.csv(path = auto_file, 
                      sep = ',', header = True, inferSchema = True)

# Crear fichero local para escribir resultados del modelo
result_file = r'/home/bigdata/Curso/U13_Spark/output/ejercicio_02/resmodel.txt'
fres = open(result_file, 'w')

# Preparar el modelo para predecir el consumo de combustible (mpg)
# en base a variables explicativas (peso - weight)
# usando un modelo de regresion lineal

# Lista con nombres de variables explicativas a usar
regressors = 

# Construimos los vectores de variables explicativas
assembler = 

# Dividimos los datos aleatoriamente 
# en conjuntos de entrenamiento (80%) y test (20%)
auto_train, auto_test = 

# Construimos el modelo de regresion lineal
# para predecir el consumo (mpg)
lr = 

# Ajustamos el modelo con el subconjunto de entrenamiento
lr_model = 

# Imprimir los resultados del ajuste del modelo
print_fitted_model_results(lr_model, regressors, fout = fres)

# Predecir el consumo sobre el conjunto de test
lr_predict = 

# Imprimir indicadores resumen de la calidad de la prevision
eval_model_prediction(lr_predict, "prediction", "mpg", fout = fres)

# Cerrar el fichero local para los resultados del modelo
fres.close()


