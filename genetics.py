#Notas importanes.
#Un individuo tiene Cromosomas y los Cromosomas son estructuras formadas por Genes.
#Un individuo será representado por un objeto perteneciente a la clase Individual.
#Un Cromosoma estará representado por el conjunto de hiperparámetros para la red neuronal (una lista).
#   También puede verse como un registro de un dataset.
#Un gen estará representado por un hiperparámetro de la red neuronal, y a su vez serán arreglos de tipo binario.
#    Puede verse como una variable o como un atributo del dataset (o de una clase).
#El valor dado a cada gen es llamado alelo, así que tenemos alelos binarios, alelos numericos, etc.
#Una población es un conjunto de individuos, y estará representada por una lista de objetos de la clase Individual.
#La aptitud de un individuo será equivalente al Acc en la red neuronal.

#Importación de bibliotecas y módulos
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from math import sqrt

#Lectura del dataframe de forma global.
df=pd.read_csv("descriptoresFinal.csv")
#Columna a predecir.
target_column = ['clase']
#Lista de columnas que fungirán como predictores.
predictors = list(set(list(df.columns))-set(target_column))
#Normalización de los predictores (rango 0 a 1).
df[predictors] = df[predictors]/df[predictors].max()

#Arreglo de los valores de las variables independientes (entrada.)
X = df[predictors].values
#Arreglo de los valores de la variable de clase (dependiente/salida)
y = df[target_column].values
#Determinamos K para el algoritmo KFolds Cross Validation.
K=2
kf = KFold(n_splits=K)

#Clase individuo.
#Permite la creación de un nuevo individuo.
class Indivual():
    #Constructor. Puede recibir o no un cromosoma ya previamente construido (una inyección de genes/organismo genéticamente modificado).
    def __init__(self, Chromosome=None):
        #Si no hay cromosoma, este se genera con genes con valores al azar.
        if Chromosome is None:
            #Se genera número de neuronas.
            self.num_neurons = np.random.randint(2, size=4)
            #Conversión a decimal.
            aux_decimal= self.get_num_neurons2Int() 
            #Se verifica que el número de neuronas esté entre 1 y 13.
            #   Si no es así, se vuelve a generar un valor al azar.
            while aux_decimal<=2 or aux_decimal>13:
                self.num_neurons = np.random.randint(2, size=4)
                aux_decimal=self.get_num_neurons2Int()
            
            #Se genera número de capas ocultas.
            self.hidden_layers = np.random.randint(2, size=3)
            #Conversión a decimal.
            aux_decimal= self.get_hidden_layers2Int()
            #Se verifica que el número de capas ocultas sea mayor a 0.
            #   Si no es así, se vuelve a generar un valor al azar.
            while aux_decimal==0:
                self.hidden_layers = np.random.randint(2, size=3)
                aux_decimal= self.get_hidden_layers2Int()
            
            #Se genera número de épocas.
            self.num_epochs = np.random.randint(2, size=10)
            #Conversión a decimal.
            aux_decimal= self.get_num_epochs2Int()
            #Se verifica que el número de épocas sea mayor a 0.
            #   Si no es así, se vuelve a generar un valor al azar.
            while aux_decimal==0:
                self.num_epochs = np.random.randint(2, size=10)
                aux_decimal= self.get_num_epochs2Int()

            #Se genera Learning Rate.
            self.learning_rate = np.random.randint(2, size=7)
            #Conversión a decimal.
            aux_decimal= self.get_learning_rate2Float()
            #Se verifica que el learning rate esté entre 0.1 y 0.64.
            #   Si no es así, se vuelve a generar un valor al azar.
            while aux_decimal<=0 or aux_decimal>1:
                self.learning_rate = np.random.randint(2, size=7)
                aux_decimal=self.get_learning_rate2Float()

            #Se genera Momentum.
            self.momentum = np.random.randint(2, size=7)
            #Conversión a decimal.
            aux_decimal= self.get_momentum_2Float()
            #Se verifica que el momentum esté entre 0.1 y 1.
            #   Si no es así, se vuelve a generar un valor al azar.
            while aux_decimal<=0 or aux_decimal>1:
                self.momentum = np.random.randint(2, size=7)
                aux_decimal=self.get_momentum_2Float()
        #Si hay algún cromosoma, se inicializan los genes con él.
        else:
            self.num_neurons = Chromosome[0]
            self.hidden_layers = Chromosome[1]
            self.num_epochs = Chromosome[2]
            self.learning_rate = Chromosome[3]
            self.momentum = Chromosome[4]
        
        #Se calcula la aptitud (Acc) del individuo llamando a la red neuronal.
        self.fit=self.call_network()
        print(f'**Individuo**\nNum_neurons: {self.get_num_neurons2Int()}\nHidden_layers: {self.get_hidden_layers2Int()}\nNum_epochs: {self.get_num_epochs2Int()}\nLearning_rate: {self.get_learning_rate2Float()}\nMomentum: {self.get_momentum_2Float()}\nFitness: {self.fit}')
    
    #Método que retorna el valor del número de neuronas como entero.
    def get_num_neurons2Int(self):
        return int("".join(str(x) for x in self.num_neurons), 2)
    #Método que retorna el valor del número de neuronas como array de bits.
    def get_num_neurons(self):
        return self.get_num_neurons

    #Método que retorna el valor del número de capas ocultas como entero.
    def get_hidden_layers2Int(self):
        return int("".join(str(x) for x in self.hidden_layers), 2)
    #Método que retorna el valor del número de capas ocultas como array de bits.
    def get_hidden_layers(self):
        return self.hidden_layers
    
    #Método que retorna el valor del número de épocas como entero.
    def get_num_epochs2Int(self):
        return int("".join(str(x) for x in self.num_epochs), 2)
    #Método que retorna el valor del número de épocas como array de bits.
    def get_num_epochs(self):
        return self.num_epochs
    
    #Método que retorna el valor del Learning Rate como flotante.
    def get_learning_rate2Float(self):
        return int("".join(str(x) for x in self.learning_rate), 2)*0.01
    #Método que retorna el valor del Learning Rate como array de bits.
    def get_learning_rate(self):
        return self.learning_rate

    #Método que retorna el valor del Momentum como flotante.
    def get_momentum_2Float(self):
        return int("".join(str(x) for x in self.momentum ), 2)*0.01
    #Método que retorna el valor del Momentum como array de bits.
    def get_momentum (self):
        return self.momentum
    
    #Entrena a la red neuronal con los genes del individuo.
    def call_network(self):
        #Obtenemos una lista para pasarle el parámetro de capas ocultas a la red neuronal.
        capasOcultas = []
        for i in range (self.get_hidden_layers2Int()):
            capasOcultas.append(self.get_num_neurons2Int())
        #Aplicamos los hiperparámetros necesarios para generar el modelo de la red neuronal (instancia).
        mlp = MLPClassifier(hidden_layer_sizes=tuple(capasOcultas), activation='relu', learning_rate='constant', learning_rate_init=self.get_learning_rate2Float(), momentum=self.get_momentum_2Float(), solver='sgd', max_iter=self.get_num_epochs2Int(), verbose=False)
        _scoring=["accuracy"]
        #Entrenamos la red neuronal y validamos su desempeño con Cross Validation.
        scores = cross_validate(estimator=mlp,X=X,y=np.ravel(y),cv=K,return_train_score=True)
        #print(scores)
        """resultado=confusion_matrix(y_train,predict_train)
        diagonal = np.trace(resultado)
        accuracy = (diagonal / y_test.shape[0])*100
        print(classification_report(y_train,predict_train))
        print(f"Accuracy: {accuracy}%")"""
        accuracy=np.max(scores['test_score'])
        return accuracy

#Función que crea una poblacion inicial. Se retorna una arreglo de individuos.
def init_population(population_size):
    population=[]
    for i in range(population_size):
        population.append(Indivual())
    return population

#1.Se genera Poblacion (o posibles soluciones)
#2.Calculo del fitness (o costo) (seria el ACC)
#3.Seleccion de padres
#4.Crossover o cruza de los padres
#5.Mutacion en los hijos
#Repetir varias generaciones

if __name__ == '__main__':
    population_size=5
    target_column = ['clase'] 
    predictors = list(set(list(df.columns))-set(target_column))
    df[predictors] = df[predictors]/df[predictors].max()
    
    X = df[predictors].values
    y = df[target_column].values
    #Divide dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=40)
    
    fish=Indivual()
    #population=init_population(population_size)
    
