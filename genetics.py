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
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_validate,KFold,cross_val_score,StratifiedKFold
from math import sqrt

#Lectura del dataframe de forma global.
df=pd.read_csv("descriptoresFinal.csv")
#Columna a predecir.
target_column = ['clase']
#Lista de columnas que fungirán como predictores.
predictors = list(set(list(df.columns))-set(target_column))
#Normalización de los predictores (rango 0 a 1).
df[predictors] = df[predictors]/df[predictors].max()

#Arreglo de los valores de las variables independientes (entrada).
X = df[predictors].values
#Arreglo de los valores de la variable de clase (dependiente/salida).
y = df[target_column].values
#Determinamos K para el algoritmo Stratified KFolds Cross Validation.
K=2
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)

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
            self.hidden_layers = np.random.randint(2, size=4)
            #Conversión a decimal.
            aux_decimal= self.get_hidden_layers2Int()
            #Se verifica que el número de capas ocultas sea mayor a 0.
            #   Si no es así, se vuelve a generar un valor al azar.
            while aux_decimal==0 or aux_decimal>4:
                self.hidden_layers = np.random.randint(2, size=4)
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
        
        #Se calcula el fitness del individuo.
        self.calculate_fitness()
    
    #Método que retorna el valor del número de neuronas como entero.
    def get_num_neurons2Int(self):
        return int("".join(str(x) for x in self.num_neurons), 2)
    #Método que retorna el valor del número de neuronas como array de bits.
    def get_num_neurons(self):
        return self.num_neurons

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
    def printChromosome(self):
        print(f'**Individuo**\nNum_neurons: {self.get_num_neurons2Int()}\nHidden_layers: {self.get_hidden_layers2Int()}\nNum_epochs: {self.get_num_epochs2Int()}\nLearning_rate: {self.get_learning_rate2Float()}\nMomentum: {self.get_momentum_2Float()}\nFitness: {self.fit}')

        
    
    #Función fitness
    #Entrena a la red neuronal con los genes del individuo y regresa el
    # Acc como fitness del individuo.
    def calculate_fitness(self):
        #Obtenemos una lista para pasarle el parámetro de capas ocultas a la red neuronal.
        capasOcultas = []
        acc_list=[]
        for i in range (self.get_hidden_layers2Int()):
            capasOcultas.append(self.get_num_neurons2Int())
        #Aplicamos los hiperparámetros necesarios para generar el modelo de la red neuronal (instancia).
        mlp = MLPClassifier(hidden_layer_sizes=tuple(capasOcultas), activation='relu', learning_rate='constant', learning_rate_init=self.get_learning_rate2Float(), momentum=self.get_momentum_2Float(), solver='sgd', max_iter=self.get_num_epochs2Int(), verbose=False)
        #Entrenamos mediante validación cruzada.
        for train_i, test_i in skf.split(X, y):
            #Divide el dataset en entrenamiento y prueba para los predictores.
            x_train_fold, x_test_fold = X[train_i],X[train_i]
            #Divide el dataset en entrenamiento y prueba para el target.
            y_train_fold, y_test_fold = y[test_i], y[test_i]
            #Entrena red neuronal.
            mlp.fit(x_train_fold, np.ravel(y_train_fold))
            #Prueba red neuronal y guarda el resultado en acc_list.
            acc_list.append(mlp.score(x_test_fold, np.ravel(y_test_fold)))

        #print('List of possible accuracy:', acc_list)
        #Considera el acc máximo
        accuracy=np.max(acc_list)
        self.fit=accuracy

class Population():
    #Constructor que crea una poblacion inicial. Se retorna una arreglo de individuos.
    def __init__(self, population_size):
        self.population=[]
        for i in range(population_size):
            self.population.append(Indivual())

    #Selecciona a los mejores individuos de la población.
    def selection(self, n_selection):
        #Ordena a los individuos de la población de menor a mayor con base en sus fitness.
        sorted_population= sorted(self.population, key=lambda x: x.fit)
        #Selecciona los individuos con mejor fitness y los guarda en una lista
        selected= sorted_population[len(sorted_population)-n_selection :]
        #Ordena de mayor a menor a los individuos seleccionados.
        self.selected= sorted(selected, key=lambda x: x.fit, reverse=True)
    
    def reproduction(self):
        point=0
        parents=[]
        
        for individuo in self.selected:
            #Selecciona dos padres al azar.
            parents=random.sample(self.selected, 2)

            #Define punto de separación.
            point = np.random.randint(1, 3)
            #Combina las estructuras de ambos padres para el gen: número de neuronas.
            individuo.get_num_neurons()[:point]=parents[0].get_num_neurons()[:point]
            individuo.get_num_neurons()[point:]=parents[1].get_num_neurons()[point:]

            #Define punto de separación.
            point = np.random.randint(1, 3)
            #Combina las estructuras de ambos padres para el gen: número de neuronas.
            individuo.get_hidden_layers()[:point]=parents[0].get_hidden_layers()[:point]
            individuo.get_hidden_layers()[point:]=parents[1].get_hidden_layers()[point:]

            #Define punto de separación.
            point = np.random.randint(1, 9)
            #Combina las estructuras de ambos padres para el gen: número de neuronas.
            individuo.get_num_epochs()[:point]=parents[0].get_num_epochs()[:point]
            individuo.get_num_epochs()[point:]=parents[1].get_num_epochs()[point:]

            #Define punto de separación.
            point = np.random.randint(1, 6)
            #Combina las estructuras de ambos padres para el gen: número de neuronas.
            individuo.get_learning_rate()[:point]=parents[0].get_learning_rate()[:point]
            individuo.get_learning_rate()[point:]=parents[1].get_learning_rate()[point:]

            #Define punto de separación.
            point = np.random.randint(1, 6)
            #Combina las estructuras de ambos padres para el gen: número de neuronas.
            individuo.get_momentum()[:point]=parents[0].get_momentum()[:point]
            individuo.get_momentum()[point:]=parents[1].get_momentum()[point:]

            individuo.calculate_fitness()
            """print(f"Punto:{point}")
            print(f"Padre {parents[0].get_num_epochs()}")
            print(f"Madre {parents[1].get_num_epochs()}")
            print(f"Nuevo gen: {individuo.get_num_epochs()}")"""
            self.population.append(individuo)
            #population[i][:point] = father[0][:point]
            #population[i][point:] = father[1][point:]
        #self.population=self.selected
    
    def get_best(self):
        return max(individuo.fit for individuo in self.population)

    def get_min(self):
        return min(individuo.fit for individuo in self.population)
    #Checar
    def get_mean(self):
        return np.mean(individuo.fit for individuo in self.population)
            
            


#Algoritmo genético.
def genetics(population_size, n_generations):
    print("Generando poblacion")
    ParametrosRNA = Population(population_size)
    for i in range(n_generations):
        print(f"**********Generacion {i}**************")
        ParametrosRNA.selection(10)
        ParametrosRNA.reproduction()
        #print(ParametrosRNA.population[8].printChromosome())
        #print(ParametrosRNA.population[9].printChromosome())
        print(len(ParametrosRNA.population))
        print(ParametrosRNA.get_best())
        print(ParametrosRNA.get_min())




#1.Se genera Poblacion (o posibles soluciones)-
#2.Calculo del fitness (o costo) (seria el ACC)-
#3.Seleccion de padres
#4.Crossover o cruza de los padres
#5.Mutacion en los hijos
#Repetir varias generaciones

if __name__ == '__main__':
    genetics(population_size=20, n_generations=2)
    #fish=Indivual()
    #print(fish.get_num_epochs())
    
