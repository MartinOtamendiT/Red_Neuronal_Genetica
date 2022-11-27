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
#Los pasos del algoritmo serán los siguientes:
#   1.Se genera Población inicial (posibles soluciones).
#   2.Cálculo del fitness de cada individuo (Acc de la red neuronal).
#   3.Seleccion de los mejores individuos de la población (selección natural).
#   4.Cruza de los padres (reproducción).
#   5.Mutacion en los hijos.
#   6.Repetición de los pasos 2 a 5 por N generaciones en la población.

#*************************Importación de bibliotecas y módulos***********************
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
from sklearn.exceptions import ConvergenceWarning
from math import sqrt
import warnings
import os

#************************Procesos globales***************************
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
#Listas de mínimos, medias y máximos de fitness por generación.
mins = []
means = []
maxs = []

#*************************Clase individuo*********************************
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
            #Se verifica que el número de capas ocultas sea mayor a 0 y menor a 4.
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
            self.learning_rate = np.random.randint(2, size=5)
            #Conversión a decimal.
            aux_decimal= self.get_learning_rate2Float()
            #Se verifica que el learning rate esté entre 0.1 y 0.3.
            #   Si no es así, se vuelve a generar un valor al azar.
            while aux_decimal<=0 or aux_decimal>0.3:
                self.learning_rate = np.random.randint(2, size=5)
                aux_decimal=self.get_learning_rate2Float()

            #Se genera Momentum.
            self.momentum = np.random.randint(2, size=5)
            #Conversión a decimal.
            aux_decimal= self.get_momentum_2Float()
            #Se verifica que el momentum esté entre 0.1 y 1.
            #   Si no es así, se vuelve a generar un valor al azar.
            while aux_decimal<=0 or aux_decimal>0.3:
                self.momentum = np.random.randint(2, size=5)
                aux_decimal=self.get_momentum_2Float()
            
            #Se calcula el fitness del individuo.
            self.calculate_fitness()
        #Si hay algún cromosoma, se inicializan los genes con él.
        else:
            self.num_neurons = Chromosome[0]
            self.hidden_layers = Chromosome[1]
            self.num_epochs = Chromosome[2]
            self.learning_rate = Chromosome[3]
            self.momentum = Chromosome[4]

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
    #Método que retorna el valor del fitness del individuo.
    def get_fitness (self):
        return self.fit

    #Método que valida si los genes del individuo se encuentran dentro de los
    #   rangos establecidos para evaluar su aptitud.
    def validate_parameter(self):
        bandera = True
        if (self.get_num_neurons2Int() < 2 or self.get_num_neurons2Int() > 13):
            bandera = False
        elif (self.get_hidden_layers2Int() == 0 or self.get_hidden_layers2Int() > 4):
            BANDERA = False
        elif (self.get_num_epochs2Int == 0):
            bandera = False
        elif (self.get_learning_rate2Float() == 0 or self.get_learning_rate2Float() > 0.3):
            bandera = False
        elif (self.get_momentum_2Float() > 0.3):
            bandera = False
        return bandera

    #Retorna el cromosoma del individuo como cadena.
    def getChromosome(self):
        return str(f'Num_neurons: {self.get_num_neurons2Int()}\nHidden_layers: {self.get_hidden_layers2Int()}\nNum_epochs: {self.get_num_epochs2Int()}\nLearning_rate: {self.get_learning_rate2Float()}\nMomentum: {self.get_momentum_2Float()}\nFitness: {self.fit}')

    #Función fitness
    #Entrena a la red neuronal con los genes del individuo y regresa el
    #   Accuracy como fitness del individuo.
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
            x_train_fold, x_test_fold = X[train_i],X[test_i]
            #Divide el dataset en entrenamiento y prueba para el target.
            y_train_fold, y_test_fold = y[train_i], y[test_i]
            #Entrena red neuronal.
            mlp.fit(x_train_fold, np.ravel(y_train_fold))
            acc_list.append(mlp.score(x_train_fold, np.ravel(y_train_fold)))
            #Prueba red neuronal y guarda el resultado en acc_list.
            acc_list.append(mlp.score(x_test_fold, np.ravel(y_test_fold)))

        print('Lista de accuracies de test:', acc_list)
        #Toma como fitness el accuracy máximo de los obtenidos en los tests.
        accuracy=np.max(acc_list)
        self.fit=accuracy
    
    #Método que muta uno de los genes del individuo.
    def mutate(self):
        #Selecciona uno de los genes del individuo al azar para realizar la mutación.
        randGen = random.randint(0,4)
        
        #El gen seleccionado es el número de neuronas.
        if (randGen == 0):
            #Selecciona al azar uno de los bits del gen para mutarlo.
            randBit= random.randint(0, 3)
            #Si el bit tiene valor 0, se cambia a 1.
            if(self.num_neurons[randBit]==0):
                self.num_neurons[randBit] = 1
            #Si el bit tiene valor 1, se cambia a 0.
            else:
                self.num_neurons[randBit] = 0
        #El gen seleccionado es el número de capas ocultas.
        elif (randGen == 1):
            randBit = random.randint(0, 3)
            if(self.hidden_layers[randBit]==0):
                self.hidden_layers[randBit] = 1
            else:
                self.hidden_layers[randBit] = 0
        #El gen seleccionado es el número de épocas.
        elif (randGen == 2):
            randBit = random.randint(0, 9)
            if(self.num_epochs[randBit]==0):
                self.num_epochs[randBit]=1
            else:
                self.num_epochs[randBit] = 0
        #El gen seleccionado es el learning rate.
        elif (randGen == 3):
            randBit = random.randint(0, 4)
            if(self.learning_rate[randBit]==0):
                self.learning_rate[randBit]=1
            else:
                self.learning_rate[randBit] = 0
        #El gen seleccionado es el momentum.
        elif (randGen == 4):
            randBit = random.randint(0, 4)
            if(self.momentum[randBit]==0):
                self.momentum[randBit]=1
            else:
                self.momentum[randBit] = 0
            
#*********************************Clase población**********************************
#Permite crear una población, reproducirla y evolucionarla.
class Population():
    #Constructor que crea una poblacion inicial. Se crea un arreglo de individuos.
    def __init__(self, population_size):
        self.population_size=population_size
        self.population=[]
        self.generations=1
        for i in range(population_size):
            self.population.append(Indivual())

    #Método que selecciona a los mejores individuos de la población.
    def selection(self, n_selection):
        #Ordena a los individuos de la población de menor a mayor con base en sus fitness.
        sorted_population= sorted(self.population, key=lambda x: x.fit)
        #Selecciona los individuos con mejor fitness y los guarda en una lista.
        selected= sorted_population[len(sorted_population)-n_selection :]
        #Ordena de mayor a menor a los individuos seleccionados.
        self.selected= sorted(selected, key=lambda x: x.fit, reverse=True)
        #De la población, solo sobreviven los más aptos.
        self.population=self.selected
        #Actualiza el tamaño de la población después de la selección.
        self.population_size=len(self.population)
    
    #Método que reproduce a los individuos de la población.
    def reproduction(self, percentage_mutation):
        point=0
        parents=[]
        Chromosome1=[]
        Chromosome2=[]
        children=[]
        childrenMutated=[]
        childrenNoMutated=[] 
        
        #****************Comienza la cruza por cada individuo en el arreglo de seleccionados*******************
        #Se usara el método de One Point.
        for i in range(len(self.selected)):
            #Selecciona a dos individuos al azar para reproducirlos.
            parents=random.sample(self.selected, 2)

            #Define punto de separación.
            point = np.random.randint(1, 3)
            #Combina las estructuras de ambos padres para el gen: número de neuronas.
            Chromosome1.append(np.concatenate((parents[0].get_num_neurons()[:point],parents[1].get_num_neurons()[point:])))
            Chromosome2.append(np.concatenate((parents[1].get_num_neurons()[:point],parents[0].get_num_neurons()[point:])))
            
            #Define punto de separación.
            point = np.random.randint(1, 3)
            #Combina las estructuras de ambos padres para el gen: número de capas ocultas.
            Chromosome1.append(np.concatenate((parents[0].get_hidden_layers()[:point],parents[1].get_hidden_layers()[point:])))
            Chromosome2.append(np.concatenate((parents[1].get_hidden_layers()[:point],parents[0].get_hidden_layers()[point:])))

            #Define punto de separación.
            point = np.random.randint(1, 9)
            #Combina las estructuras de ambos padres para el gen: número de épocas.
            Chromosome1.append(np.concatenate((parents[0].get_num_epochs()[:point],parents[1].get_num_epochs()[point:])))
            Chromosome2.append(np.concatenate((parents[1].get_num_epochs()[:point],parents[0].get_num_epochs()[point:])))

            #Define punto de separación.
            point = np.random.randint(1, 6)
            #Combina las estructuras de ambos padres para el gen: learning rate.
            Chromosome1.append(np.concatenate((parents[0].get_learning_rate()[:point],parents[1].get_learning_rate()[point:])))
            Chromosome2.append(np.concatenate((parents[1].get_learning_rate()[:point],parents[0].get_learning_rate()[point:])))

            #Define punto de separación.
            point = np.random.randint(1, 6)
            #Combina las estructuras de ambos padres para el gen: momentum.
            Chromosome1.append(np.concatenate((parents[0].get_momentum()[:point],parents[1].get_momentum()[point:])))
            Chromosome2.append(np.concatenate((parents[1].get_momentum()[:point],parents[0].get_momentum()[point:])))
            
            #Nacen hijos de la feliz pareja.
            child1=Indivual(Chromosome1)
            child2=Indivual(Chromosome2)

            #Se verifica que los genes de los hijos sean válidos para entrar en la población
            #   y por ende esta crece. De lo contrario, los nuevos individuos se descartan
            #   dado que no podrían sobrevivir frente a las adversidades de la naturaleza.
            if(child1.validate_parameter()):
                child1.calculate_fitness()
                children.append(child1)
            if(child2.validate_parameter()):
                child2.calculate_fitness()
                children.append(child2)
            #Reinicio de cromosomas.
            Chromosome1=[]
            Chromosome2=[]
        
        #*************************Comienza el proceso de mutación**********************************
        #Se cálcula el número de hijos a mutar.
        numMutation = (len(children)*percentage_mutation)//100

        #Si el número de hijos a mutar es mayor a 0, entonces los muta.
        if (numMutation > 0):
            for i in range(numMutation):
                #Selecciona al azar un hijo.
                ranchild = random.randint(0, len(children)-1)
                #Muta al individuo.
                children[ranchild].mutate()
                #Se válida que la mutación sea beneficiosa y le permita sobrevivir.
                if(children[ranchild].validate_parameter()):
                    children[ranchild].calculate_fitness()
                    #Introduce al individuo mutado a la población seleccionada.
                    self.selected.append(children[ranchild])
                    #Se marca al individuo mutado.
                    childrenMutated.append(children[ranchild]) 
            #Los demás hijos no mutados son marcados (guardados en una lista).        
            childrenNoMutated=[i for i in children if i not in childrenMutated]
            #Después de ser marcados los no mutados, son apendizados en la población seleccionada.
            for individuo in childrenNoMutated:
                self.selected.append(individuo)
        #Si no hay individuos a mutar, se introducen a la población en automático.
        else:
            for i in range(len(children)):
                self.selected.append(children[i])
          
        #Población apta + hijos de población apta. Estos generarán la siguiente generación,
        #   y por ende son la nueva población.
        self.population=self.selected
        #Actualiza el número de generaciones.
        self.generations+=1

    #Guarda a los individuos de la población en un CSV (registro de la población).
    def population2Data(self):
        #Crea dataframe de generación.
        df=pd.DataFrame(columns=["num_neurons","hidder_layers","num_epochs","learning_rate","momentum", "fitness"])
        #Registra a cada individuo en el csv.
        for i in range(self.population_size):
            #Crea un diccionario para guardar las características importantes del individuo.
            individual={
                "num_neurons":self.population[i].get_num_neurons2Int(),
                "hidder_layers":  self.population[i].get_hidden_layers2Int(),
                "num_epochs": self.population[i].get_num_epochs2Int(),
                "learning_rate": self.population[i].get_learning_rate2Float(),
                "momentum": self.population[i].get_momentum_2Float(),
                "fitness": self.population[i].fit
            }
            #Apendiza los valores en el dataframe.
            df.loc[len(df)] = individual
        
        #Registra generación en un CSV.
        df.to_csv(f'./Generations/Generation_{self.generations}.csv', index=False)

    #Función que retorna el mejor resultado de fitness (accuracy) dentro de la población.
    def get_best(self):
        return max(individuo.fit for individuo in self.population)
    
    #Función que retorna el peor resultado de de fitness (accuracy) dentro de la población.
    def get_min(self):
        return min(individuo.fit for individuo in self.population)
    
    #Función que calcula la media de los fitness (accuracy) dentro de la población.
    def get_mean(self):
        fitness=[]
        for individuo in self.population:
            fitness.append(individuo.get_fitness())
        return np.mean(fitness)
    
    # Método que retorna el tamaño de la población al momento.
    def get_population_size(self):
        return self.population_size
    
    #Método que retorna el número de generaciones de la población.
    def get_generations(self):
        return self.generations
                 
#************************************************Algoritmo genético******************************************
def genetics(population_size, n_selection, percentage_mutation):
    print("*** Generando poblacion ***")
    #Se genera población inicial.
    ParametrosRNA = Population(population_size)
    i=1
    #Bucle que selecciona y reproduce a los individuos de la población N veces.
    while True:
        print(f"**********Generacion {i}**************")
        #************ Se seleccionan a los mejores individuos de la población*******************
        ParametrosRNA.selection(n_selection)
        #Guarda generación en CSV.
        ParametrosRNA.population2Data()
        #Se imprimen detalles de la población.
        print(f"Tamaño de la población: {ParametrosRNA.get_population_size()}")
        print(f"Máximo Fitness: {ParametrosRNA.get_best()}")
        maxs.append(ParametrosRNA.get_best())
        print(f"Media Fitness: {ParametrosRNA.get_mean()}")
        means.append(ParametrosRNA.get_mean())
        print(f"Minimo Fitness: {ParametrosRNA.get_min()}")
        mins.append(ParametrosRNA.get_min())
        
        #Se pregunta por la creación de una siguiente población.
        decision = input("¿Desea crear otra generación? s/n \n")
        #Si no se quiere crear una nueva población, el algoritmo termina.
        if (decision == 'n'):
            break
        #Si se crea otra población, se realiza la reproducción entre padres y la mutación de los hijos.
        ParametrosRNA.reproduction(percentage_mutation)
        i+=1
    
    #Al finalizar el algoritmo, se muestran los resultados obtenidos.
    print("Algoritmo finalizando. Mostrando resultados........")
    #Se grafican el mínimo, la media y el máximo fitness por generación.
    graficar(i)
    print(f"Min: {mins[i-1]}, Mean:{means[i-1]}, Max:{maxs[i-1]}")
    print(f"El cromosoma del individuo mas apto fue:\n{ParametrosRNA.population[0].getChromosome()}")

#Método para graficar las estadísticas de los acc por generación.
def graficar(generations):
    #Plotea los puntos indicados por generación.
    for gen in range(generations):
        y = []
        y.append(mins[gen])
        y.append(means[gen])
        y.append(maxs[gen])
        x = [gen+1 for num in range(len(y))]
        plt.plot(x,y,"o-")
    
    #Muestra gráfica final.
    plt.show()
        
#***************************************Módulo principal********************************************
if __name__ == '__main__':
    #Línea que evita mostrar los warnings generados durante el entrenamiento de las redes neuronales.
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    #Crea una carpeta para guardar los csv de las generaciones de la población en caso de no existir.
    if not os.path.exists('./Generations'):
        os.mkdir('./Generations')
        print("Se ha creado el directorio Generations")
    #Se corre el algoritmo genético.  
    genetics(population_size=50,n_selection=25,percentage_mutation=20)
    
