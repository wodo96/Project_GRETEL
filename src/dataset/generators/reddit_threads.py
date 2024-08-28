import networkx as nx
import numpy as np
import zipfile
import os
from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance



class REDDIT(Generator):

    def init(self):

        #Import parameters from config file
        self.base_path = self.local_config['parameters']['data_dir']
        self.maxNodes = self.local_config['parameters']['max_nodes']
        self.number_of_graphs = self.local_config['parameters']['number_of_graphs']
        self.generate_dataset()

    def generate_dataset(self):
        print("igdfnv")

        zipped_file = (self.base_path + 'reddit_threads.zip')
        dir_path = self.base_path + 'reddit_threads/'

        #Unzip the file if it is not already unzipped
        if not os.path.exists(dir_path):
            with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
                zip_ref.extractall(self.base_path)

        A_path = dir_path + 'reddit_threads_A.txt'
        graph_indicator_path = dir_path + 'reddit_threads_graph_indicator.txt'
        graph_labels_path = dir_path + 'reddit_threads_graph_labels.txt'

        dict_graf = {}
        with open(graph_indicator_path, 'r') as f:
            graph_ind = np.array([int(line) for line in f])
    
        #from 1 to 203088
        dict_graf = {key: np.array([]).reshape([-1,2]) for key in np.unique(graph_ind)}
        #print(graphs)
        #{1: ....., 2: ......, 3: ....., ..., 203088: .....}


        #open attributes
        with open(A_path, 'r') as f:
            for line in f:
                node1, node2 = map(int, line.strip().split(', '))
                graph_id = graph_ind[node1-1]
                dict_graf[graph_id] = np.vstack([dict_graf[graph_id], [node1, node2]]).astype(int)


        '''print("norm: " + str(dict_graf[2]))
        print("max: " + str(dict_graf[2].max()))
        print("min: " + str(dict_graf[2].min()))'''


        '''
        Dict_graf[graph_ind[node1 - 1]] sarà uguale:
        quando node1 = 1, avremo:
            conteggio per graph_ind parte da zero, qundi si arriva a 0 fino a 10 (per quanto riguarda il contatore uguale a 1) che nel file (indicator.txt), parte da 1 e arriva ad 11.
            infatti dict_graph ha 20 elementi perché l'ultimo elemento sta sotto la soglia di (11-1) 

            dict_graph sarà composto da array di array (matrice) dove ogni elemento del primo array corrisponde alla coppia identificata nel file (Attributes) che nel primo caso avrà
            venti tuple perché il riferimento è il primo nodo della coppia. Considerando che ci sono 11 valori che corrispondono ad 1, il primo valore delle tuple dovrà essere minore od uguale
            a 11 per poter rientrare nella prima soglia.
        '''

        '''for i in range (len(dict_graf)):
            print(dict_graf[i+1])
            if (i==20):
                break'''

        #print("First element of dict_graph: " + str(dict_graf[graph_id][0][0]))



        #what's true and false
        with open (graph_labels_path) as f:
            graph_labels = np.array([int(line) for line in f])
        
        #gli elementi vanno da 0 a 203087
        #per ogni elemento di dict_graph, dobbiamo associare un valore di label
        #quindi tutti gli elementi del primo array di array di dict_graph avranno il peso/valore del primo label.

        print("Length of dict graph, equal to: " + str(len(dict_graf)))
        for i in np.arange(1, len(dict_graf)):
            if (i % 50000 == 0):
                print("Graphs processed: " + str(i))
            data = self.normalization_mat(np.asarray(dict_graf[i]))
            if data is not None:
                graphInstance = GraphInstance(i, graph_labels[i - 1], data)
                self.dataset.instances.append(graphInstance)
        

    def normalization_mat(self, data):
    
        #print ("Data: "+ str(data))
        max = data.max()
        min = data.min()
        #print("max: " + str(max) + " min: " + str(min))
        
        norm = (data - min).T
        #print(str(norm))

        min = min - 1 #to make it start from 0
        current_nodes = max - min
        
        #check if the current nodes of the single matrix is greather than the bound given
        if (current_nodes > self.maxNodes):
            return None

        matrix = np.zeros((self.maxNodes, self.maxNodes), np.int32)
        edges=[]
        for i in range(len(norm[0])):
            edges.append((norm[0][i],norm[1,i]))
        #print(list(edges))

        #build the bidirectional matrix
        for edge in edges:
                v1,v2=int(edge[0]),int(edge[1])
                matrix[v1,v2] = 1
                matrix[v2,v1] = 1
        #print(matrix)

        return matrix
