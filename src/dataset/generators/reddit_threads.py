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
        for key in np.arange(1, self.number_of_graphs):
            dict_graf[key] = np.array([]).reshape([-1,2])
            #dict_graf.update({graph_ind[key] : key})

        i=0
        with open(A_path, 'r') as f:
            for line in f:
                A = line.strip().split(', ')
                a_1 = int(A[0])
                a_2 = int(A[1])
                print("a_1: " + str(a_1) + " a_2: "+ str(a_2) + " i: " + str(i))
                g_i = graph_ind[a_1]
                print (g_i)
                dict_graf[g_i] = np.vstack([dict_graf[g_i], [a_1,a_2]])
                
                i+=1
                if (i == 20):
                    break

        for i in range (len(dict_graf)):
            print(dict_graf[i+1])
            if (i==20):
                break



        '''for i in range (self.number_of_graphs):
            for j in range (self.number_of_graphs):
                graph_ind[i][j] = A[j]'''

