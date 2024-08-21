import numpy as np

from src.core.configurable import Configurable

class BaseManipulator(Configurable):
    
    def __init__(self, context, local_config, dataset):
        self.dataset = dataset
        super().__init__(context, local_config)
        
    def init(self):
        super().init()
        self.manipulated = False
        self._process()
         
    def _process(self):
        for instance in self.dataset.instances:
            node_features_map = self.node_info(instance)
            edge_features_map = self.edge_info(instance)
            graph_features_map = self.graph_info(instance)
            self.manipulate_features_maps((node_features_map, edge_features_map, graph_features_map))
            # overriding the features
            # resize in num_nodes x feature dim
            instance.node_features = self.__process_features(instance.node_features, node_features_map, self.dataset.node_features_map)
            instance.edge_features = self.__process_features(instance.edge_features, edge_features_map, self.dataset.edge_features_map)
            instance.graph_features = self.__process_features(instance.graph_features, graph_features_map, self.dataset.graph_features_map)

           

    def _process_instance(self,instance):
        node_features_map = self.node_info(instance)
        edge_features_map = self.edge_info(instance)
        graph_features_map = self.graph_info(instance)
        self.manipulate_features_maps((node_features_map, edge_features_map, graph_features_map))
        # overriding the features
        # resize in num_nodes x feature dim
        instance.node_features = self.__process_features(instance.node_features, node_features_map, self.dataset.node_features_map)
        instance.edge_features = self.__process_features(instance.edge_features, edge_features_map, self.dataset.edge_features_map)
        instance.graph_features = self.__process_features(instance.graph_features, graph_features_map, self.dataset.graph_features_map)

       
    def node_info(self, instance):
        return {}
    
    def graph_info(self, instance):
        return {}
    
    def edge_info(self, instance):
        return {}
    
    def manipulate_features_maps(self, feature_values):
        if not self.manipulated:
            node_features_map, edge_features_map, graph_features_map = feature_values
            self.dataset.node_features_map = self.__process_map(node_features_map, self.dataset.node_features_map)
            self.dataset.edge_features_map = self.__process_map(edge_features_map, self.dataset.edge_features_map)
            self.dataset.graph_features_map = self.__process_map(graph_features_map, self.dataset.graph_features_map)
            self.manipulated = True
    
    def __process_map(self, curr_map, dataset_map):
        _max = max(dataset_map.values()) if dataset_map.values() else -1
        for key in curr_map:
            if key not in dataset_map:
                _max += 1
                dataset_map[key] = _max
        return dataset_map
    
    def __process_features(self, features, curr_map, dataset_map):
        #print("features shape before processing:", features.shape if isinstance(features, np.ndarray) else "Not an ndarray")
        #print("curr_map:", curr_map)
        #print("dataset_map:", dataset_map)

        if curr_map:
            if not isinstance(features, np.ndarray):
                features = np.array([])
            try:
                old_feature_dim = features.shape[1]
            except IndexError:
                old_feature_dim = 0
            # If the feature vector doesn't exist, then
            # here we're creating it for the first time
            if old_feature_dim:
                features = np.pad(features,
                                pad_width=((0, 0), (0, len(dataset_map) - old_feature_dim)),
                                constant_values=0)
            else:
                features = np.zeros((len(list(curr_map.values())[0]), len(dataset_map)))
            #print("features shape after padding/initialization:", features.shape)
            
            for key in curr_map:
                index = dataset_map[key]
                try:
                    if features[:, index].shape != np.array(curr_map[key]).shape:
                        #print(f"Dimension mismatch for key {key}: reshaping curr_map[{key}] from shape {np.array(curr_map[key]).shape} to {features[:, index].shape}")
                        curr_map[key] = np.reshape(curr_map[key], features[:, index].shape)
                    features[:, index] = curr_map[key]
                except Exception as e:
                    #print(f"Error processing key {key}: {e}")
                    #print("Features: ",features)
                    #print("curr_map: ",curr_map)
                    try:
                        target_shape = features[:, index].shape
                        reshaped_array = np.zeros(target_shape)
                        reshaped_array[:np.array(curr_map[key]).shape[0]] = np.array(curr_map[key])
                        curr_map[key] = reshaped_array
                        features[:, index] = curr_map[key]
                    except Exception as inner_e:
                        print(f"Error forcing reshape for key {key}: {inner_e}")
                        print("Features: ", features)
                        print("curr_map: ", curr_map)
                        raise

        return features