from src.graph import Graph
import numpy as np
class Similaryity:
    def __init__(self,graph,decay_factor=0.7):
        self.graph = graph
        self.decay_factor = decay_factor
        self.new_sim_matrix = np.zeros((self.graph.get_node_num(),self.graph.get_node_num()))
        self.old_sim_matrix = np.zeros((self.graph.get_node_num(),self.graph.get_node_num()))
        self.init_sim_matrix()

    def init_sim_matrix(self):
        for i in range(self.graph.get_node_num()):
            for j in range(self.graph.get_node_num()):
                if i == j : self.old_sim_matrix[i][j] = 1
                else :      self.old_sim_matrix[i][j] = 0
        for i in range(self.graph.get_node_num()):
            for j in range(self.graph.get_node_num()):
                if i == j : self.new_sim_matrix[i][j] = 1
                else :      self.new_sim_matrix[i][j] = 0
        
    def calculate_simrank(self,node1,node2):
        if node1 == node2 : return 1

        node1_object = self.graph.find(node1)
        node2_object = self.graph.find(node2)
        parents1 = node1_object.get_parents()
        parents2 = node2_object.get_parents()
        if(parents1 == [] or parents2 == []): return 0.0

        SimRank = 0

        for parent1 in parents1:
            for parent2 in parents2:
                parent1_index = self.graph.get_node_index(parent1.name)
                parent2_index = self.graph.get_node_index(parent2.name)
                SimRank += self.old_sim_matrix[parent1_index][parent2_index]
        return SimRank * ( self.decay_factor / (len(parents1) * len(parents2)))

    def update_sim_matrix(self,node1_index,node2_index,new_SimRank):
        self.new_sim_matrix[node1_index][node2_index] = new_SimRank


def simrank(graph,sim_matrix,iteration=30,decay_factor=0.7):
    for _ in range(iteration):
        for node1 in graph.get_node_list() : 
            for node2 in graph.get_node_list() :
                cur1 = graph.get_node_index(node1.name)
                cur2 = graph.get_node_index(node2.name)
                new_SimRank = sim_matrix.calculate_simrank(cur1,cur2)
                sim_matrix.update_sim_matrix(cur1,cur2,new_SimRank)

        sim_matrix.old_sim_matrix = sim_matrix.new_sim_matrix.copy()
    return sim_matrix.old_sim_matrix

