import numpy as np 
class Graph:
    def __init__(self):
        self.nodes = []

    def sort_nodes(self):
        self.nodes.sort(key=lambda x: int(x.name))

    def exist(self, name):
        for node in self.nodes : 
            if node.name == name : return node
        new_node = Node(name)
        self.nodes.append(new_node)
        return new_node

    def find(self, name):
        return self.nodes[name]

    def add_node(self, parent, child):
        parent_node = self.exist(parent)
        child_node = self.exist(child)
        parent_node.link_child(child_node)
        child_node.link_parent(parent_node)

    def print_auth(self):
        a = np.array([])
        self.sort_nodes()
        for node in self.nodes : a = np.append(a,round(node.auth,3))
        return a

    def print_hub(self):
        a = np.array([])
        self.sort_nodes()
        for node in self.nodes : a = np.append(a,round(node.hub,3))
        return a

    def print_page_rank(self):
        a = np.array([])
        # sort nodes by name from number small to large
        self.sort_nodes()
        for node in self.nodes : a = np.append(a,round(node.pagerank,3))
        return a

    def get_node_list(self):
        return self.nodes

    def get_node_num(self):
        return len(self.nodes)

    def get_node_id(self,name):
        for node in self.nodes : 
            if node.name == name : return node
        return None
    def normalize_hits(self):
        auth_sum = 0.
        hub_sum = 0.
        for node in self.nodes : 
            auth_sum += node.auth
            hub_sum += node.hub
        for node in self.nodes : 
            node.auth = node.auth / auth_sum
            node.hub = node.hub / hub_sum

    def get_node_index(self, name):
        for i in range(len(self.nodes)):
            if self.nodes[i].name == name : 
                return i
        return None

    def normalize_pagerank(self):
        pagerank_sum = 0.
        for node in self.nodes : 
            pagerank_sum += node.pagerank
        for node in self.nodes : 
            if pagerank_sum == 0. : break
            node.pagerank = node.pagerank / pagerank_sum

class Node:
    def __init__(self,name):
        self.name = name
        self.parents = []
        self.children = []
        self.auth = 1.
        self.hub = 1.
        self.pagerank = 1.

    def get_parents(self):
        return self.parents

    def link_child(self, new_child):
        self.children.append(new_child)

    def link_parent(self, new_parent):
        self.parents.append(new_parent)

    ############### HITS update ###############
    def update_auth(self):
        auth = 0.
        for parent in self.parents : 
            auth += parent.hub 
        return auth

    def update_hub(self):
        hub = 0.0
        for child in self.children : 
            hub += child.auth   
        return hub

    ############### PAGERANK update ###############
    def update_pagerank(self, damping_factor , numOfNodes):
        pagerank = 0.0
        for parent in self.parents : 
            pagerank += parent.pagerank / len(parent.children)
        random_jumping = damping_factor / numOfNodes
        return random_jumping + (1-damping_factor) * pagerank
        
     
