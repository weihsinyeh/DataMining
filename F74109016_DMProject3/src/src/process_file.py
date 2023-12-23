from src.graph import Graph
import numpy as np
def init_graph(fname):
    # if fname start from ibm
    if fname.split('/')[-1].split('.')[0].startswith('ibm'):
        with open(fname) as f:
            lines = f.readlines()
        graph = Graph()
        for line in lines:
            [_, parent, child] = line.strip().split('      ')
            graph.add_node(parent,child)
            
        graph.sort_nodes()
        return graph
    else:
        with open(fname) as f:
            lines = f.readlines()
        graph = Graph()
        for line in lines:
            [parent, child] = line.strip().split(',')
            graph.add_node(parent,child) 
        graph.sort_nodes()
        return graph    

def save_auth_hub_file(graph,fname):
    file_name = fname.split('/')[-1].split('.')[0]
    fname = 'results/' +file_name+'/'+ file_name + '_HITS_authority.txt'

    array = graph.print_auth()

    np.savetxt(fname, array, delimiter=' ', fmt='%.3f', newline=' ')
    fname = 'results/' +file_name+'/'+ file_name + '_HITS_hub.txt'
    array = graph.print_hub()
    np.savetxt(fname, array, delimiter=' ', fmt='%.3f', newline=' ')
    
def save_pagerank_file(graph,fname):
    file_name = fname.split('/')[-1].split('.')[0]
    fname = 'results/' +file_name+'/'+ file_name + '_PageRank.txt'
    array = graph.print_page_rank()
    np.savetxt(fname, array, delimiter=' ', fmt='%.3f', newline=' ')


def save_SimRank(sim_matrix,fname):
    file_name = fname.split('/')[-1].split('.')[0]
    fname = 'results/' +file_name+'/'+ file_name + '_SimRank.txt'
    sim_matrix = np.round(np.asarray(sim_matrix), 3)
    np.savetxt(fname, sim_matrix, delimiter=' ', fmt='%.3f')
