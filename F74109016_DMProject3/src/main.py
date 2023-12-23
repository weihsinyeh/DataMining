import numpy as np
import os
import time
import matplotlib.pyplot as plt
from optparse import OptionParser
from src.HITS import hits
from src.PAGERANK import pagerank
from src.SIMRANK import simrank , Similaryity
from src.process_file import init_graph , save_auth_hub_file , save_pagerank_file , save_SimRank
# Read graph data at {path} and return an adjacent matrix of that graph.
# every node is a class
time_HITs = []
time_PageRank = []
time_SimRank = []
file_name_list = []
i = 0
def UseHit(file_path,iteration=30):
    # start
    time_start = time.time()
    graph = init_graph(file_path)
    graph = hits(graph, iteration) 
    # end
    time_end = time.time()
    time_HITs[i] = time_end - time_start
    save_auth_hub_file(graph,file_path)
   

def UsePageRank(file_path,iteration=30,damping_factor=0.1):
    # start
    time_start = time.time()
    graph = init_graph(file_path)
    # make every node in graph's pagerank = 1 / N
    for node in graph.get_node_list() : 
        node.pagerank = 1. / graph.get_node_num()
    graph = pagerank(graph, iteration, damping_factor) 
    # end
    time_end = time.time()
    time_PageRank[i] = time_end - time_start
    '''
    print("damping_factor = ",damping_factor)
    ls = []
    for node in graph.get_node_list() : 
        ls.append(round(node.pagerank,3))
    print(ls)
    '''
    save_pagerank_file(graph,file_path)
    

def UseSimRank(file_path,iteration=30,decay_factor=0.7):
    # start
    start_time = time.time()
    graph = init_graph(file_path)
    sim_matrix = Similaryity(graph,decay_factor)
    sim_matrix = simrank(graph,sim_matrix, iteration, decay_factor)
    # end
    end_time = time.time()
    time_SimRank[i] = end_time - start_time
    '''
    print("decay_factor = ",decay_factor)
    print (sim_matrix)
    '''
    save_SimRank(sim_matrix,file_path)

# https://hackmd.io/QrUCc51DT32iiJBelEbEdQ

if __name__ == '__main__' :
    optparser = OptionParser()
    optparser.add_option('-f', '--file', dest='input_file', help='CSV_filename', default='Dataset/graph_1.txt')
    optparser.add_option('--iteration', dest='iteration', help='iteration (int)', default=30, type = 'int')

    (options, args) = optparser.parse_args()
    file_path = options.input_file
    iteration = options.iteration
    file_name_list = ['graph_1.txt','graph_2.txt','graph_3.txt','graph_4.txt','graph_5.txt','graph_6.txt','ibm-5000.txt']
  
    time_HITs = [0] * len(file_name_list)
    time_PageRank = [0] * len(file_name_list)
    time_SimRank = [0] * len(file_name_list)
    result_dir = 'result'
    fname = file_path.split('/')[-1].split('.')[0]
    #damping_factor = 0.1
    #decay_factor = 0.7
    #iteration = 30
    UseHit(file_path,iteration=30)
    UsePageRank(file_path,iteration=30,damping_factor=0.1)
    UseSimRank(file_path,iteration=30,decay_factor=0.7)

    folder = "/home/weihsin/projects/DM/HW3/Dataset"

    
    for file in file_name_list:
        file_path = folder + "/" + file
        file_name = file_path.split('/')[-1].split('.')[0]
        print(file_name)
        UseHit(file_path,iteration=30)
        UsePageRank(file_path,iteration=30,damping_factor=0.1)
        if( file_name != 'graph_6' and file_name != 'ibm-5000') : 
            UseSimRank(file_path,iteration=30,decay_factor=0.7)
        i+=1
    print("time_HITs = ",time_HITs)
    print(time_HITs)
    print("time_PageRank = ",time_PageRank)
    print(time_PageRank)
    print("time_SimRank = ",time_SimRank)
    print(time_SimRank)
    '''
    plt.figure(figsize=(10,5))
    plt.plot(file_name_list,time_HITs,label='HITS')
    plt.plot(file_name_list,time_PageRank,label='PageRank')
    plt.plot(file_name_list,time_SimRank,label='SimRank')
    plt.xlabel('graph')
    plt.ylabel('time')
    plt.legend()
    plt.savefig('time3.png') 
    '''
        
    