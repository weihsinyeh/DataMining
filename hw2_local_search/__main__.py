import BF 
import SA 
import random
import os
import time
def create_2Darray(size):
    array = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(i,size):
            if i == j:
                array[i][j] = 0
            else:
                array[i][j] = random.randint(1,60)
                array[j][i] = array[i][j]
    file_name = 'map'+str(size)+'.txt'
    with open(os.path.join(os.path.dirname(__file__), file_name), 'w') as f : 
        f.write(str(array))
    return array
if __name__ == '__main__':
    ## You can create a 2D array map by given a size parameter
    city_num = 5
    ''''
    create_2Darray(city_num)
    '''
    map_name = './map'+str(city_num)+'.txt'
    f = open(map_name, 'r')
    data = f.read()
    array = eval(data)
    print("city num:",len(array))
    print("########################## Brute Force:")
    start = time.time()
    ### Way 1 : DFS
    best_dist,best_path = BF.BF(array)
    ### Way 2 : validation permutation
    #best_path,best_dist =BF.test(array)
    end = time.time()
    print("best_path: ",best_path)
    #distance=0
    #for i in range(0,len(best_path)-1):
    #    distance += array[best_path[i]][best_path[i+1]]
    #print("distance: ",distance)
    print("best_dist: ",best_dist)
    print("elapsed time: ",end-start)
    
    print("########################## Simulated Annealing:")
    t0 = 50000000    #Initial temperature
    tmin = 0.1  #End of iteration, which means minimum temperature
    iter = 100      #每次降溫內迭代次數
    coolrate  = 0.98 # cooling rate
    start = time.time()
    best_path,best_dist = SA.SA(array,t0,tmin,iter,coolrate)
    distance = 0
    #for i in range(0,len(best_path)-1):
    #    distance += array[best_path[i]][best_path[i+1]]
    #print("distance: ",distance)
    end = time.time()
    print("best_path: ",best_path)
    print("best_dist: ",best_dist)
    print("elapsed time: ",end-start)
