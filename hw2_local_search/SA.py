import random
import math
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn
from sklearn.manifold import MDS
import numpy as np
def Inversion(path,length):
    path_copy = path.copy()
    place  = random.sample(range(1, length),2)
    path_copy[place[0]],path_copy[place[1]] = path_copy[place[1]],path_copy[place[0]]
    return path_copy
def TtotalDistance(distmap,path):
    distance=0
    for i in range(0,len(path)-1):
        distance += distmap[path[i]][path[i+1]]
    return distance
def SA(distmap,t0,tmin,iter,coolnum) :
    length = distmap[0].__len__() # city length
    evetime_distance ,evetime_route,every_temperature= [],[],[]
    t = t0
    while True:
        path = random.sample(range(0, length), length)# randomly pass route
        path.append(path[0])  # add back path
        total_distance = TtotalDistance(distmap,path) # add a distance and update it subsequently
        if t <= tmin: break
        for times in range(iter):
            new_path = Inversion(path,length)
            new_distance = TtotalDistance(distmap,new_path)
            difference = total_distance - new_distance 
            if difference > 0:  # change smaller direct replace
                path = new_path
                total_distance = new_distance
            else:
                # diff/t -> 0 , prob -> 1    
                # t decrease , prob decrease  
                # big diff , prob small
                # small diff , prob big
                prob = math.exp(difference/t) 
                randnum = random.uniform(0,1)
                if randnum < prob:
                    path = new_path
                    total_distance = new_distance
                
        evetime_route.append(path)
        evetime_distance.append(total_distance)
        every_temperature.append(t)
        t = t * coolnum
    #### draw the process of distance of every iteration
    #### You can see the process of Simulated Annealing
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
    ax1.set_xlabel("Iteration", fontsize=15)
    ax1.set_ylabel("Distance", fontsize=15)
    ax1.plot(evetime_distance, linewidth=2.5, label="Everytime smallest distance", color='dodgerblue')
    ax1.legend()
    # graph_name = 'SA_city_'+str(length) +'_t_'+str(t0)+'_tmin_'+str(tmin)+'_iter_'+str(k)+'_cool_'+str(coolnum)+'.png' 
    ax1.set_title('SA_city:'+str(length) +' t:'+str(t0)+' tmin:'+str(tmin)+' iter:'+str(iter)+' cool:'+str(coolnum)+'.png' )

    mds = MDS(n_components=2, dissimilarity="precomputed")
    city_coordinates = mds.fit_transform(distmap)
    ax2.scatter(city_coordinates[:, 0], city_coordinates[:, 1], color='blue', s=100)
    start = True
    for i, (x, y) in enumerate(city_coordinates):
        if start:
            ax2.text(x, y, f'Start City {i}', fontsize=12)
            start = False
        else:
            ax2.text(x, y, f'City {i}', fontsize=12)

    path_coordinates = np.array(evetime_route[-1])
    for i in range(len(path_coordinates) - 1):
        start = path_coordinates[i]
        end = path_coordinates[i + 1]
        ax2.plot([city_coordinates[start][0], city_coordinates[end][0]],[city_coordinates[start][1], city_coordinates[end][1]], color='dodgerblue', linewidth=5)

    ax2.set_title("City Coordinates with Shortest Path")
    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")
    ax2.grid()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig('SA_city'+str(length) +'_t_'+str(t0)+'_tmin_'+str(tmin)+'_iter_'+str(iter)+'_cool_'+str(coolnum)+'.png' )
    '''
    return evetime_route[-1],evetime_distance[-1]

    

   
