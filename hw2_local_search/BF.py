city_num = 0
map_data = []
import itertools
def dfs(start, visited, path, dist, best_dist, best_path, city_visited):
    path.append(start)
    city_visited[start] = 1
    visited += 1    
    if visited == city_num:
        dist = dist + map_data[path[0]][start]
        if dist < best_dist[0]:  # 使用列表来跟踪最佳距离，以便在函数内部修改
            best_dist[0] = dist 
            path.append(path[0])
            best_path[0] = path.copy()   
            path.pop()  
    else:
        for i in range(0, city_num):
            if city_visited[i] == 0:
                dist += map_data[start][i]
                dfs(i, visited, path, dist , best_dist, best_path, city_visited)
                dist -= map_data[start][i]

    visited -= 1
    city_visited[start] = 0
    path.pop()

def BF(input_list):
    global city_num
    city_num = len(input_list)
    global map_data
    map_data = input_list

    best_dist = [float('inf')]  # 使用列表来跟踪最佳距离
    best_path = [[]]

    for i in range(city_num):
        visited = 0
        path = []
        city_visited = [0] * city_num
        dfs(i, visited, path, 0, best_dist, best_path, city_visited)

    return best_dist[0],best_path[0]

def test(input_list):
    # 排列組和 0- n-1
    data = list(itertools.permutations(range(len(input_list))))
    min = float('inf')
    path = []
    for item in data:
        sum = 0
        for j in range(1,input_list.__len__()):
            sum += input_list[item[j-1]][item[j]]
        sum += input_list[item[input_list.__len__()-1]][item[0]]
        if sum < min:
            if(sum == 16) : print(item)
            min = sum
            path = item.append(item[0])
    #print(min)
    return path,min

