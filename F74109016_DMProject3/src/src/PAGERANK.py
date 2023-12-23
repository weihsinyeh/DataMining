from src.graph import Graph

def pagerank(graph, iteration=30 ,damping_factor=0.1):
    for _ in range(iteration):
        new_update = []
        for node in graph.get_node_list() : 
            name = node.name
            cur = graph.get_node_id(name)
            new_update.append(cur.update_pagerank(damping_factor,graph.get_node_num()))

        for node in graph.get_node_list() :
            # 統一更新
            name = node.name
            cur = graph.get_node_id(name)
            cur.pagerank = new_update[graph.get_node_index(name)]

        graph.normalize_pagerank()
    return graph