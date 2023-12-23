from src.graph import Graph

def hits(graph, iteration=30):
    for _ in range(iteration):
        auth = []
        hub = []
        for node in graph.get_node_list() : 
            name = node.name
            cur = graph.get_node_id(name)
            auth.append(cur.update_auth())
        for node in graph.get_node_list() : 
            name = node.name
            cur = graph.get_node_id(name)
            hub.append(cur.update_hub())
        for node in graph.get_node_list() :
            # 統一更新
            name = node.name
            cur = graph.get_node_id(name)
            cur.auth = auth[graph.get_node_index(name)]
            cur.hub = hub[graph.get_node_index(name)]
        graph.normalize_hits()
    return graph
