import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import resource
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
def read_data(data_file):
    # Read the data from the file
    data = pd.read_csv(data_file)
    values = data.values
    # print(data.head())
    # print(values.shape)
    # print(values[1:5,])
    """
       FromNodeId  ToNodeId
    0      237603     62478
    1      202854    227760
    2       19861    258311
    3        7106    270771
    4      137046    275894
    (2300000, 2)
    [[202854 227760]
    [ 19861 258311]
    [  7106 270771]
    [137046 275894]]
    """
    #print(np.max(values), np.min(values))
    print("max id node: ", np.max(values), "min id node: ", np.min(values))

    ## there are 30 nodes not in the graph
    ## there some node that not in the graph
    ## rearrange the node id from 0 to node_num-1
    
    node_ids = np.unique(values)
    node_num = len(node_ids)
    print("node number: ", node_num)
    node_ids_map = {}
    for i in range(node_num):
        node_ids_map[node_ids[i]] = i ## map the node id to 0 to node_num-1
    graph = []
    for item in values:
        graph.append([node_ids_map[item[0]], node_ids_map[item[1]]])

    return np.array(graph), node_ids

def draw_errors(errors, save_name):
    
    plt.plot(errors)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    # plt.title("Convergence of PageRank")
    plt.savefig(save_name)
    plt.close()

def save_top_1000_pagerank(pagerank, node_ids_map ,save_file):
    ## pagerank is a list , each element is the pagerank of the node
    rank = np.argsort(pagerank)[::-1]
    top_nodes = []
    for i in range(1000):
        top_nodes.append((node_ids_map[rank[i]], round(pagerank[rank[i]], 6)))
    
    ## use pandas to save the data to csv file
    df = pd.DataFrame(top_nodes, columns=["NodeId", "PageRank_Value"])
    df.to_csv(save_file, index=False)

def standard_pagerank(graph, beta, iters=100):
    node_num = np.max(graph)  ## the number of nodes, id from 1 to node_num

    ## compute the out degree of each node
    out_degree = np.zeros(node_num, dtype=np.float32)
    for item in graph:
        out_degree[item[0]-1] += 1
    
    ## compute the M
    M = np.zeros((node_num, node_num), dtype=np.float32) # out of memory

    for item in graph:
        M[item[1]-1, item[0]-1] = beta*1/out_degree[item[0]-1] ## i->j, M[j,i] = 1/d(i) ##colum sum is 1*beta

    ## add randam jump
    M += (1-beta)/node_num ## add random jump, avoid dead end and spider trap

    ## compute the pagerank
    pagerank = np.ones(node_num, dtype=np.float32)/node_num

    ## iteration
    errors = []
    for i in tqdm(range(iters)):
        pagerank1 = np.dot(M, pagerank)
        error = np.linalg.norm(pagerank1 - pagerank)
        errors.append(error)

    return pagerank, errors

def sparse_pagerank2(graph, beta=0.85, iters=100):
    
    
    node_num = np.max(graph)+1  ## the number of nodes, id from 1 to node_num
    # create the sparse matrix
    row_indices = [x for x, y in graph]
    col_indices = [y for x, y in graph]
    data = np.ones(len(graph))
    M = csr_matrix((data, (col_indices, row_indices)), shape=(node_num, node_num))
    
    # compute the out degree of each node
    out_degree = np.array(M.sum(axis=0)).flatten()
    out_degree[out_degree != 0] = 1.0 / out_degree[out_degree != 0]
    M = M.multiply(out_degree)
    
    # compute pagerank
    pagerank = np.ones(node_num) / node_num
    errors = []
    teleport = (1 - beta) / node_num
    for i in tqdm(range(iters)):
        pagerank_new = beta * M.dot(pagerank) + teleport

        ## deal with the dead end, assure the sum of pagerank is 1
        pagerank_new += (1-np.sum(pagerank_new))/node_num

        ## compute the error
        error = np.linalg.norm(pagerank_new - pagerank)
        errors.append(error)
        pagerank = pagerank_new
    
    return pagerank, errors

## use the sparse matrix to store the M to save memory
def sparse_pagerank1(graph, beta=0.85, iters=100):
    node_num = np.max(graph)+1  ## the number of nodes, id from 0 to node_num-1

    ## compute the out degree of each node
    out_degree = np.zeros(node_num, dtype=np.float32)
    adj_list = {} ## spare matrix store
    for item in graph:
        out_degree[item[0]] += 1
        if item[0] not in adj_list:
            adj_list[item[0]] = []
        adj_list[item[0]].append(item[1])
    
    
    ## compute the pagerank
    
    pagerank = np.ones(node_num, dtype=np.float32)/node_num
    teleport = (1 - beta) / node_num
    ## iteration
    errors = []
    for i in tqdm(range(iters)):
        pagerank_new = np.full(node_num, teleport)
        for node in range(node_num):
            if node not in adj_list: ## no out degree, dead end
                continue
            for next_node in adj_list[node]:
                pagerank_new[next_node] += beta*pagerank[node]/out_degree[node]

        ## deal with the dead end, assure the sum of pagerank is 1
        pagerank_new += (1-np.sum(pagerank_new))/node_num

        ## compute the error
        error = np.linalg.norm(pagerank_new - pagerank)
        errors.append(error)
        pagerank = pagerank_new
    
    return pagerank, errors



if __name__ == "__main__":
   
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="sparse1", help="standard, sparse1, sparse2")
    parser.add_argument("--data_file", type=str, default="web_links.csv", help="the graph file")
    parser.add_argument("--beta", type=float, default=0.85, help="the beta value for random teleports")
    parser.add_argument("--iters", type=int, default=30, help="the number of iterations")
    args = parser.parse_args()

    ## record memory
    initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    graph, node_ids_map = read_data(args.data_file) ## node_ids_map is the original node id, can used to map the node id in graph to the original node id

    t1 = time.time()
    if args.method == "standard":
        pagerank, errors = standard_pagerank(graph, args.beta, args.iters)
    elif args.method == "sparse1":
        pagerank, errors = sparse_pagerank1(graph, args.beta, args.iters)
    elif args.method == "sparse2":
        pagerank, errors = sparse_pagerank2(graph, args.beta, args.iters)
    else:
        print("Invalid method")
        exit(1)
    t2 = time.time()
    final_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("errors: ", errors)
    print(f"Time cost for {args.method}: {t2-t1} seconds")
    print(f"Memory cost for {args.method}: {(final_memory-initial_memory)/1024} MB")
    save_top_1000_pagerank(pagerank, node_ids_map , f"{args.method}_pagerank_top1000.csv")
    draw_errors(errors, f"{args.method}_pagerank_errors.png")




