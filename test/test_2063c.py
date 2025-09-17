from collections import defaultdict

def dfs(node, parent, graph, sizes):
    size = 1
    for neighbor in graph[node]:
        if neighbor != parent:
            size += dfs(neighbor, node, graph, sizes)
    sizes[node] = size
    return size

def max_components(n, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    sizes = [0] * (n + 1)
    dfs(1, -1, graph, sizes)
    
    max_comp = 0
    for i in range(1, n + 1):
        if sizes[i] == 1:
            max_comp += 1
        else:
            break
    
    for i in range(n, 0, -1):
        if sizes[i] == 1:
            max_comp += 1
            break
    
    return max_comp

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    
    results = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        edges = []
        for _ in range(n - 1):
            u, v = map(int, data[index:index + 2])
            index += 2
            edges.append((u, v))
        
        result = max_components(n, edges)
        results.append(result)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
