import warnings
from copy import deepcopy

import networkx as nx
import numpy as np
import xxhash
from scipy.spatial import KDTree
from scipy.spatial import distance
from scipy.spatial.distance import pdist

budget_eps = 1e-5

class RelnetGraph(object):
    """Base graph state."""
    def __init__(self, g):
        self.num_nodes = g.number_of_nodes()
        self.node_labels = np.arange(self.num_nodes)
        self.all_nodes_set = set(self.node_labels)

        # Check if no edges
        if g.number_of_edges() == 0:
            self.num_edges = 0
            self.edge_pairs = np.ndarray(shape=(0, 2), dtype=np.int32)
        else:
            x, y = zip(*(sorted(g.edges())))
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = np.ravel(self.edge_pairs)

        self.node_degrees = np.array([deg for (node, deg) in sorted(g.degree(), key=lambda deg_pair: deg_pair[0])])
        self.first_node = None
        self.dynamic_edges = None

    def add_edge(self, first_node, second_node):
        """Return new graph with new edge added."""
        nx_graph = self.to_networkx()
        nx_graph.add_edge(first_node, second_node)
        s2v_graph = RelnetGraph(nx_graph)
        return s2v_graph, 1

    def add_edge_dynamically(self, first_node, second_node):
        """Add edge to graph in-place."""
        self.dynamic_edges.append((first_node, second_node))
        self.node_degrees[first_node] += 1
        self.node_degrees[second_node] += 1
        return 1

    def populate_banned_actions(self, budget=None):
        """Populate the banned actions based on budget."""
        if budget is not None:
            if budget < budget_eps:
                self.banned_actions = self.all_nodes_set
                return

        if self.first_node is None:
            self.banned_actions = self.get_invalid_first_nodes(budget)
        else:
            self.banned_actions = self.get_invalid_edge_ends(self.first_node, budget)

    def get_invalid_first_nodes(self, budget=None):
        """Check which first nodes are invalid (already fully connected)."""
        return set([node_id for node_id in self.node_labels if self.node_degrees[node_id] == (self.num_nodes - 1)])

    def get_invalid_edge_ends(self, query_node, budget=None):
        """Get the nodes already connected to the query node."""
        results = set()
        results.add(query_node)

        existing_edges = self.edge_pairs.reshape(-1, 2)
        existing_left = existing_edges[existing_edges[:,0] == query_node]
        results.update(np.ravel(existing_left[:,1]))

        existing_right = existing_edges[existing_edges[:,1] == query_node]
        results.update(np.ravel(existing_right[:,0]))

        if self.dynamic_edges is not None:
            dynamic_left = [entry[0] for entry in self.dynamic_edges if entry[0] == query_node]
            results.update(dynamic_left)
            dynamic_right = [entry[1] for entry in self.dynamic_edges if entry[1] == query_node]
            results.update(dynamic_right)
        return results

    def init_dynamic_edges(self):
        """Initialize the dynamic edges."""
        self.dynamic_edges = []

    def apply_dynamic_edges(self):
        """Apply the dynamic edges to the graph."""
        nx_graph = self.to_networkx()
        for edge in self.dynamic_edges:
            nx_graph.add_edge(edge[0], edge[1])
        return RelnetGraph(nx_graph)

    def to_networkx(self):
        """Convert to networkx graph."""
        edges = self.convert_edges()
        g = nx.Graph()
        g.add_nodes_from(self.node_labels)
        g.add_edges_from(edges)
        return g

    def convert_edges(self):
        """Convert edge pairs to list of edges."""
        return np.reshape(self.edge_pairs, (self.num_edges, 2))

    def display(self, ax=None):
        """Display the graph."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw_shell(nx_graph, with_labels=True, ax=ax)

    def display_with_positions(self, node_positions, ax=None):
        """Display the graph with node positions."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw(nx_graph, pos=node_positions, with_labels=True, ax=ax)

    def draw_to_file(self, filename):
        """Draw the graph to a file."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_size_length = self.num_nodes
        figsize = (fig_size_length, fig_size_length)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        self.display_with_positions(self.node_positions, ax=ax)
        fig.savefig(filename)
        plt.close()

    def get_adjacency_matrix(self):
        """Get the adjacency matrix of the graph."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            adj_matrix = np.asarray(nx.convert_matrix.to_numpy_matrix(nx_graph, nodelist=self.node_labels))

        return adj_matrix

    def compute_restrictions(self, restriction_mechanism, conn_radius_modifier):
        # no-op for non-geometric graphs.
        pass

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        gh = get_graph_hash(self, size=32, include_first=True)
        return f"Graph State with hash {gh}"

class GeometricRelnetGraph(RelnetGraph):
    """Geometric graph state."""
    distance_eps = 1e-4

    def __init__(self, g, node_positions, edge_lengths=None,
                 allowed_connections=None, maximally_connected_nodes=None, shortest_allowed_connection=None,
                 tracked_edges=None):
        super().__init__(g)
        self.node_positions = node_positions

        if edge_lengths is None:
            self.edge_lengths = self.compute_edge_lengths()
        else:
            self.edge_lengths = edge_lengths

        self.allowed_connections, self.maximally_connected_nodes, self.shortest_allowed_connection = \
            allowed_connections, maximally_connected_nodes, shortest_allowed_connection

        self.tracked_edges = tracked_edges

    def start_edge_tracking(self):
        """Start tracking edges."""
        self.tracked_edges = list()

    def compute_restrictions(self, restriction_mechanism, conn_radius_modifier):
        """Compute the restrictions on node selection."""
        if restriction_mechanism == 'max_current':
            conn_radii = self.compute_conn_radii_max_current(self.edge_lengths, conn_radius_modifier)
        elif restriction_mechanism == 'range':
            conn_radii = self.compute_conn_radii_range(self.edge_lengths, conn_radius_modifier)
        elif restriction_mechanism == 'none':
            conn_radii = np.full(self.num_nodes, np.inf, dtype=np.float32)
        else:
            raise ValueError(f"restriction mechanism {restriction_mechanism} not known!")
        self.allowed_connections, self.maximally_connected_nodes, self.shortest_allowed_connection = \
            self.compute_connectivity_information(conn_radii)

    def add_edge(self, first_node, second_node):
        """Return new graph with new edge added."""
        nx_graph = self.to_networkx()
        nx_graph.add_edge(first_node, second_node)

        edge_length = self.compute_edge_length(first_node, second_node)
        new_edge_lengths = deepcopy(self.edge_lengths)
        new_edge_lengths[(first_node, second_node)] = edge_length

        new_allowed_connections = deepcopy(self.allowed_connections)
        new_maximally_connected_nodes = deepcopy(self.maximally_connected_nodes)
        new_shortest_allowed_connection = deepcopy(self.shortest_allowed_connection)

        self.update_connectivity_info(first_node, second_node,
                                      edge_length,
                                      new_allowed_connections,
                                      new_maximally_connected_nodes,
                                      new_shortest_allowed_connection)

        if self.tracked_edges is not None:
            tracked_edges = deepcopy(self.tracked_edges)
            tracked_edges.append((first_node, second_node))
        else:
            tracked_edges = None

        s2v_graph = GeometricRelnetGraph(nx_graph, self.node_positions, new_edge_lengths,
                                         new_allowed_connections, new_maximally_connected_nodes, new_shortest_allowed_connection,
                                         tracked_edges=tracked_edges)
        edge_cost = edge_length

        return s2v_graph, edge_cost

    def compute_edge_length(self, first_node, second_node):
        """Get the Euclidean distance between two nodes."""
        pos_u, pos_v = self.node_positions[first_node], self.node_positions[second_node]
        # print(f'pos_u: {pos_u}')
        # print(f'pos_v: {pos_v}')
        edge_length = distance.euclidean(pos_u, pos_v)
        return edge_length

    def update_connectivity_info(self, first_node, second_node,
                                 edge_length,
                                 allowed_connections,
                                 maximally_connected_nodes,
                                 shortest_allowed_connection):
        """Update the connectivity information of the graph."""

        allowed_connections[first_node].pop(second_node)
        if len(allowed_connections[first_node]) > 0:
            if shortest_allowed_connection[first_node] == edge_length:
                shortest_allowed_connection[first_node] = min(allowed_connections[first_node].values())
        else:
            maximally_connected_nodes.add(first_node)
            if first_node in shortest_allowed_connection:
                shortest_allowed_connection.pop(first_node)

        if first_node in allowed_connections[second_node]:
            allowed_connections[second_node].pop(first_node)

            if len(allowed_connections[second_node]) > 0:
                if shortest_allowed_connection[second_node] == edge_length:
                    shortest_allowed_connection[second_node] = min(allowed_connections[second_node].values())
            else:
                maximally_connected_nodes.add(second_node)
                if second_node in shortest_allowed_connection:
                    shortest_allowed_connection.pop(second_node)

    def add_edge_dynamically(self, first_node, second_node):
        """Add edge to graph in-place."""
        super().add_edge_dynamically(first_node, second_node)
        edge_length = self.compute_edge_length(first_node, second_node)


        edge_cost = edge_length

        self.update_connectivity_info(first_node, second_node,
                                      edge_length,
                                      self.allowed_connections,
                                      self.maximally_connected_nodes,
                                      self.shortest_allowed_connection)
        return edge_cost

    def get_invalid_first_nodes(self, budget=None):
        """Get the first nodes which are invalid (fully connected or closest node too far away)."""
        if budget is None:
            raise ValueError("budget information needed to populate invalid actions in geometric graphs!")

        max_nodes = self.maximally_connected_nodes
        closest_too_far = {entry[0] for entry in self.shortest_allowed_connection.items() if entry[1] > budget}
        return max_nodes.union(closest_too_far)


    def get_invalid_edge_ends(self, query_node, budget=None):
        """Get the second nodes not allowed given the budget and query node."""
        if budget is None:
            raise ValueError("budget information needed to populate invalid actions in geometric graphs!")

        allowed = {entry[0] for entry in self.allowed_connections[query_node].items() if entry[1] <= budget}
        invalid_ends = self.all_nodes_set - allowed
        return invalid_ends

    def apply_dynamic_edges(self):
        """Apply the dynamic edges to the graph."""
        nx_graph = self.to_networkx()
        for edge in self.dynamic_edges:
            nx_graph.add_edge(edge[0], edge[1])

        s2v_graph = GeometricRelnetGraph(nx_graph, self.node_positions)
        return s2v_graph

    def display(self, ax=None):
        """Display the geometric graph."""
        return self.display_with_positions(self.node_positions, ax=ax)

    def get_node_positions(self):
        """Get the node positions."""
        return self.node_positions

    def compute_edge_lengths(self):
        """Compute the edge lengths of the graph."""
        edge_lengths = dict()
        edges = self.convert_edges()
        for edge in edges:
            dist = self.compute_edge_length(edge[0], edge[1])
            edge_lengths[(edge[0], edge[1])] = dist
        return edge_lengths

    def get_edge_lengths_as_arr(self):
        """Get the edge lengths as an array."""
        num_edges = self.num_edges
        edge_lengths_arr = np.zeros(num_edges, dtype=np.float64)

        for i in range(0, num_edges*2, 2):
            edge_from = self.edge_pairs[i]
            edge_to = self.edge_pairs[i+1]


            edge_length = self.edge_lengths[(edge_from, edge_to)] \
                            if (edge_from, edge_to) in self.edge_lengths \
                            else self.edge_lengths[(edge_to, edge_from)]

            edge_lengths_arr[int(i/2)] = edge_length

        return edge_lengths_arr

    def get_all_pairwise_distances(self):
        """Get the pairwise distances between all nodes."""
        # scipy function that returns diagonal entries as 1D array
        d = pdist(self.node_positions, metric='euclidean')
        return d

    def compute_conn_radii_max_current(self, edge_lengths, conn_radius_modifier):
        """Compute the maximum connection radius for each node."""
        max_radii = np.zeros(self.num_nodes, dtype=np.float32)
        edges = self.convert_edges()
        for edge in edges:
            u, v = edge[0], edge[1]
            edge_length = edge_lengths[(u, v)]
            max_radius = conn_radius_modifier * edge_length

            max_radii[u] = max(max_radii[u], max_radius)
            max_radii[v] = max(max_radii[v], max_radius)

        return max_radii

    def compute_conn_radii_range(self, edge_lengths, conn_radius_modifier):
        """Compute the connection radius for each node based on average edge length."""
        avg_length = np.mean([v for k,v in edge_lengths.items()])
        allowed_radius = avg_length * conn_radius_modifier
        range_radii = np.full(self.num_nodes, allowed_radius, dtype=np.float32)
        return range_radii

    def compute_connectivity_information(self, conn_radii):
        """Compute the allowed connections for each node."""
        allowed_connections = dict()
        shortest_allowed_connection = dict()
        kdtree = KDTree(self.node_positions)
        for node in self.node_labels:
            max_radius = conn_radii[node]
            possible_edges = set(kdtree.query_ball_point(self.node_positions[node], max_radius + self.distance_eps, p=2))
            invalid_edges = super().get_invalid_edge_ends(node, -1)

            node_allowed_cons = possible_edges - invalid_edges
            allowed_cons = {n: self.compute_edge_length(node, n) for n in node_allowed_cons}

            allowed_connections[node] = allowed_cons
            if len(allowed_cons) > 0:
                shortest_allowed_connection[node] = min(allowed_cons.values())

        maximally_connected_nodes = set([node_id for node_id in self.node_labels if len(allowed_connections[node_id]) == 0])
        return allowed_connections, maximally_connected_nodes, shortest_allowed_connection

    @staticmethod
    def get_cost_spent(orig_graph, final_graph):
        """Get the cost spent in adding edges."""
        edge_length_orig = sum(orig_graph.edge_lengths.values())
        edge_length_final = sum(final_graph.edge_lengths.values())
        return edge_length_final - edge_length_orig
    

class GeometricCoveringGraph(GeometricRelnetGraph):
    """Geometric graph state for covering problems."""
    def __init__(self, g, node_positions, covering_nodes=None, edge_lengths=None, tracked_edges=None):
        super().__init__(g, node_positions, edge_lengths=edge_lengths, tracked_edges=tracked_edges)
        
        self.covering_nodes = set() if covering_nodes is None else covering_nodes


    def populate_banned_actions(self):
        """Get the banned actions for the current graph state."""
        self.banned_actions = self.get_invalid_nodes()

    def get_invalid_nodes(self):
        """Invalid nodes are those with degree 0."""
        return set([node_id for node_id in self.node_labels if self.node_degrees[node_id] == 0])
    
    def cover_node(self, node_id):
        """Cover a node in the graph."""

        # Add node to cover set
        new_covering_nodes = deepcopy(self.covering_nodes)
        new_covering_nodes.add(node_id)

        # Get networkx graph
        nx_graph = self.to_networkx()

        if self.tracked_edges is not None:
            tracked_edges = deepcopy(self.tracked_edges)
        else:
            tracked_edges = None

        # Remove all edges connected to the node
        edges_to_remove = list(nx_graph.edges(node_id))
        for edge in edges_to_remove:
            nx_graph.remove_edge(*edge)
            self.edge_lengths.pop(edge, None)
        
        self.edge_lengths = self.compute_edge_lengths()
        s2v_graph = GeometricCoveringGraph(nx_graph, self.node_positions, new_covering_nodes,
                                           self.edge_lengths, tracked_edges)
        
        return s2v_graph

    def cover_node_dynamically(self, node_id):
        """Cover a node in the graph in-place."""
        self.covering_nodes.add(node_id)
        nx_graph = self.to_networkx()
        for edge in nx_graph.edges(node_id):
            if self.node_degrees[edge[0]] > 0 and self.node_degrees[edge[1]] > 0:
                self.node_degrees[edge[0]] -= 1
                self.node_degrees[edge[1]] -= 1
                self.dynamic_edges.append(edge)
        
        return

    def apply_dynamic_edges(self):
        """Remove dynamic edges (covered edges) from the graph."""
        nx_graph = self.to_networkx()

        for edge in self.dynamic_edges:
            nx_graph.remove_edge(*edge)

        self.edge_lengths = self.compute_edge_lengths()

        s2v_graph = GeometricCoveringGraph(nx_graph, self.node_positions, self.covering_nodes, self.edge_lengths)
        return s2v_graph
        
    def draw_cover_set(self, filename, cover_set=None):
        """Draw the graph to a file."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_size_length = self.num_nodes
        figsize = (fig_size_length, fig_size_length)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        colors = ['b', 'r']
        color_map = [colors[0] if node_id in cover_set else colors[1] for node_id in self.node_labels]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw(nx_graph, pos=self.node_positions, with_labels=True, ax=ax, node_color=color_map)

        fig.savefig(filename)
        plt.close()


class GeometricRoutingGraph(GeometricRelnetGraph):
    """Geometric graph state for routing problems."""
    def __init__(self, g, node_positions, 
                 starting_node, current_node,
                 visited_nodes=None, edge_lengths=None, tracked_edges=None):
        super().__init__(g, node_positions, edge_lengths, tracked_edges)
        
        self.current_node = current_node
        self.starting_node = starting_node
        if visited_nodes is None:
            self.visited_nodes = set([starting_node])
            assert current_node == starting_node, "Current node must be the starting node if visited_nodes is None!"
        else:
            self.visited_nodes = visited_nodes

        assert self.starting_node in self.visited_nodes, "Starting node must have been visited!"
        assert self.current_node in self.visited_nodes, "Current node must have been visited!"

    def populate_banned_actions(self):
        """Get the banned actions for the current graph state (visited nodes)."""
        self.banned_actions = self.get_invalid_nodes()
    
    def get_invalid_nodes(self):
        """Get the invalid nodes for the current graph state."""
        return self.visited_nodes

    def add_edge(self, first_node, second_node):
        """Add an edge to the graph state."""
        nx_graph = self.to_networkx()
        nx_graph.add_edge(first_node, second_node)

        edge_length = self.compute_edge_length(first_node, second_node)
        new_edge_lengths = deepcopy(self.edge_lengths)
        new_edge_lengths[(first_node, second_node)] = edge_length

        self.visited_nodes.add(second_node)

        if self.tracked_edges is not None:
            tracked_edges = deepcopy(self.tracked_edges)
            tracked_edges.append((first_node, second_node))
        else:
            tracked_edges = None
        
        s2v_graph = GeometricRoutingGraph(nx_graph, self.node_positions,
                                          self.starting_node, second_node,
                                          self.visited_nodes, new_edge_lengths, tracked_edges)
        edge_cost = edge_length
        
        return s2v_graph, edge_cost

    def add_edge_dynamically(self, first_node, second_node):
        """Add edge to graph in-place."""
        self.dynamic_edges.append((first_node, second_node))
        self.node_degrees[first_node] += 1
        self.node_degrees[second_node] += 1

        edge_length = self.compute_edge_length(first_node, second_node)

        edge_cost = edge_length

        self.visited_nodes.add(second_node)
        return edge_cost

    def apply_dynamic_edges(self):
        """Apply the dynamic edges to the graph."""
        nx_graph = self.to_networkx()
        for edge in self.dynamic_edges:
            nx_graph.add_edge(edge[0], edge[1])

        self.edge_lengths = self.compute_edge_lengths()
        s2v_graph = GeometricRoutingGraph(nx_graph, self.node_positions, 
                                          self.starting_node, self.current_node, 
                                          self.visited_nodes, self.edge_lengths, self.tracked_edges)
        return s2v_graph
    
    def edge_exists(self, first_node, second_node):
        """Check if edge exists in graph."""
        if first_node == second_node:
            return True
        if self.dynamic_edges is None:
            return False
        return (first_node, second_node) in self.dynamic_edges or (second_node, first_node) in self.dynamic_edges

def get_graph_hash(g, size=32, include_first=False):
    """Get hash of graph state defined by its edge pairs and first node."""
    if size == 32:
        hash_instance = xxhash.xxh32()
    elif size == 64:
        hash_instance = xxhash.xxh64()
    else:
        raise ValueError("only 32 or 64-bit hashes supported.")

    if include_first:
        if g.first_node is not None:
            hash_instance.update(np.array([g.first_node]))
        else:
            hash_instance.update(np.zeros(g.num_nodes))

    hash_instance.update(g.edge_pairs)
    graph_hash = hash_instance.intdigest()
    return graph_hash

def get_node_hash(g, size=32):
    """Get hash of graph state defined by its visited nodes."""
    if size == 32:
        hash_instance = xxhash.xxh32()
    elif size == 64:
        hash_instance = xxhash.xxh64()
    else:
        raise ValueError("only 32 or 64-bit hashes supported.")

    # Define hash based on visited nodes and the current node
    hash_instance.update(np.array([g.current_node]))
    hash_instance.update(np.array(list(g.visited_nodes)))

    node_hash = hash_instance.intdigest()
    return node_hash

def get_cover_hash(g, size=32):
    """Get hash of graph state defined by its covering nodes."""
    if size == 32:
        hash_instance = xxhash.xxh32()
    elif size == 64:
        hash_instance = xxhash.xxh64()
    else:
        raise ValueError("only 32 or 64-bit hashes supported.")

    # Define hash based on visited nodes and the current node
    hash_instance.update(np.array(list(g.covering_nodes)))

    cover_hash = hash_instance.intdigest()
    return cover_hash