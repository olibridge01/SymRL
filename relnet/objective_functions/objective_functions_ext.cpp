#include <iostream>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>
#include <numeric>
#include <functional>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/graph_utility.hpp>

#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>


#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

using namespace boost;
namespace py = boost::python;
namespace np = boost::python::numpy;

typedef adjacency_list<vecS, vecS, undirectedS> Graph;
typedef property_map<Graph, vertex_index_t>::type IndexMap;
typedef graph_traits<Graph>::vertex_iterator VertexIter;
typedef std::vector<int> IntVec;
typedef std::map<IntVec, int> VecIntMap;



typedef double t_weight;
typedef boost::property<boost::edge_weight_t, t_weight> EdgeWeightProperty;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                              boost::no_property, EdgeWeightProperty> WeightedGraph;
typedef boost::property_map<WeightedGraph, boost::edge_weight_t>::type WeightMap;
typedef boost::exterior_vertex_property<WeightedGraph, t_weight> DistanceProperty;
typedef DistanceProperty::matrix_type DistanceMatrix;
typedef DistanceProperty::matrix_map_type DistanceMatrixMap;
typedef std::vector<double> DoubleVec;


enum class RemovalStrategy { random, targeted};
enum class RobustnessMeasure { critical_fraction, largest_component_size };

void print_vector(IntVec& vec_to_print) {
    for (IntVec::iterator it=vec_to_print.begin(); it!=vec_to_print.end(); ++it)
        std::cout << ' ' << *it;
    std::cout << std::endl;
}

void print_double_vector(DoubleVec& vec_to_print) {
    for (DoubleVec::iterator it=vec_to_print.begin(); it!=vec_to_print.end(); ++it)
        std::cout << ' ' << *it;
    std::cout << std::endl;
}


void populate_index_labels(Graph& g, IntVec& index_labels) {
    for (auto v : boost::make_iterator_range(vertices(g))) {
       index_labels[v] = v;
    }
}

void generate_random_sequence(Graph& g, IntVec& removal_sequence, int N) {
    populate_index_labels(g, removal_sequence);
    std::random_shuffle (removal_sequence.begin(), removal_sequence.end());
}

void generate_targeted_sequence(Graph& g, const IntVec& node_degrees, IntVec& removal_sequence, int N) {
    populate_index_labels(g, removal_sequence);

    IntVec random_sequence(N);
    generate_random_sequence(g, random_sequence, N);

    std::sort(
        removal_sequence.begin(),
        removal_sequence.end(),
        [node_degrees, random_sequence](int a, int b) {
            return (node_degrees[a] > node_degrees[b]) || (node_degrees[a] == node_degrees[b] && random_sequence[a] > random_sequence[b]);
        }
    );
}

void generate_removal_sequence(RemovalStrategy strat, Graph& g, const IntVec& node_degrees, IntVec& removal_sequence, int N) {
    switch(strat) {
        case RemovalStrategy::random: return generate_random_sequence(g,removal_sequence, N);
        case RemovalStrategy::targeted: return generate_targeted_sequence(g, node_degrees, removal_sequence, N);
    }
}

bool should_cache_intermediate(RemovalStrategy strat) {
    switch(strat) {
        case RemovalStrategy::random: return false;
        case RemovalStrategy::targeted: return true;
    }
}

double evaluate_measure(RobustnessMeasure measure, Graph& g, IntVec& component, int nodes_removed) {
    switch(measure) {
        case RobustnessMeasure::critical_fraction: {
            double num = (double) (connected_components(g, &component[0]) - nodes_removed);
            return num;
        }
        case RobustnessMeasure::largest_component_size: {
            int ncc = connected_components(g, &component[0]);
            IntVec component_sizes(ncc);
            for (int i = 0; i != component.size(); ++i) {
                component_sizes[component[i]] += 1;
            }
            int max_size = *std::max_element(component_sizes.begin(), component_sizes.end());
            return max_size;
        }
    }
}

bool should_stop_evaluation(RobustnessMeasure measure, int nodes_removed, int N, double measure_result) {
    switch(measure) {
        case RobustnessMeasure::critical_fraction: {
            return measure_result > 1 || nodes_removed == N-1;
        }
        case RobustnessMeasure::largest_component_size: {
            double lcs_threshold = std::max(0.01, 1 / (double) N);
            double lcs_fraction = (measure_result / (double) N);
            return lcs_fraction <= lcs_threshold;
        }
    }
}

double aggregate_results(RobustnessMeasure measure, int nodes_removed, int N, IntVec& successive_results) {
    switch(measure) {
        case RobustnessMeasure::critical_fraction: {
            int cnum = nodes_removed;
            if (cnum == N-1) {
                cnum++;
            }
            return (double) cnum / (double) N;
        }
        case RobustnessMeasure::largest_component_size: {
            int num_rem = N - nodes_removed;
            IntVec remaining_values(num_rem, 1);
            successive_results.reserve(successive_results.size() + remaining_values.size());
            successive_results.insert(successive_results.end(), remaining_values.begin(), remaining_values.end());

            auto compute_r = [N](double acc, int lcs) {
                double term = ((double) lcs / (double) N);
                return acc + term;
            };
            double r = std::accumulate(successive_results.begin(),
                                       successive_results.end(),
                                       0.0, compute_r);
            return r / (double) N;
        }
    }
}

double simulate_robustness(RemovalStrategy strat,
                           RobustnessMeasure measure,
                           Graph& g, const IntVec& node_degrees, VecIntMap& intermediate_cache, int N, unsigned base_graph_hash, int sim_num, int random_seed) {
    IntVec successive_results;
    double final_result;

    std::srand (unsigned ( (base_graph_hash * sim_num * random_seed) ) );
    IntVec component(N);
    IntVec removal_sequence(N);
    generate_removal_sequence(strat, g, node_degrees, removal_sequence, N);

    for(int i=0; i<N-1; ++i) {
        int node_to_remove = removal_sequence[i];

        clear_vertex(node_to_remove, g);
        int nodes_removed = i+1;

        double measure_result;

        if(should_cache_intermediate(strat)) {
            IntVec seq_so_far = IntVec(removal_sequence.begin(), removal_sequence.begin() + i + 1);

            auto search = intermediate_cache.find(seq_so_far);
            if (search != intermediate_cache.end()) {
                measure_result = search->second;
            } else {
                measure_result = evaluate_measure(measure, g, component, nodes_removed);
                intermediate_cache.insert(std::make_pair(seq_so_far, measure_result));
            }
        }
        else {
            measure_result = evaluate_measure(measure, g, component, nodes_removed);
        }

        successive_results.push_back(measure_result);
        if (should_stop_evaluation(measure, nodes_removed, N, measure_result)) {
            final_result = aggregate_results(measure, nodes_removed, N, successive_results);
            break;
        }
    }

    return final_result;
}

double compute_robustness(RemovalStrategy strat,
                          RobustnessMeasure measure,
                          int N, int M, np::ndarray const& edge_list, int num_sims, unsigned base_graph_hash, int random_seed) {
    int* input_ptr = reinterpret_cast<int*>(edge_list.get_data());

    Graph g;
    int first_node, second_node;
    int input_size = M * 2;

    for (int i = 0; i < input_size-1; i+=2) {
        first_node = *(input_ptr + i);
        second_node = *(input_ptr + i + 1);
        add_edge(first_node, second_node, g);
    }

    IntVec node_degrees(N);
    for (auto v : boost::make_iterator_range(vertices(g))) {
        node_degrees[v] = out_degree(v, g);
    }

    double results_sum = 0;

    VecIntMap intermediate_cache;
    for (int sim_num = 1; sim_num < (num_sims + 1); ++sim_num) {
        Graph g_copy;
        copy_graph(g, g_copy);
        double sim_result = simulate_robustness(strat, measure, g_copy, node_degrees, intermediate_cache, N, base_graph_hash, sim_num, random_seed);
        results_sum += sim_result;
    }

    double exp_robustness = results_sum / ((double) num_sims);
    return exp_robustness;
}

double critical_fraction_random(int N, int M, np::ndarray const& edge_list, int num_sims, unsigned base_graph_hash, int random_seed)
{
    return compute_robustness(RemovalStrategy::random, RobustnessMeasure::critical_fraction, N, M, edge_list, num_sims, base_graph_hash, random_seed);
}

double critical_fraction_targeted(int N, int M, np::ndarray const& edge_list, int num_sims, unsigned base_graph_hash, int random_seed)
{
    return compute_robustness(RemovalStrategy::targeted, RobustnessMeasure::critical_fraction, N, M, edge_list, num_sims, base_graph_hash, random_seed);
}

double size_largest_component_random(int N, int M, np::ndarray const& edge_list, int num_sims, unsigned base_graph_hash, int random_seed)
{
    return compute_robustness(RemovalStrategy::random, RobustnessMeasure::largest_component_size, N, M, edge_list, num_sims, base_graph_hash, random_seed);
}

double size_largest_component_targeted(int N, int M, np::ndarray const& edge_list, int num_sims, unsigned base_graph_hash, int random_seed)
{
    return compute_robustness(RemovalStrategy::targeted, RobustnessMeasure::largest_component_size, N, M, edge_list, num_sims, base_graph_hash, random_seed);
}


double eff_from_distances(DoubleVec& dist_vec, double denominator) {
    double eff_value = 0.0d;
    double dist, inverse_dist, eff_term;

    for (DoubleVec::iterator it=dist_vec.begin(); it!=dist_vec.end(); ++it) {
        dist = *it;
        inverse_dist = 1 / dist;
        eff_term = inverse_dist / denominator;
        eff_value += eff_term;
    }

    return eff_value;
}

double global_efficiency(int N, int M, np::ndarray const& edge_list,
                                       np::ndarray const& edge_lengths,
                                       np::ndarray const& pairwise_dists) {
    int* input_ptr = reinterpret_cast<int*>(edge_list.get_data());
    double* edge_len_ptr = reinterpret_cast<double*>(edge_lengths.get_data());
    double* dists_ptr = reinterpret_cast<double*>(pairwise_dists.get_data());

    WeightedGraph g;
    int first_node, second_node;
    double edge_length;
    int input_size = M * 2;

    int max_M = (N * (N-1)) / 2;
    double denominator = double(max_M * 2);


    for (int i = 0; i < input_size-1; i+=2) {
        first_node = *(input_ptr + i);
        second_node = *(input_ptr + i + 1);
        edge_length = *(edge_len_ptr + (i / 2));

        // std::cout << "Edge length " << edge_length << std::endl;
        add_edge(first_node, second_node, edge_length, g);
    }

    double pairwise_dist;
    DoubleVec ideal_sp_vec(max_M);

    for (int i = 0; i < max_M ; ++i) {
        pairwise_dist = *(dists_ptr + i);
        // std::cout << "Pairwise dist " << pairwise_dist << std::endl;
        ideal_sp_vec[i] = pairwise_dist;
    }

    WeightMap weight_pmap = boost::get(boost::edge_weight, g);
    DistanceMatrix distances(num_vertices(g));
    DistanceMatrixMap dm(distances, g);

    bool valid = floyd_warshall_all_pairs_shortest_paths(g, dm, boost::weight_map(weight_pmap));

    // check if there no negative cycles
    if (!valid) {
        std::cerr << "Error - Negative cycle in matrix" << std::endl;
        return double(-1);
    }

    DoubleVec sp_vec;
    for (std::size_t i = 0; i < num_vertices(g); ++i) {
        for (std::size_t j = i; j < num_vertices(g); ++j) {
            if(i==j) {
                continue;
            }
            if(distances[i][j] == std::numeric_limits<t_weight>::max()) {
                // found a value that is "inf"; returning -1
                return double(-1);
            }
            else {
                pairwise_dist = distances[i][j];
                sp_vec.push_back(pairwise_dist);
            }
       }
    }

    double actual_eff = eff_from_distances(sp_vec, denominator);
    double ideal_eff = eff_from_distances(ideal_sp_vec, denominator);
//    std::cout << "Actual eff " << actual_eff << std::endl;
//    std::cout << "Ideal eff " << ideal_eff << std::endl;
    return actual_eff / ideal_eff;
}


BOOST_PYTHON_MODULE(objective_functions_ext)
{
    Py_Initialize();
    np::initialize();

    def("critical_fraction_random", critical_fraction_random);
    def("critical_fraction_targeted", critical_fraction_targeted);
    def("size_largest_component_random", size_largest_component_random);
    def("size_largest_component_targeted", size_largest_component_targeted);
    def("global_efficiency", global_efficiency);
}
