import copy
import itertools
import os
import warnings

import graphviz
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from xclip.feature_circuits.graph_utility import create_dag, plot_graphviz_dag

warnings.simplefilter(action='ignore', category=FutureWarning)


domains = ['real', 'clipart', 'infograph', 'painting', 'quickdraw', 'sketch']
ood_classes = {
    'aircraft carrier': 0,
    'axe': 11,
    'banana': 13,
    'barn': 15,
    'bed': 25,
    'candle': 58,
    'lion': 174,
    'mountain': 190,
    'necklace': 197,
    'penguin': 218,
    'pizza': 225,
    'saxophone': 250,
    'television': 305,
    'tractor': 319,
    'traffic light': 320,
}


def plot_graph_similarity(domain_a_vs_domain_b, domains, circuit_analysis_dir, split, score_type='jaccard'):
    # Create an empty similarity matrix with given order
    similarity_matrix = pd.DataFrame(0, index=domains, columns=domains, dtype=float)

    # Fill the similarity matrix
    for (domain_a, domain_b), value in domain_a_vs_domain_b.items():
        similarity_matrix.loc[domain_a, domain_b] = value
        similarity_matrix.loc[domain_b, domain_a] = value  # Ensure symmetry

    # mask = np.triu(np.ones_like(similarity_matrix, dtype=bool)) # Mask upper triangular part
    # sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", mask=mask)
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.savefig(os.path.join(circuit_analysis_dir, f'{score_type}_{split}.pdf'))
    plt.close()

    print(f'Average similarity scores for {split} set:')
    not_quickdraw = []
    for domain_a in domains:
        scores = []
        for domain_b in domains:
            if domain_a == domain_b:
                continue
            scores.append(similarity_matrix.loc[domain_a, domain_b])
            if domain_b != 'quickdraw':
                not_quickdraw.append(similarity_matrix.loc[domain_a, domain_b])
        print(f'{domain_a}: {round(np.mean(scores), 3)}')
    print(f'Average similarity scores for {split} set (excluding quickdraw): {round(np.mean(not_quickdraw), 3)}')
    print('\n')


def jaccard_graph_similarity(G1, G2):
    """Computes the Jaccard similarity between two graphs based on edge sets."""
    E1 = set(G1.edges())
    E2 = set(G2.edges())

    intersection = len(E1 & E2)
    union = len(E1 | E2)

    return intersection / union if union > 0 else 0


# implementation from https://github.com/emanuele/jstsp2015/blob/master/gk_weisfeiler_lehman.py
class GK_WL:
    """
    Weisfeiler_Lehman graph kernel.
    """

    @staticmethod
    def old_adjacency_list_format(G):
        # For each node in the order of G.nodes(), return its list of neighbors
        return [list(G.neighbors(node)) for node in G.nodes()]

    def compare_list(self, graph_list, h=1, node_label=True):
        """Compute the all-pairs kernel values for a list of graphs.

        This function can be used to directly compute the kernel
        matrix for a list of graphs. The direct computation of the
        kernel matrix is faster than the computation of all individual
        pairwise kernel values.

        Parameters
        ----------
        graph_list: list
            A list of graphs (list of networkx graphs)
        h : interger
            Number of iterations.
        node_label : boolean
            Whether to use original node labels. True for using node labels
            saved in the attribute 'node_label'. False for using the node
            degree of each node as node attribute.

        Return
        ------
        K: numpy.array, shape = (len(graph_list), len(graph_list))
        The similarity matrix of all graphs in graph_list.

        """
        self.graphs = graph_list
        n = len(graph_list)
        lists = [0] * n
        k = [0] * (h + 1)
        n_nodes = 0
        n_max = set()

        # Compute adjacency lists and n_nodes, the total number of
        # nodes in the dataset.
        for i in range(n):
            # lists[i] = graph_list[i].adjacency_list()
            lists[i] = self.old_adjacency_list_format(graph_list[i])
            n_nodes = n_nodes + graph_list[i].number_of_nodes()

            # Computing the maximum number of nodes in the graphs. It
            # will be used in the computation of vectorial
            # representation.
            # if(n_max < graph_list[i].number_of_nodes()):
            n_max |= set(graph_list[i].nodes)

        n_max = len(n_max)
        phi = np.zeros((n_max, n), dtype=np.uint64)

        # INITIALIZATION: initialize the nodes labels for each graph
        # with their labels or with degrees (for unlabeled graphs)

        labels = [0] * n
        label_lookup = {}
        label_counter = 0

        # label_lookup is an associative array, which will contain the
        # mapping from multiset labels (strings) to short labels
        # (integers)

        if node_label is True:
            name_to_idx_lookup = {}
            for i in range(n):
                # It is assumed that the graph has an attribute
                # 'node_label'
                # l_aux = nx.get_node_attributes(graph_list[i],
                #                                'node_label').values()
                l_aux = list(graph_list[i].nodes)

                labels[i] = np.zeros(len(l_aux), dtype=np.int32)
                name_to_idx_lookup[i] = {name: idx for idx, name in enumerate(l_aux)}

                for j in range(len(l_aux)):
                    if l_aux[j] not in label_lookup:
                        label_lookup[l_aux[j]] = label_counter
                        labels[i][j] = label_counter
                        label_counter += 1
                    else:
                        labels[i][j] = label_lookup[l_aux[j]]
                    # labels are associated to a natural number
                    # starting with 0.
                    phi[labels[i][j], i] += 1
        else:
            for i in range(n):
                labels[i] = np.array(graph_list[i].degree().values())
                for j in range(len(labels[i])):
                    phi[labels[i][j], i] += 1

        # Simplified vectorial representation of graphs (just taking
        # the vectors before the kernel iterations), i.e., it is just
        # the original nodes degree.
        self.vectors = np.copy(phi.transpose())

        k = np.dot(phi.transpose(), phi)

        # MAIN LOOP
        it = 0
        new_labels = copy.deepcopy(labels)

        for it in range(h):
            # create an empty lookup table
            label_lookup = {}
            label_counter = 0

            phi = np.zeros((n_nodes, n), dtype=np.uint64)
            for i in range(n):
                for v in range(len(lists[i])):
                    # form a multiset label of the node v of the i'th graph
                    # and convert it to a string
                    # lists[i][v]
                    j_indices = [name_to_idx_lookup[i][name] for name in lists[i][v]]

                    long_label = np.concatenate((np.array([labels[i][v]]), np.sort(labels[i][j_indices])))
                    long_label_string = str(long_label)
                    # if the multiset label has not yet occurred, add it to the
                    # lookup table and assign a number to it
                    if long_label_string not in label_lookup:
                        label_lookup[long_label_string] = label_counter
                        new_labels[i][v] = label_counter
                        label_counter += 1
                    else:
                        new_labels[i][v] = label_lookup[long_label_string]
                # fill the column for i'th graph in phi
                aux = np.bincount(new_labels[i]).astype(np.uint64)
                phi[new_labels[i], i] += aux[new_labels[i]]

            k += np.dot(phi.transpose(), phi)
            labels = copy.deepcopy(new_labels)

        # Compute the normalized version of the kernel
        k_norm = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])

        return k_norm

    def compare(self, g_1, g_2, h=1, node_label=True):
        """Compute the kernel value (similarity) between two graphs.
        The kernel is normalized to [0,1] by the equation:
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2))

        Parameters
        ----------
        g_1 : networkx.Graph
            First graph.
        g_2 : networkx.Graph
            Second graph.
        h : interger
            Number of iterations.
        node_label : boolean
            Whether to use the values under the graph attribute 'node_label'
            as node labels. If False, the degree of the nodes are used as
            labels.

        Returns
        -------
        k : The similarity value between g1 and g2.
        """
        gl = [g_1, g_2]
        return self.compare_list(gl, h, node_label)[0, 1]


def main(args):
    circuit_analysis_dir = os.path.join(args.model_dir, 'circuit_analysis')
    assert os.path.isdir(circuit_analysis_dir), f'circuit analysis directory not found at {circuit_analysis_dir}'
    assert all([os.path.isdir(os.path.join(circuit_analysis_dir, domain)) for domain in domains]), (
        'circuit analysis directories not found for all domains'
    )

    if (
        args.regnerate_scores
        or args.regenerate_graphs
        or not os.path.exists(
            os.path.join(circuit_analysis_dir, f'dag_{args.edge_k}_{args.score_type}_all_domain_a_vs_domain_b.pt')
        )
    ):
        pt_files = sorted(
            [f for f in os.listdir(os.path.join(circuit_analysis_dir, 'real')) if f.endswith('_edges.pt')]
        )
        # print(pt_files)

        all_domain_a_vs_domain_b, ood_domain_a_vs_domain_b, id_domain_a_vs_domain_b = {}, {}, {}
        for pt_file in tqdm(pt_files, leave=False, desc='Iterate over classes...'):  # iterate class ids
            # assert all([os.path.isfile(os.path.join(circuit_analysis_dir, domain, pt_file)) for domain in domains]), f'file not found at {os.path.join(circuit_analysis_dir, domain, pt_file)}'
            label = int(pt_file.split('_')[0])
            if not all([os.path.isfile(os.path.join(circuit_analysis_dir, domain, pt_file)) for domain in domains]):
                for domain in domains:
                    if not os.path.isfile(os.path.join(circuit_analysis_dir, domain, pt_file)):
                        print(f'file not found at {os.path.join(circuit_analysis_dir, domain, pt_file)}')
                continue
            all_graphs = {}
            for domain in tqdm(domains, leave=False, desc='Create graphs...'):
                nodes = torch.load(
                    os.path.join(circuit_analysis_dir, domain, pt_file.replace('edges', 'nodes')), map_location='cpu'
                )
                edges = torch.load(os.path.join(circuit_analysis_dir, domain, pt_file), map_location='cpu')
                features_by_submod = torch.load(
                    os.path.join(circuit_analysis_dir, domain, pt_file.replace('edges', 'features_by_submod')),
                    map_location='cpu',
                )

                if args.regenerate_graphs or not os.path.exists(
                    os.path.join(circuit_analysis_dir, domain, pt_file.replace('_edges.pt', f'_dag_{args.edge_k}.dot'))
                ):
                    G_graphviz, G_nx = create_dag(nodes, edges, features_by_submod, edge_k=args.edge_k)
                    nx.drawing.nx_pydot.write_dot(
                        G_nx,
                        os.path.join(
                            circuit_analysis_dir, domain, pt_file.replace('_edges.pt', f'_dag_{args.edge_k}.dot')
                        ),
                    )
                else:
                    G_nx = nx.drawing.nx_pydot.read_dot(
                        os.path.join(
                            circuit_analysis_dir, domain, pt_file.replace('_edges.pt', f'_dag_{args.edge_k}.dot')
                        )
                    )
                    if args.plot:
                        G_graphviz = graphviz.Source.from_file(
                            os.path.join(
                                circuit_analysis_dir, domain, pt_file.replace('_edges.pt', f'_dag_{args.edge_k}.dot')
                            )
                        )

                all_graphs[domain] = G_nx
                if args.plot:
                    plot_graphviz_dag(
                        G_graphviz, os.path.join(circuit_analysis_dir, domain, pt_file.replace('_edges.pt', ''))
                    )

            if 'wl' in args.score_type:
                if '_h1' in args.score_type:
                    K = GK_WL().compare_list([all_graphs[domain] for domain in domains], h=1, node_label=True)
                elif '_h2' in args.score_type:
                    K = GK_WL().compare_list([all_graphs[domain] for domain in domains], h=3, node_label=True)
                elif '_h3' in args.score_type:
                    K = GK_WL().compare_list([all_graphs[domain] for domain in domains], h=3, node_label=True)
                else:
                    raise NotImplementedError(f'Score type {args.score_type} not implemented.')

                for domain_a, domain_b in itertools.combinations(domains, 2):
                    domain_a_idx = domains.index(domain_a)
                    domain_b_idx = domains.index(domain_b)
                    score = K[domain_a_idx, domain_b_idx]

                    if (domain_a, domain_b) not in all_domain_a_vs_domain_b:
                        all_domain_a_vs_domain_b[(domain_a, domain_b)] = []
                    all_domain_a_vs_domain_b[(domain_a, domain_b)].append(score)

                    if label in ood_classes.values():
                        if (domain_a, domain_b) not in ood_domain_a_vs_domain_b:
                            ood_domain_a_vs_domain_b[(domain_a, domain_b)] = []
                        ood_domain_a_vs_domain_b[(domain_a, domain_b)].append(score)
                    else:  # id
                        if (domain_a, domain_b) not in id_domain_a_vs_domain_b:
                            id_domain_a_vs_domain_b[(domain_a, domain_b)] = []
                        id_domain_a_vs_domain_b[(domain_a, domain_b)].append(score)
            else:
                for domain_a, domain_b in tqdm(
                    itertools.combinations(domains, 2), leave=False, desc='Compute pairwise graph similarity...'
                ):
                    # for v in nx.similarity.optimize_graph_edit_distance(all_graphs[domain_a], all_graphs[domain_b]):
                    #     print(v)
                    # print(f'{domain_a} vs {domain_b}: {graph_edit_dist}')
                    if args.score_type == 'jaccard':
                        score = jaccard_graph_similarity(all_graphs[domain_a], all_graphs[domain_b])
                    elif 'wl' in args.score_type:
                        score = GK_WL().compare(all_graphs[domain_a], all_graphs[domain_b], h=3, node_label=True)
                    else:
                        raise NotImplementedError(f'Score type {args.score_type} not implemented.')
                    # spectral_similarity_score = spectral_similarity(all_graphs[domain_a], all_graphs[domain_b])

                    if (domain_a, domain_b) not in all_domain_a_vs_domain_b:
                        all_domain_a_vs_domain_b[(domain_a, domain_b)] = []
                    all_domain_a_vs_domain_b[(domain_a, domain_b)].append(score)

                    if label in ood_classes.values():
                        if (domain_a, domain_b) not in ood_domain_a_vs_domain_b:
                            ood_domain_a_vs_domain_b[(domain_a, domain_b)] = []
                        ood_domain_a_vs_domain_b[(domain_a, domain_b)].append(score)
                    else:  # id
                        if (domain_a, domain_b) not in id_domain_a_vs_domain_b:
                            id_domain_a_vs_domain_b[(domain_a, domain_b)] = []
                        id_domain_a_vs_domain_b[(domain_a, domain_b)].append(score)

        all_domain_a_vs_domain_b = {k: np.mean(v) for k, v in all_domain_a_vs_domain_b.items()}
        torch.save(
            all_domain_a_vs_domain_b,
            os.path.join(circuit_analysis_dir, f'dag_{args.edge_k}_{args.score_type}_all_domain_a_vs_domain_b.pt'),
        )

        ood_domain_a_vs_domain_b = {k: np.mean(v) for k, v in ood_domain_a_vs_domain_b.items()}
        torch.save(
            ood_domain_a_vs_domain_b,
            os.path.join(circuit_analysis_dir, f'dag_{args.edge_k}_{args.score_type}_ood_domain_a_vs_domain_b.pt'),
        )

        id_domain_a_vs_domain_b = {k: np.mean(v) for k, v in id_domain_a_vs_domain_b.items()}
        torch.save(
            id_domain_a_vs_domain_b,
            os.path.join(circuit_analysis_dir, f'dag_{args.edge_k}_{args.score_type}_id_domain_a_vs_domain_b.pt'),
        )
    else:
        all_domain_a_vs_domain_b = torch.load(
            os.path.join(circuit_analysis_dir, f'dag_{args.edge_k}_{args.score_type}_all_domain_a_vs_domain_b.pt'),
            map_location='cpu',
        )
        ood_domain_a_vs_domain_b = torch.load(
            os.path.join(circuit_analysis_dir, f'dag_{args.edge_k}_{args.score_type}_ood_domain_a_vs_domain_b.pt'),
            map_location='cpu',
        )
        id_domain_a_vs_domain_b = torch.load(
            os.path.join(circuit_analysis_dir, f'dag_{args.edge_k}_{args.score_type}_id_domain_a_vs_domain_b.pt'),
            map_location='cpu',
        )

    plot_graph_similarity(all_domain_a_vs_domain_b, domains, circuit_analysis_dir, 'all', score_type=args.score_type)
    plot_graph_similarity(ood_domain_a_vs_domain_b, domains, circuit_analysis_dir, 'ood', score_type=args.score_type)
    plot_graph_similarity(id_domain_a_vs_domain_b, domains, circuit_analysis_dir, 'id', score_type=args.score_type)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Configure CLIP models for neuron analysis.')
    parser.add_argument('--model_dir', type=str, required=True, help='path to model directory')
    parser.add_argument('--score_type', type=str, default='jaccard', help='type of similarity score to compute')
    parser.add_argument('--edge_k', type=int, default=3, help='number of edges to keep')
    parser.add_argument('--plot', action='store_true', help='plot the graph')
    parser.add_argument('--regenerate_graphs', action='store_true', help='regenerate graphs')
    parser.add_argument('--regnerate_scores', action='store_true', help='regenerate scores')
    args = parser.parse_args()
    main(args)
