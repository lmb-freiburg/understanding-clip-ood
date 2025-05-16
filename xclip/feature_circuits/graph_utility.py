import math
import os

import networkx as nx
from graphviz import Digraph

# Constants
EXAMPLE_FONT = 36  # Font size for example words at the bottom
CHAR_WIDTH = 0.1  # Width per character in labels
NODE_BUFFER = 0.1  # Extra width added to each node
INTERNODE_BUFFER = 0.1  # Extra space between nodes (not added to node width)
MIN_COL_WIDTH = 1  # Minimum width for a column
VERTICAL_SPACING = 0.8  # Vertical spacing between nodes in the same column
NODE_HEIGHT = 0.4  # Height of each node
CIRCUIT_ROW_LABEL_X = -10  # X position for row labels in plot_circuit


def create_dag(
    nodes,
    edges,
    features_by_submod,
    # layers=6,
    # example_text="The managers that the parent likes",
    # node_threshold=0.1,
    edge_k=2,
    pen_thickness=3,
    horizontal_spacing=0.2,
    annotations=None,
):
    def to_hex(number):
        scale = max(
            abs(min([v.to_tensor().min() for n, v in nodes.items() if n != 'y'])),
            abs(max([v.to_tensor().max() for n, v in nodes.items() if n != 'y'])),
        )
        number = number / scale

        if number < 0:
            red = 255
            green = blue = int((1 + number) * 255)
        elif number > 0:
            blue = 255
            red = green = int((1 - number) * 255)
        else:
            red = green = blue = 255

        text_hex = '#000000' if (red * 0.299 + green * 0.587 + blue * 0.114) > 170 else '#ffffff'
        hex_code = f'#{red:02X}{green:02X}{blue:02X}'

        return hex_code, text_hex

    def split_label(label):
        if len(label) > 20:  # Add a line break for labels longer than 20 characters
            if '/' in label:
                split_index = label.find('/', 10) + 1  # Find the first '/' after the 10th character
                if split_index > 0:
                    return label[:split_index], label[split_index:]
            words = label.split()
            mid = math.ceil(len(words) / 2)
            return ' '.join(words[:mid]), ' '.join(words[mid:])
        return label, ''

    if annotations is None:

        def get_label(name):
            return split_label(name.split(', ')[-1])  # Remove sequence information
    else:

        def get_label(name):
            seq, feat = name.split(', ')
            if feat in annotations:
                return split_label(annotations[feat])
            return split_label(feat)  # Remove sequence information

    G = Digraph(name='Feature circuit')
    G.graph_attr.update(rankdir='BT', newrank='true')
    G.node_attr.update(shape='box', style='rounded')

    G_nx = nx.DiGraph()

    # First pass: collect nodes and calculate widths
    nodes_by_submod = {}
    node_info = {}  # Store node info (label, color, etc.) for later use
    node_widths = {}  # Store width of each node
    for layer_name in nodes.keys():
        if layer_name == 'input':
            continue
        submod_nodes = nodes[layer_name].to_tensor()
        nodes_by_submod[f'{layer_name}'] = {
            layer_neuron_idx.item(): submod_nodes[layer_neuron_idx].item()
            for layer_neuron_idx in features_by_submod[layer_name]
        }

        # Store node info and calculate node width
        for neuron, val in nodes_by_submod[f'{layer_name}'].items():
            name = f'{layer_name}/{neuron}'
            fillhex, texthex = to_hex(val)
            label = name
            is_epsilon = neuron == submod_nodes.size(0) - 1
            node_shape = 'triangle' if is_epsilon else 'box'
            node_info[name] = {
                'label': label,
                'fillcolor': fillhex,
                'fontcolor': texthex,
                'shape': node_shape,
            }

            width = CHAR_WIDTH * len(label) + NODE_BUFFER
            node_widths[name] = width

    # Second pass: calculate positions and add nodes
    y_offset = 0
    for submod, submod_nodes in nodes_by_submod.items():
        # Add row label
        G.node(
            f'row_{submod}',
            label=submod,
            pos=f'{CIRCUIT_ROW_LABEL_X},{y_offset}!',  # Use the new constant
            shape='plaintext',
        )

        if len(submod_nodes) > 0:
            nodes_in_layer = nodes_by_submod[f'{submod}']
            total_width = sum(node_widths[f'{submod}/{n}'] for n in nodes_in_layer) + INTERNODE_BUFFER * (
                len(nodes_in_layer) - 1
            )
            total_width = max(total_width, MIN_COL_WIDTH)  # Ensure minimum width
            start_x = -total_width / 2  # Center the nodes

            for neuron_idx in nodes_in_layer:
                node_name = f'{submod}/{neuron_idx}'
                node_info_dict = node_info[node_name]
                x_pos = start_x + node_widths[node_name] / 2
                G.node(
                    node_name,
                    label=node_info_dict['label'],
                    fillcolor=node_info_dict['fillcolor'],
                    fontcolor=node_info_dict['fontcolor'],
                    style='filled',
                    shape=node_info_dict['shape'],
                    pos=f'{x_pos},{y_offset}!',
                    width=str(node_widths[node_name]),
                    height=str(NODE_HEIGHT),
                    fixedsize='true',
                )

                G_nx.add_node(node_name)

                start_x += node_widths[node_name] + INTERNODE_BUFFER

        y_offset += NODE_HEIGHT + VERTICAL_SPACING  # Proper vertical spacing between layers

    # Add edges
    for upstream_layer_name, v in edges.items():
        for downstream_layer_name, edge_weight_matrix in v.items():
            for downstream_weight_matrix_index, downstream_neuron_idx in enumerate(
                features_by_submod[downstream_layer_name]
            ):
                upstream_weight_matrix_indices = edge_weight_matrix[downstream_weight_matrix_index].topk(edge_k).indices
                for upstream_weight_matrix_index in upstream_weight_matrix_indices:
                    upstream_neuron_idx = features_by_submod[upstream_layer_name][upstream_weight_matrix_index]
                    uname = f'{upstream_layer_name}/{upstream_neuron_idx}'
                    dname = f'{downstream_layer_name}/{downstream_neuron_idx}'
                    weight = edge_weight_matrix[downstream_weight_matrix_index, upstream_weight_matrix_index].item()
                    G.edge(
                        uname,
                        dname,
                        # penwidth=str(abs(weight) * pen_thickness),
                        penwidth=str(pen_thickness),
                        color='red' if weight > 0 else 'blue',
                    )

                    G_nx.add_edge(uname, dname, weight=weight)

    return G, G_nx


def plot_graphviz_dag(G, save_dir):
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    G.render(save_dir, format='png', cleanup=True, engine='neato')
