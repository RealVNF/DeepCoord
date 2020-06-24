# load existing network topology, generate new node capacities, and save to new file
import networkx as nx
import random


def set_new_caps(network_file, new_network_file, min_cap, max_cap):
    """Create copy of given network with new, random capacities within given bounds"""
    # read network and print current nodes
    network = nx.read_graphml(network_file)
    print(f"Current nodes of {network_file}:")
    for v in network.nodes(data=True):
        print(v)

    # set new node attributes, print again, and save to new file
    print(f"\nNew nodes; saved to {new_network_file}:")
    for v in network.nodes(data=True):
        v[1]['NodeCap'] = random.randint(min_cap, max_cap)
        print(v)
    nx.write_graphml(network, new_network_file)

    return new_network_file


def set_ingress(network_file, new_network_file, ingress_id):
    """Create copy of network with node ingress_id as ingress; keep existing ingress nodes"""
    network = nx.read_graphml(network_file)
    print(f"Current nodes of {network_file}:")
    for v in network.nodes(data=True):
        print(v)

    # set new node attributes, print again, and save to new file
    network.nodes[ingress_id]['NodeType'] = 'Ingress'
    print(f"\nNew nodes; saved to {new_network_file}:")
    for v in network.nodes(data=True):
        print(v)
    nx.write_graphml(network, new_network_file)

    return new_network_file


if __name__ == "__main__":
    # network to read and new file to write
    # random.seed(1234)
    network_file = '../res/networks/abilene/abilene-in1-cap1.graphml'
    new_network_file = set_new_caps(network_file, '../res/networks/abilene/abilene-in1-rand-cap0-2.graphml', min_cap=0, max_cap=2)
    new_network_file = set_ingress(new_network_file, '../res/networks/abilene/abilene-in2-rand-cap0-2.graphml', ingress_id='1')
    new_network_file = set_ingress(new_network_file, '../res/networks/abilene/abilene-in3-rand-cap0-2.graphml', ingress_id='2')
    new_network_file = set_ingress(new_network_file, '../res/networks/abilene/abilene-in4-rand-cap0-2.graphml', ingress_id='3')
    new_network_file = set_ingress(new_network_file, '../res/networks/abilene/abilene-in5-rand-cap0-2.graphml', ingress_id='4')
