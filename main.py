#!/usr/bin/python
# coding=utf-8
"""Main script, we will see what this will do."""
import argparse
import components
import json
import logging
import networkx as nx
import matplotlib.pylab as plt
import random


def parse_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json',
                        dest='json',
                        type=str,
                        help='should contain network configuration',
                        default='test.json')
    parser.add_argument('-c', '--coding',
                        dest='coding',
                        type=int,
                        help='Batch size for coded packets. Default None(without coding)',
                        default=8)
    parser.add_argument('-f', '--fieldsize',
                        dest='fieldsize',
                        type=int,
                        help='Fieldsize used for coding.',
                        default=2)
    return parser.parse_args()


def readconf(jsonfile):
    """Read configuration file."""
    with open(jsonfile) as f:
        config = json.loads(f.read())
    return config


class Simulator:
    """Round based simulator to simulate traffic in meshed network."""

    def __init__(self, loglevel=logging.DEBUG, jsonfile=None, coding=None, fieldsize=2):
        self.config = {}
        self.graph = None
        self.nodes = []
        self.packets = []
        self.newpackets = []
        self.batch = 0
        self.coding = coding
        self.fieldsize = fieldsize
        self.pos = None
        import logging
        logging.basicConfig(filename='simulator.log', level=loglevel, format='%(asctime)s %(message)s', filemode='w')
        self.getready(jsonfile)
        self.done = False
        self.paths = {}

    def broadcast(self, packet):
        """Broadcast given packet to all neighbors."""
        nodename = packet.getpos()
        for neighbor in list(self.graph.neighbors(nodename)):
            if self.graph.edges[nodename, neighbor]['weight'] > random.random() and neighbor != 'S':    # roll dice
                info = packet.getall()  # Duplicate packet in receiver node
                nodes = info[6]
                if neighbor in nodes.keys():
                    if isinstance(nodes[neighbor], list):
                        nodes[neighbor].append(info[5] + 1)
                    else:
                        nodes[neighbor] = [nodes[neighbor], info[5] + 1]
                else:
                    nodes[neighbor] = info[5] + 1
                if self.coding is None:             # Add received Packet to buffer without coding
                    self.newpackets.append(components.Packet(src=info[0], dst=info[1], batch=info[2], pos=neighbor,
                                                             coding=info[4], latency=(info[5] + 1), nodes=nodes))
                else:
                    for name in self.nodes:         # Add received Packet to buffer with coding
                        if str(name) == neighbor:
                            name.buffpacket(batch=info[2], coding=info[4], preveotx=self.graph.nodes[nodename]['EOTX'])
                            newpacket = name.getcoded()
                            if newpacket is not None:
                                self.newpackets.append(
                                    components.Packet(src=info[0], dst=info[1], batch=info[2], pos=neighbor,
                                                      coding=newpacket, latency=(info[5] + 1), nodes=nodes))
                            if str(name) == 'D':    # Save path the packet has traveled
                                if info[2] in self.paths:
                                    self.paths[info[2]].append(nodes)
                                else:
                                    self.paths[info[2]] = [nodes]
                            break
        packet.update()

    def createnetwork(self, config):
        """Create network using networkx library based on configuration given as dict."""
        configlist = []
        for (src, con) in config.items():
            for (dst, wgt) in con.items():
                configlist.append((src, dst, wgt))
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from(configlist)
        self.nodes = [components.Node(name=name, coding=self.coding, fieldsize=self.fieldsize)
                      for name in self.graph.nodes]

    def drawused(self):
        """Highlight paths used in graph drawn in getready()."""
        edgelists = []
        for batch in self.paths:
            for path in self.paths[batch]:
                edges = {}
                nodes = list(path.keys())
                for j in range(len(nodes) - 1):
                    if (nodes[j], nodes[j + 1]) in edges.keys():
                        edges[(nodes[j], nodes[j + 1])] += 1
                    else:
                        edges[(nodes[j], nodes[j + 1])] = 1
                edgelists.append(edges)
        logging.info('Paths used {}'.format(edgelists))
        for sublist in edgelists:
            alpha = list(sublist.values())[0]/10
            nx.draw_networkx_edges(self.graph, pos=self.pos, edgelist=list(sublist.keys()), width=8, alpha=alpha,
                                   edge_color='r')
        plt.savefig('usedgraph.png')

    def getgraph(self):
        """Return graph."""
        return self.graph

    def getready(self, jsonfile):
        """Do the basic stuff to get ready."""
        self.createnetwork(readconf(jsonfile))
        self.calceotx()
        self.pos = nx.spring_layout(self.graph).copy()
        nx.draw(self.graph, pos=self.pos, with_labels=True, node_size=1500, node_color="skyblue", node_shape="o",
                alpha=0.7, linewidths=4, font_size=25, font_color="grey", font_weight="bold", width=2,
                edge_color="grey")
        labels = {key: round(value, 1) for key, value in nx.get_node_attributes(self.graph, 'EOTX').items()}
        nx.draw_networkx_labels(self.graph, pos=self.pos, labels=labels)
        nx.draw_networkx_edge_labels(self.graph, pos=self.pos, edge_labels=nx.get_edge_attributes(self.graph, 'weight'))
        plt.savefig('graph.png')
        logging.info('Created network from JSON successfully!')

    def getpath(self):
        """Get path of the packet which arrived successfully."""
        return self.paths

    def newbatch(self):
        """Spawn new batch if old one is done."""
        if not self.done:
            logging.warning('Old batch is not done yet')
        self.batch += 1
        self.done = False
        for node in self.nodes:
            if str(node) == 'D':
                node.newbatch()

    def spawnpacket(self):
        """Spawn new packet at src and broadcast it."""
        self.packets.append(components.Packet(batch=self.batch, coding=self.coding, fieldsize=self.fieldsize))

    def update(self):
        """Update one timestep."""
        if not self.done:
            self.spawnpacket()
            for node in self.nodes:
                if str(node) != 'D':        # Destination should not send any packet
                    for packet in self.packets:
                        if node.getname() == packet.getpos():   # Just send the packet if you have it
                            self.broadcast(packet)
                else:
                    self.done = node.isdone()
            self.packets = self.newpackets.copy()       # Replace old packets by new ones
            self.newpackets = []                        # Clear list for next run
            for node in self.nodes:
                node.rcvpacket()
            return False
        else:
            return True

    def calceotx(self):
        """Calculate ETX for every node. i/j current node,
        N amount of nodes,
        e_ij loss between i and j,
        z_i expected amount of transmissions,
        p_ij probability j receives transmission from i (1-e_ij) edge weight,
        p_iK probability all nodes in K receives packet from i after z_i transmissions,
        Z_s = min(sum(z_i)) called cost,
        d(s) minimum cost of a path from s to t,
        c_sk cost of edge sk,
        R number of packets must be transmitted,
        x_ik amount of innovative packets received at k sent from i(see formula 5.2),
        q_iK is the probability that at least one node in set K receives transmission
        L_i load of each node sum_k(x_ik)
        K = 32 = Batchsize"""

        eotx_t = {str(node): 1 for node in self.nodes}
        eotx_p = eotx_t.copy()
        q = {}
        for node in self.nodes:
            self.graph.nodes[str(node)]['EOTX'] = float('inf')
            q[str(node)] = node
        q['D'].seteotx(0.)
        self.graph.nodes['D']['EOTX'] = 0.
        while q:
            node, value = min(q.items(), key=lambda x: x[1].geteotx())    # Calculate from destination to source
            del q[node]
            for neighbor in self.graph.neighbors(node):
                if neighbor not in q:
                    continue
                eotx_t[neighbor] += \
                    self.graph.edges[node, neighbor]['weight'] * eotx_p[neighbor] * self.graph.nodes[node]['EOTX']
                eotx_p[neighbor] *= (1 - self.graph.edges[node, neighbor]['weight'])
                self.graph.nodes[neighbor]['EOTX'] = eotx_t[neighbor] / (1 - eotx_p[neighbor])
                q[neighbor].seteotx(self.graph.nodes[neighbor]['EOTX'])


if __name__ == '__main__':
    llevel = logging.DEBUG
    logging.basicConfig(
        filename='main.log', level=llevel, format='%(asctime)s %(levelname)s\t %(message)s', filemode='w')
    args = parse_args()
    sim = Simulator(loglevel=llevel, jsonfile=args.json, coding=args.coding, fieldsize=args.fieldsize)
    for i in range(20):
        done = False
        while not done:
            done = sim.update()
        sim.newbatch()
    sim.drawused()
    logging.info('Packet arrived at destination. {}'.format(sim.getpath()))
