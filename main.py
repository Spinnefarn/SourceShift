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
    """Roundbased simulator to simulate traffic in meshed network."""

    def __init__(self, loglevel=logging.DEBUG, jsonfile=None, coding=None, fieldsize=2):
        self.config = {}
        self.graph = None
        self.nodes = []
        self.packets = []
        self.newpackets = []
        self.batch = 0
        self.coding = coding
        self.fieldsize = fieldsize
        import logging
        logging.basicConfig(filename='simulator.log', level=loglevel, format='%(asctime)s %(message)s', filemode='w')
        self.getready(jsonfile)

    def broadcast(self, packet):
        """Broadcast given packet to all neighbors."""
        nodename = packet.getpos()
        for neighbor in list(self.graph.neighbors(nodename)):
            if self.graph.edges[nodename, neighbor]['weight'] > random.random():    # roll dice
                info = packet.getall()  # Duplicate packet in receiver node
                nodes = info[6]
                if neighbor in nodes.keys():
                    # logging.warning('Loop detected!')
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
                            name.rcvpacket(batch=info[2], coding=info[4])
                            self.newpackets.append(
                                components.Packet(src=info[0], dst=info[1], batch=info[2], pos=neighbor,
                                                  coding=name.getcoded(), latency=(info[5] + 1), nodes=nodes))
        packet.update()

    def checkdst(self):
        """Check if destination received all information."""
        for node in self.nodes:
            if str(node) == 'D':
                return node.done()

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

    def getgraph(self):
        """Return graph."""
        return self.graph

    def getready(self, jsonfile):
        """Do the basic stuff to get ready."""
        self.createnetwork(readconf(jsonfile))
        self.calceotx()
        nx.draw_networkx(self.graph)
        plt.savefig('graph.png')
        logging.info('Created network from JSON successfully!')

    def getpath(self):
        """Get path of the packet which arrived successfully."""
        for packet in self.packets:
            if packet.getpos() == 'D':
                return packet.getpath()

    def spawnpacket(self):
        """Spawn new packet at src and broadcast it."""
        if self.checkdst():         # Spawn new batch if destination got all previous packets
            self.batch += 1
            self.packets = [components.Packet(batch=self.batch, coding=self.coding, fieldsize=self.fieldsize)]
        self.packets.append(components.Packet(batch=self.batch, coding=self.coding, fieldsize=self.fieldsize))

    def update(self):
        """Update one timestep."""
        self.spawnpacket()
        for node in self.nodes:
            if str(node) != 'D':        # Destination should not send any packet
                for packet in self.packets:
                    if node.getname() == packet.getpos():   # Just send the packet if you have it
                        self.broadcast(packet)
        self.packets = self.newpackets.copy()       # Replace old packets by new ones
        self.newpackets = []                        # Clear list for next run

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
        eotx_p = {str(node): 1 for node in self.nodes}
        q = {}
        for node in self.graph.nodes:
            self.graph.nodes[node]['EOTX'] = float('inf')
            q[node] = float('inf')
        q['D'] = 0.
        self.graph.nodes['D']['EOTX'] = 0.
        while q:
            node, value = min(q.items(), key=lambda x: x[1])    # Has to be calculated from destination to source
            del q[node]
            for neighbor in self.graph.neighbors(node):
                if neighbor not in q:
                    continue
                eotx_t[neighbor] += \
                    self.graph.edges[node, neighbor]['weight'] * eotx_p[neighbor] * self.graph.nodes[node]['EOTX']
                eotx_p[neighbor] *= (1 - self.graph.edges[node, neighbor]['weight'])
                self.graph.nodes[neighbor]['EOTX'] = eotx_t[neighbor] / (1 - eotx_p[neighbor])
                q[neighbor] = self.graph.nodes[neighbor]['EOTX']


if __name__ == '__main__':
    llevel = logging.DEBUG
    logging.basicConfig(
        filename='main.log', level=llevel, format='%(asctime)s %(levelname)s\t %(message)s', filemode='w')
    args = parse_args()
    sim = Simulator(loglevel=llevel, jsonfile=args.json, coding=args.coding, fieldsize=args.fieldsize)
    for i in range(4):
        done = False
        sim.spawnpacket()
        while not done:
            sim.update()
            done = sim.checkdst()
        logging.info('Packet arrived at destination. {}'.format(sim.getpath()))
