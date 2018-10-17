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
    parser.add_argument('-j',
                        dest='json',
                        help='should contain network configuration',
                        default='network2.json')
    return parser.parse_args()


def readconf(jsonfile):
    """Read configuration file."""
    with open(jsonfile) as f:
        config = json.loads(f.read())
    return config


class Simulator:
    """Roundbased simulator to simulate traffic in meshed network."""

    def __init__(self, loglevel=logging.DEBUG, jsonfile=None, coding=None):
        self.config = {}
        self.graph = None
        self.nodes = []
        self.packets = []
        self.newpackets = []
        self.batch = 0
        self.coding = coding
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
                    logging.warning('Loop detected!')
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
        self.nodes = [components.Node(name=name, coding=self.coding) for name in self.graph.nodes]

    def getgraph(self):
        """Return graph."""
        return self.graph

    def getready(self, json):
        """Do the basic stuff to get ready."""
        self.createnetwork(readconf(json))
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
        if self.checkdst():         # Spawn new batch if destination got all packets
            self.batch += 1
            self.packets = [components.Packet(batch=self.batch, coding=self.coding)]
        if len(self.packets) == 0:  # Spawn new packet of current batch if old one gets lost
            self.packets.append(components.Packet(batch=self.batch, coding=self.coding))

    def update(self):
        """Update one timestep."""
        if len(self.packets) == 0:      # Spawn new packet if there is none at the network
            self.spawnpacket()
        for node in self.nodes:
            if str(node) != 'D':        # Destination should not send any packet
                for packet in self.packets:
                    if node.getname() == packet.getpos():   # Just send the packet if you have it
                        self.broadcast(packet)
        self.packets = self.newpackets.copy()       # Replace old packets by new ones
        self.newpackets = []                        # Clear list for next run

    def calcetx(self):
        """Calculate ETX for every node. i/j current node, N amount of nodes, e_ij loss between,
        z_i expected amount of transmissions"""
        e_ik = {}
        e_ij = {}
        z_i = {}
        for node in self.graph.nodes:
            path_ik = nx.dijkstra_path(self.graph, node, 'D')
            e_ik[node] = 1
            for i in range(len(path_ik) - 1):
                e_ik[node] *= self.graph.edges[path_ik[i], path_ik[i + 1]]['weight']
            path_ij = nx.dijkstra_path(self.graph, 'S', node)
            e_ij[node] = 1
            for i in range(len(path_ij) - 1):
                e_ij[node] *= self.graph.edges[path_ij[i], path_ij[i + 1]]['weight']


if __name__ == '__main__':
    llevel = logging.DEBUG
    logging.basicConfig(
        filename='main.log', level=llevel, format='%(asctime)s %(levelname)s\t %(message)s', filemode='w')
    args = parse_args()
    sim = Simulator(loglevel=llevel, jsonfile=args.json, coding=4)
    for i in range(4):
        done = False
        sim.spawnpacket()
        while not done:
            sim.update()
            done = sim.checkdst()
        logging.info('Packet arrived at destination. {}'.format(sim.getpath()))

