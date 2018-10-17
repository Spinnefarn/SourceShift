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


class Simulator:
    """Roundbased simulator to simulate traffic in meshed network."""

    def __init__(self, loglevel, json):
        self.config = {}
        self.graph = None
        self.nodes = []
        self.packets = []
        self.newpackets = []
        import logging
        logging.basicConfig(filename='simulator.log', level=loglevel, format='%(asctime)s %(message)s', filemode='w')
        self.getready(json)

    def broadcast(self, packet):
        """Broadcast given packet to all neighbors."""
        nodename = packet.getpos()
        for neighbor in list(self.graph.neighbors(nodename)):
            if self.graph.edges[nodename, neighbor]['weight'] > random.random():
                info = packet.getall()  # Duplicate packet in receiver node
                nodes = info[6]
                if neighbor in nodes.keys():
                    logging.warning('Loop detected!')
                nodes[neighbor] = info[5] + 1
                self.newpackets.append(components.Packet(src=info[0], dst=info[1], batch=info[2], pos=neighbor,
                                                         coding=info[4], latency=(info[5] + 1), nodes=nodes))
        packet.update()

    def checkrcv(self):
        """Check if destination received all information."""
        for packet in self.packets:
            if packet.getpos() == 'D':
                return True

    def createnetwork(self, config):
        """Create network using networkx library based on configuration given as dict."""
        configlist = []
        for (src, con) in config.items():
            for (dst, wgt) in con.items():
                configlist.append((src, dst, wgt))
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from(configlist)
        self.nodes = [components.Node(name=name) for name in self.graph.nodes]

    def getgraph(self):
        """Return graph."""
        return self.graph

    def getready(self, json):
        """Do the basic stuff to get ready."""
        self.createnetwork(self.readconf(json))
        nx.draw_networkx(self.graph)
        plt.savefig('graph.png')
        logging.info('Created network from JSON successfully!')

    def getpath(self):
        """Get path of the packet which arrived successfully."""
        for packet in self.packets:
            if packet.getpos() == 'D':
                return packet.getpath()

    def readconf(self, jsonfile):
        """Read configuration file."""
        with open(jsonfile) as f:
            config = json.loads(f.read())
        return config

    def spawnpacket(self):
        """Spawn new packet at src and broadcast it."""
        if len(self.packets) == 0:
            self.packets.append(components.Packet())

    def update(self):
        """Update one timestep."""
        for node in self.nodes:
            for packet in self.packets:
                if node.getname() == packet.getpos():
                    self.broadcast(packet)
        self.packets = self.newpackets.copy()       # Replace old packets by new ones
        self.newpackets = []                        # Clear list for next run


if __name__ == '__main__':
    llevel = logging.DEBUG
    logging.basicConfig(filename='main.log', level=llevel, format='%(asctime)s %(message)s', filemode='w')
    args = parse_args()
    sim = Simulator(llevel, args.json)
    done = False
    while not done:
        sim.spawnpacket()
        sim.update()
        done = sim.checkrcv()
    logging.info('Packet arrived at destination. It went following path(Node:Timeslot): {}'.format(sim.getpath()))
