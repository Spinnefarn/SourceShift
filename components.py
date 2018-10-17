#!/usr/bin/python
# coding=utf-8
"""Contains bacic network objects like Packets etc.."""


class Packet:
    """Representation of a single packet on the network."""
    def __init__(self, src='S', dst='D', latency=0, batch=None, pos='S', coding=None, nodes=None):
        self.src = src
        self.dst = dst
        self.batch = batch
        self.pos = pos
        self.coding = coding
        self.latency = latency
        if nodes is None:
            self.nodes = {}
        else:
            self.nodes = nodes

    def __str__(self):
        return str(self.pos)

    def update(self, node=None):
        """Add current location to list of visited locations."""
        self.latency += 1
        if node is not None:
            self.nodes[node] = self.latency
            self.pos = node

    def getpos(self):
        """Return current position."""
        return self.pos

    def getpath(self):
        """Return route packet has traveled."""
        return self.nodes.copy()

    def getall(self):
        """Return all informaion."""
        return self.src, self.dst, self.batch, self.pos, self.coding, self.latency, self.nodes.copy()


class Node:
    """Representation of a node on the network."""
    def __init__(self, name):
        self.name = name
        self.buffer = []
        self.etx = float('inf')
        self.eotx = float('inf')
        self.complete = (name == 'S')

    def __str__(self):
        return str(self.name)

    def getname(self):
        """Return name of the node."""
        return self.name
