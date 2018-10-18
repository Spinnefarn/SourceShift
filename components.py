#!/usr/bin/python
# coding=utf-8
"""Contains bacic network objects like Packets etc.."""
import numpy as np
import logging


class Packet:
    """Representation of a single packet on the network."""
    def __init__(self, src='S', dst='D', latency=0, batch=None, pos='S', coding=None, fieldsize=2, nodes=None):
        self.src = src
        self.dst = dst
        self.batch = batch
        self.pos = pos
        self.fieldsize = fieldsize
        if isinstance(coding, int):
            self.coding = [0]
            while sum(self.coding) == 0:
                self.coding = np.random.randint(fieldsize, size=coding)
        else:
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
    def __init__(self, name='S', coding=None, fieldsize=2):
        self.name = name
        self.buffer = np.array([], dtype=int)
        self.coding = coding
        self.fieldsize = fieldsize
        self.batch = 0
        self.etx = float('inf')
        self.eotx = float('inf')
        self.complete = (name == 'S')

    def __str__(self):
        return str(self.name)

    def done(self):
        """Return True if able to decode."""
        if self.name != 'S' and not self.complete:
            self.complete = self.coding == np.linalg.matrix_rank(self.buffer)
        return self.complete

    def getname(self):
        """Return name of the node."""
        return self.name

    def getcoded(self):
        """Return a (re)coded packet."""
        if self.name == 'S':
            coding = [0]
            while sum(coding) == 0:     # Create a new random packet at source(no recoding)
                coding = np.random.randint(self.fieldsize, size=self.coding)
                return coding
        recoded = []
        for i in range(len(self.buffer[0])):
            recoded.append(sum(self.buffer[:, i]) % self.fieldsize)              # Recode to get new packet
        return np.array(recoded)

    def rcvpacket(self, batch=0, coding=None):
        """Add received Packet to buffer."""
        if self.name == 'S':        # Cant get new information if you're source
            return
        elif batch != self.batch or not len(self.buffer):   # Delete it if you're working on deprecated batch
            self.buffer = np.array([coding], dtype=int)
        else:    # Just add new information if its new
            if np.linalg.matrix_rank(self.buffer) < np.linalg.matrix_rank(np.vstack([self.buffer, coding])):
                self.buffer = np.vstack([self.buffer, coding])
            else:
                logging.info('Got linear dependent packet at ' + self.name + str(coding))