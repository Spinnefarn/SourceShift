#!/usr/bin/python
# coding=utf-8
"""Contains bacic network objects like Packets etc.."""
import numpy as np
import logging


class Node:
    """Representation of a node on the network."""

    def __init__(self, name='S', coding=None, fieldsize=2):
        self.name = name
        self.buffer = np.array([], dtype=int)
        self.incbuffer = []
        self.coding = coding
        self.fieldsize = fieldsize
        self.batch = 0
        self.eotx = float('inf')
        self.creditcounter = 0.
        self.credit = 0.
        self.complete = (name == 'S')
        self.trash = []

    def __str__(self):
        return str(self.name)

    def buffpacket(self, batch=0, coding=None, preveotx=0):
        """Buffer incoming packets so they will be received at end of time slot."""
        self.incbuffer.append((batch, coding, preveotx))

    def isdone(self):
        """Return True if able to decode."""
        return self.complete

    def getbatch(self):
        """Return current batch."""
        return self.batch

    def getcoded(self):
        """Return a (re)coded packet."""
        if self.name == 'D':            # Make sure destination will never send a packet
            return None
        elif self.name == 'S':
            coding = [0]
            while sum(coding) == 0:  # Use random coding instead of recoding if rank is full
                coding = np.random.randint(self.fieldsize, size=self.coding)
            return coding
        elif self.credit > 0:           # Check tx credit
            if self.complete or self.isdone():
                coding = [0]
                while sum(coding) == 0:     # Use random coding instead of recoding if rank is full
                    coding = np.random.randint(self.fieldsize, size=self.coding)
                return coding
            recoded = []
            modbuffer = np.array([], dtype=int)
            for line in self.buffer:
                newline = [0]
                if len(modbuffer):
                    while sum(newline) == 0:
                        newline = (np.random.randint(self.fieldsize) * line) % self.fieldsize
                    modbuffer = np.vstack([modbuffer, newline])
                else:
                    while sum(newline) == 0:
                        newline = (np.random.randint(self.fieldsize) * line) % self.fieldsize
                    modbuffer = np.array([newline])
            for i in range(len(self.buffer[0])):
                recoded.append(sum(modbuffer[:, i]) % self.fieldsize)  # Recode to get new packet
            return np.array(recoded)
        else:
            return None

    def getcredit(self):
        """Tell the amount of tx credit."""
        return self.creditcounter

    def geteotx(self):
        """Get eotx."""
        return self.eotx

    def getname(self):
        """Return name of the node."""
        return self.name

    def gettrash(self):
        """Return trash."""
        return np.unique(self.trash, return_counts=True)

    def newbatch(self):
        """Make destination awaiting new batch."""
        self.batch += 1
        self.buffer = np.array([], dtype=int)

    def rcvpacket(self, timestamp):
        """Add received Packet to buffer. Do this at end of timeslot."""
        for batch, coding, preveotx in self.incbuffer:
            if self.name == 'S':  # Cant get new information if you're source
                return
            elif batch != self.batch or not len(self.buffer):  # Delete it if you're working on deprecated batch
                self.buffer = np.array([coding], dtype=int)
            else:  # Just add new information if its new
                if np.linalg.matrix_rank(self.buffer) < np.linalg.matrix_rank(np.vstack([self.buffer, coding])):
                    self.buffer = np.vstack([self.buffer, coding])
                else:
                    if self.complete or self.isdone():
                        logging.info('Got linear dependent packet at {}, decoder full time = {}'.format(self.name,
                                                                                                         timestamp))
                        self.trash.append(timestamp)
                    else:
                        logging.info('Got linear dependent packet at {} coding {}, time {}'.format(self.name,
                                                                                                    str(coding),
                                                                                                    timestamp))
                        self.trash.append(timestamp)
            if preveotx > self.eotx:
                self.creditcounter += self.credit
        self.incbuffer = []
        if self.name != 'S' and not self.complete:
            self.complete = self.coding == np.linalg.matrix_rank(self.buffer)

    def reducecredit(self):
        """Reduce tx credit."""
        self.creditcounter -= 1

    def setcredit(self, credit):
        """Set custom tx credit."""
        self.credit = credit

    def seteotx(self, eotx=float('inf')):
        """Set eotx to given value."""
        self.eotx = eotx
