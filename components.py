#!/usr/bin/python
# coding=utf-8
"""Contains bacic network objects like Packets etc.."""
import numpy as np
import logging


def determinant(matrix, size, fieldsize):
    """Solve determinant with laplace."""
    c = []
    d = 0
    if size == 0:
        for i in range(len(matrix)):
            d = (d + matrix[i]) % fieldsize
        return d
    elif len(matrix) == 1:
        return matrix[0]
    elif len(matrix) == 4:
        c = ((matrix[0] * matrix[3]) - (matrix[1] * matrix[2])) % fieldsize
        return c
    else:
        for j in range(size):
            new_matrix = []
            for i in range(size*size):
                if i % size != j and i >= size:
                    new_matrix.append(matrix[i])

            c.append((determinant(new_matrix, size - 1, fieldsize) * matrix[j] * ((-1) ** (j + 2))) % fieldsize)
        d = determinant(c, 0, fieldsize)
        return d


def calcrank(matrix, fieldsize):
    """Another try."""
    if not len(matrix):
        return 0
    elif len(matrix) == 1:
        return 1
    for i in range(len(matrix[:, 0]), 0, -1):
        for j in range(2 ** len(matrix[0])):
            binj = [int(x) for x in bin(j)[2:]]
            if sum(binj) == i:
                linematrix = []
                binj = [0] * (len(matrix[0]) - len(binj)) + binj
                for linenumber in range(i):
                    for idx, binv in enumerate(binj):
                        if binv:
                            linematrix.append(matrix[linenumber][idx])
                if determinant(linematrix, i, fieldsize):
                    return i


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
        self.creditcounter = 0. if self.name != 'S' else float('inf')
        self.credit = 0.
        self.complete = (name == 'S')
        self.trash = []
        self.quiet = False
        self.history = []
        self.sendhistory = []
        self.rank = 0 if self.name != 'S' else coding

    def __str__(self):
        return str(self.name)

    def buffpacket(self, batch=0, coding=None, preveotx=0, special=False, ts=0):
        """Buffer incoming packets so they will be received at end of time slot."""
        self.incbuffer.append((batch, coding, preveotx, special))
        self.history.append((batch, coding, preveotx, special, ts))

    def becomesource(self):
        """Act like source. Will be triggered if all neighbors are complete."""
        if not self.complete:
            logging.error('All neighbors are complete but not this one? '.format(self.name))
        self.creditcounter = float('inf')

    def isdone(self):
        """Return True if able to decode."""
        self.complete = self.coding == self.rank
        return self.complete

    def getbatch(self):
        """Return current batch."""
        return self.batch

    def getcoded(self):
        """Return a (re)coded packet."""
        if self.name == 'D':  # Make sure destination will never send a packet
            return None
        elif self.name == 'S':
            coding = [0]
            while sum(coding) == 0:  # Use random coding instead of recoding if rank is full
                coding = np.random.randint(self.fieldsize, size=self.coding)
            return coding
        elif self.creditcounter > 0:  # Check tx credit
            if self.isdone():
                coding = [0]
                while sum(coding) == 0:  # Use random coding instead of recoding if rank is full
                    coding = np.random.randint(self.fieldsize, size=self.coding)
                return coding
            recoded = [0]
            while sum(recoded) == 0:
                modbuffer = np.array([], dtype=int)
                for line in self.buffer:
                    if len(modbuffer):
                        newline = (np.random.randint(self.fieldsize) * line) % self.fieldsize
                        modbuffer = np.vstack([modbuffer, newline])
                    else:
                        newline = (np.random.randint(self.fieldsize) * line) % self.fieldsize
                        modbuffer = np.array([newline])
                recoded = []
                for i in range(len(self.buffer[0])):
                    recoded.append(sum(modbuffer[:, i]) % self.fieldsize)  # Recode to get new packet
            self.sendhistory.append(recoded)
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

    def getquiet(self):
        """Return true if quiet."""
        return self.quiet

    def getrank(self):
        """Return current rank."""
        return self.rank

    def gettrash(self, maxts):
        """Return trash."""
        x, y = np.unique(self.trash, return_counts=True)
        x, y = list(x), list(y)
        trashdict = dict(zip(x, y))
        intdict = {}
        for ts in range(maxts):
            if ts not in trashdict:
                intdict[int(ts)] = 0
            else:
                intdict[int(ts)] = int(trashdict[ts])
        values = [intdict[key] for key in sorted(intdict.keys())]
        return sorted(intdict.keys()), values

    def listenstate(self, information):
        """Set state someone told."""
        self.buffer = information[0].copy()
        self.incbuffer = []
        self.rank = information[1]
        self.trash = information[2].copy()
        if self.batch < information[3]:
            self.batch = information[3]
            self.quiet = False
        self.creditcounter = information[4]
        self.complete = self.coding == self.rank

    def newbatch(self):
        """Make destination awaiting new batch."""
        self.batch += 1
        self.buffer = np.array([], dtype=int)
        self.complete = self.name == 'S'
        self.quiet = False

    def rcvpacket(self, timestamp):
        """Add received Packet to buffer. Do this at end of timeslot."""
        for batch, coding, preveotx, special in self.incbuffer:
            if self.name == 'S':  # Cant get new information if you're source
                break
            elif batch > self.batch or not len(self.buffer):  # Delete it if you're working on deprecated batch
                self.buffer = np.array([coding], dtype=int)
                self.batch = batch
                self.complete = False
                if self.creditcounter == float('inf'):  # Return to normal if new batch arrives
                    self.creditcounter = 0.
                if self.quiet:
                    self.quiet = False
            else:  # Just add new information if its new
                if calcrank(self.buffer, self.fieldsize) < calcrank(np.vstack([self.buffer, coding]), self.fieldsize):
                    self.buffer = np.vstack([self.buffer, coding])
                else:
                    if self.complete:
                        logging.debug('Got linear dependent packet at {}, decoder full time = {}'.format(self.name,
                                                                                                         timestamp))
                        self.trash.append(timestamp)
                        pass
                    else:
                        logging.debug('Got linear dependent packet at {} coding {}, time {}'.format(self.name,
                                                                                                    str(coding),
                                                                                                    timestamp))
                        self.trash.append(timestamp)
            if special and self.credit == 0:
                self.creditcounter += 1
            if preveotx > self.eotx:
                self.creditcounter += self.credit
        self.incbuffer = []
        if self.name != 'S':
            self.rank = calcrank(self.buffer, self.fieldsize)
            self.complete = self.coding == self.rank

    def reducecredit(self):
        """Reduce tx credit."""
        self.creditcounter -= 1

    def setcredit(self, credit):
        """Set custom tx credit."""
        self.credit = credit

    def seteotx(self, eotx=float('inf')):
        """Set eotx to given value."""
        self.eotx = eotx

    def stopsending(self):
        """Stop sending if every neighbor is complete."""
        self.quiet = True

    def tellstate(self):
        """Tell current state."""
        return self.name, self.buffer.copy(), self.rank, self.trash.copy(), self.batch, self.creditcounter
