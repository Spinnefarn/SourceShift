#!/usr/bin/python
# coding=utf-8
"""Contains bacic network objects like Packets etc.."""
import numpy as np
import logging
import fifi


def determinant(matrix, size, field):
    """Solve determinant with laplace."""
    c = []
    d = 0
    if size == 0:
        for i in range(len(matrix)):
            d = (field.add(d, matrix[i]))
        return d
    elif len(matrix) == 1:
        return matrix[0]
    elif len(matrix) == 4:
        c = field.subtract(field.multiply(int(matrix[0]), int(matrix[3])),
                           field.multiply(int(matrix[1]), int(matrix[2])))
        return c
    else:
        for j in range(size):
            new_matrix = []
            for i in range(size*size):
                if i % size != j and i >= size:
                    new_matrix.append(matrix[i])

            c.append(field.multiply(int(determinant(new_matrix, size - 1, field)), int(matrix[j])))
        d = determinant(c, 0, field)
        return d


def calcrank(matrix, field):
    """Another try."""
    if not len(matrix):
        return 0
    elif len(matrix) == 1:
        return 1
    for i in range(len(matrix), 0, -1):
        for j in range(2 ** len(matrix[0])):
            binj = [int(x) for x in bin(j)[2:]]
            if sum(binj) == i:
                linematrix = []
                binj = [0] * (len(matrix[0]) - len(binj)) + binj
                for linenumber in range(i):
                    for idx, binv in enumerate(binj):
                        if binv:
                            linematrix.append(matrix[linenumber][idx])
                if determinant(linematrix, i, field):
                    return i


def makenice(trash, maxts):
    """Make trash nice to give back."""
    x, y = np.unique(trash, return_counts=True)
    x, y = list(x), list(y)
    trashdict = dict(zip(x, y))
    intdict = {}
    for ts in range(maxts):
        if ts not in trashdict:
            intdict[int(ts)] = 0
        else:
            intdict[int(ts)] = int(trashdict[ts])
    return {key: intdict[key] for key in sorted(intdict.keys())}


class Node:
    """Representation of a node on the network."""
    def __init__(self, name='S', coding=None, fieldsize=1, random=None):
        if random is None:
            np.random.seed(1)
        else:
            np.random.seed(random)
        self.name = name
        self.buffer = np.array([], dtype=int)
        self.fieldsize = 2 ** fieldsize
        self.incbuffer = []
        self.coding = coding
        self.batch = 0
        self.eotx = float('inf')
        self.deotx = float('inf')
        self.creditcounter = 0. if self.name != 'S' else float('inf')
        self.credit = 0.
        self.complete = (name == 'S')
        self.trash = []
        self.working = True
        self.realtrash = []
        self.quiet = False
        self.history = []
        self.sendhistory = []
        self.rank = 0 if self.name != 'S' else coding
        if fieldsize == 1:
            self.field = fifi.simple_online_binary()
        elif fieldsize == 4:
            self.field = fifi.simple_online_binary4()
        elif fieldsize == 8:
            self.field = fifi.simple_online_binary8()
        elif fieldsize == 16:
            self.field = fifi.simple_online_binary16()
        else:
            raise ValueError('Unsupported field size!')

    def __str__(self):
        return str(self.name)

    def buffpacket(self, batch=0, coding=None, preveotx=0, prevdeotx=0, special=False, ts=0):
        """Buffer incoming packets so they will be received at end of time slot."""
        self.incbuffer.append((batch, coding, preveotx, prevdeotx, special))
        self.history.append((batch, coding, preveotx, prevdeotx, special, ts))

    def becomesource(self):
        """Act like source. Will be triggered if all neighbors are complete."""
        if not self.complete:
            logging.error('All neighbors are complete but not this one? '.format(self.name))
        self.creditcounter = float('inf')

    def fail(self):
        """Set nodes state to fail."""
        self.working = False

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
            self.sendhistory.append(coding)
            return coding
        elif self.creditcounter > 0:  # Check tx credit
            if self.isdone():
                coding = [0]
                while sum(coding) == 0:  # Use random coding instead of recoding if rank is full
                    coding = np.random.randint(self.fieldsize, size=self.coding, dtype=int)
                return coding
            recoded = [0]
            while sum(recoded) == 0:
                modbuffer = np.array([], dtype=int)
                for line in self.buffer:
                    randomnumber = int(np.random.randint(self.fieldsize))
                    newline = [self.field.multiply(randomnumber, int(x)) for x in line]
                    if len(modbuffer):
                        modbuffer = np.vstack([modbuffer, newline])
                    else:
                        modbuffer = np.array([newline], dtype=int)
                recoded = []
                for i in range(len(self.buffer[0])):
                    summe = 0
                    for j in modbuffer[:, i]:
                        summe = self.field.add(int(summe), int(j))
                    recoded.append(summe)  # Recode to get new packet
            self.sendhistory.append(recoded)
            return np.array(recoded)
        else:
            return None

    def getcredit(self):
        """Tell the amount of tx credit."""
        return self.creditcounter

    def getcreditinc(self):
        """Tell the amount of delta tx credit."""
        return self.credit

    def geteotx(self):
        """Get eotx."""
        return self.eotx

    def getdeotx(self):
        """Get davids eotx."""
        return self.deotx

    def gethealth(self):
        """Get state working or not."""
        return self.working

    def getname(self):
        """Return name of the node."""
        return self.name

    def getquiet(self):
        """Return true if quiet."""
        return self.quiet

    def getrank(self):
        """Return current rank."""
        return self.rank

    def getrealtrash(self, maxts):
        """Return real interesting trash."""
        return makenice(self.realtrash, maxts)

    def getsent(self):
        """Return True if the node sent at least once."""
        return len(self.sendhistory) != 0

    def gettrash(self, maxts):
        """Return trash."""
        return makenice(self.trash, maxts)

    def heal(self):
        """Start working or continue if you do already."""
        self.working = True

    def isdone(self):
        """Return True if able to decode."""
        self.complete = self.coding == self.rank
        return self.complete

    def newbatch(self):
        """Make destination awaiting new batch."""
        self.batch += 1
        self.buffer = np.array([], dtype=int)
        self.complete = self.name == 'S'
        self.rank = self.coding if self.name == 'S' else 0
        self.quiet = False

    def rcvpacket(self, timestamp):
        """Add received Packet to buffer. Do this at end of timeslot."""
        while len(self.incbuffer):
            batch, coding, preveotx, prevdeotx, special = self.incbuffer.pop()
            if self.name == 'S':  # Cant get new information if you're source
                break
            elif batch < self.batch:
                continue
            elif batch > self.batch or not len(self.buffer):  # Delete it if you're working on deprecated batch
                self.buffer = np.array([coding], dtype=int)
                self.batch = batch
                self.rank = 1 if self.name != 'S' else self.coding
                self.complete = self.rank == self.coding
                self.creditcounter = 0.
                if self.quiet:
                    self.quiet = False
                if special and self.credit == 0:
                    self.creditcounter += 1
            else:  # Just add new information if its new
                if self.complete:        # Packet can just be linear dependent if rank is full
                    newrank = self.rank
                    if special and self.credit == 0:
                        self.creditcounter += 1
                else:
                    newrank = calcrank(np.vstack([self.buffer, coding]), self.field)
                if self.rank < newrank:
                    self.buffer = np.vstack([self.buffer, coding])
                    self.rank = newrank
                    self.complete = self.coding == newrank
                    if special and self.credit == 0:
                        self.creditcounter += 1
                elif preveotx > self.eotx:
                    self.realtrash.append(timestamp)
                else:
                    self.trash.append(timestamp)
            if preveotx > self.eotx or prevdeotx > self.deotx:
                self.creditcounter += self.credit

    def reducecredit(self):
        """Reduce tx credit."""
        self.creditcounter -= 1

    def setcredit(self, credit):
        """Set custom tx credit. In case of MOREresilience use the higher credit."""
        if self.credit < credit:
            self.credit = credit

    def seteotx(self, eotx=float('inf')):
        """Set eotx to given value."""
        self.eotx = eotx

    def setdeotx(self, eotx=float('inf')):
        """Set davids eotx to given value."""
        self.deotx = eotx

    def stopsending(self):
        """Stop sending if every neighbor is complete."""
        self.quiet = True
