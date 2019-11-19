#!/usr/bin/python
# coding=utf-8
"""Contains bacic network objects like Packets etc.."""
import numpy as np
import kodo
import os


def selfieldsize(fieldsize=2):
    """Chose fieldsize."""
    if fieldsize == 2:
        return kodo.field.binary
    elif fieldsize == 4:
        return kodo.field.binary4
    elif fieldsize == 8:
        return kodo.field.binary8
    elif fieldsize == 16:
        return kodo.field.binary16
    else:
        import logging
        logging.error('Unsupported fieldsize {}'.format(fieldsize))


class Node:
    """Representation of a node on the network."""
    def __init__(self, name='S', coding=None, fieldsize=1, random=None):
        if random is None:
            np.random.seed(1)
        else:
            np.random.seed(random)
        symbol_size = 8
        self.coder = None
        self.field = selfieldsize(2 ** fieldsize)
        if name == 'S':
            self.coder = kodo.RLNCEncoder(self.field, coding, symbol_size)
            self.data = bytearray(os.urandom(self.coder.block_size()))
            self.coder.set_symbols_storage(self.data)
        else:
            self.coder = kodo.RLNCDecoder(self.field, coding, symbol_size)
            self.data = bytearray(self.coder.block_size())
            self.coder.set_symbols_storage(self.data)
        self.name = name
        self.fieldsize = 2 ** fieldsize
        self.incbuffer = []
        self.coding = coding
        self.symbol_size = symbol_size
        self.batch = 0
        self.eotx = float('inf')
        self.deotx = float('inf')
        self.creditcounter = 0. if self.name != 'S' else float('inf')
        self.credit = 0.
        self.sent = False
        self.working = True
        self.priority = 0.
        self.quiet = False
        self.history = []
        self.sendhistory = []
        self.rank = 0 if self.name != 'S' else coding

    def __str__(self):
        return str(self.name)

    def buffpacket(self, batch=0, coding=None, preveotx=0, prevdeotx=0, special=False, ts=0):
        """Buffer incoming packets so they will be received at end of time slot."""
        self.incbuffer.append((batch, coding.copy(), preveotx, prevdeotx, special))
        # self.history.append((batch, coding.copy(), preveotx, prevdeotx, special, ts))

    def becomesource(self):
        """Act like source. Will be triggered if all neighbors are complete."""
        if self.coder.is_complete() or self.creditcounter == 0. or self.credit == 0.:   # Ignore MORE it it ignores you
            self.creditcounter += 1     # Send next time slot if you should and your done

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
        elif self.name == 'S' or self.creditcounter > 0:
            self.sent = True
            # self.sendhistory.append((self.batch, self.coder.rank(), self.coder.write_payload()))
            # return self.sendhistory[-1][-1]
            return self.coder.produce_payload()
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
        if not self.priority:
            return self.eotx
        else:
            return self.priority

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

    def getsent(self):
        """Return True if the node sent at least once."""
        return self.sent

    def heal(self):
        """Start working or continue if you do already."""
        self.working = True

    def isdone(self):
        """Return True if able to decode."""
        return True if self.name == 'S' else self.coder.is_complete()

    def newbatch(self):
        """Make destination awaiting new batch."""
        self.batch += 1
        self.rank = self.coding if self.name == 'S' else 0
        if self.name != 'S':
            self.coder = kodo.RLNCDecoder(self.field, self.coding, self.symbol_size)
            self.data = bytearray(self.coder.block_size())
            self.coder.set_symbols_storage(self.data)
        self.quiet = False

    def rcvpacket(self):
        """Add received Packet to buffer. Do this at end of time slot."""
        while len(self.incbuffer):
            batch, coding, preveotx, prevdeotx, special = self.incbuffer.pop()
            if self.name == 'S':  # Cant get new information if you're source
                continue
            elif batch < self.batch:
                continue
            elif batch > self.batch or not self.coder.rank():  # Delete it if you're working on deprecated batch
                self.batch = batch
                self.sent = False
                self.coder = kodo.RLNCDecoder(self.field, self.coding, self.symbol_size)
                self.data = bytearray(self.coder.block_size())
                self.coder.set_symbols_storage(self.data)
                self.coder.consume_payload(coding)
                self.rank = self.coder.rank()
                self.creditcounter = 0.
                if self.quiet:
                    self.quiet = False
                if special and not self.credit:
                    self.creditcounter += 1
            else:  # Just add new information if its new
                if not self.coder.is_complete():
                    self.coder.consume_payload(coding)
                    newrank = self.coder.rank()
                else:
                    newrank = self.rank       # Full coder does not get new information
                if self.rank <= newrank:
                    self.rank = newrank
                    if special and not self.credit:
                        self.creditcounter += 1
            if preveotx > self.eotx or prevdeotx > self.deotx or (self.priority != 0. and self.priority > preveotx):
                self.creditcounter += self.credit

    def reducecredit(self):
        """Reduce tx credit."""
        self.creditcounter -= 1

    def resetcredit(self):
        """Reset credit."""
        if self.name != 'S':
            self.credit = 0.
            self.creditcounter = 0.

    def rmcredit(self):
        """Reset credit counter."""
        if self.name != 'S':
            self.creditcounter = 0.

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

    def setpriority(self, priority=0.):
        """Set priority in case to use ANCHOR."""
        self.priority = priority

    def stopsending(self):
        """Stop sending if every neighbor is complete."""
        self.quiet = True
