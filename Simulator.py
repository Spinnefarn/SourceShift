#!/usr/bin/python
# coding=utf-8
"""Main script, we will see what this will do."""
import components
import json
import networkx as nx
from matplotlib import use
use('Agg')
import matplotlib.pylab as plt
import random
import os
import logging


def readconf(jsonfile):
    """Read configuration file."""
    if jsonfile is not None and os.path.exists(jsonfile):
        with open(jsonfile) as file:
            config = json.loads(file.read())
        pos = {node: [config['nodes'][node]['x'], config['nodes'][node]['y']] for node in config['nodes']}
        return config['links'], pos


class Simulator:
    """Round based simulator to simulate traffic in meshed network."""
    def __init__(self, jsonfile=None, coding=None, fieldsize=2, sendall=0, own=True, edgefail=None, nodefail=None,
                 allfail=False, randcof=(10, 0.5), folder='.', maxduration=0, randomseed=None, sourceshift=False,
                 david=False, edgefailprob=0.1):
        self.airtime = {'None': {}}
        self.sourceshift = sourceshift
        self.david = david
        self.edgefail = edgefail
        self.nodefail = nodefail
        self.allfail = allfail
        self.edgefailprob = edgefailprob
        logging.debug('Random seed: '.format(randomseed))
        self.random = randomseed
        self.maxduration = maxduration
        self.config = {}
        self.graph = None
        self.folder = folder
        self.prevfail = None
        self.failhist = {}
        self.interresting = []
        self.nodes = []
        self.ranklist = {}
        self.z = {}
        self.batch = 0
        self.sendam = sendall
        self.batchhist = []
        self.resilience = [1, 1]
        self.mcut = []
        self.dijkstra = []
        self.coding = coding
        self.fieldsize = fieldsize
        self.pos = None
        self.getready(jsonfile=jsonfile, randcof=randcof)
        self.json = jsonfile
        self.randcof = randcof
        self.done = False
        self.path = {'None': {}}
        self.timestamp = 0
        self.own = own
        self.donedict = {}

    def broadcast(self, node):
        """Broadcast given packet to all neighbors."""
        packet = node.getcoded()
        for neighbor in list(self.graph.neighbors(str(node))):
            if self.graph.edges[str(node), neighbor]['weight'] > random.random():  # roll dice
                if self.graph.edges[str(node), neighbor]['weight'] == 0:
                    logging.error('Received using dead link!')
                if neighbor != 'S':  # Source will not receive a packet, but still written down
                    for name in self.nodes:  # Add received Packet to buffer with coding
                        if str(name) == neighbor:
                            if name.gethealth():    # Broken nodes should not receive
                                special = self.checkspecial(node, neighbor) if self.own else False
                                name.buffpacket(batch=node.getbatch(), coding=packet, preveotx=node.geteotx(),
                                                prevdeotx=node.getdeotx(), special=special, ts=self.timestamp)
                                if str(node) + neighbor not in self.path[self.getidentifier()]:
                                    self.path[self.getidentifier()][(str(node) + neighbor)] = []
                                self.path[self.getidentifier()][(str(node) + neighbor)].append(self.timestamp)
                            break

    def calcairtime(self):
        """Calculate the amount of used airtime in total."""
        summe = 0
        ident = self.getidentifier()
        for node in self.airtime[ident].keys():
            summe += len(self.airtime[ident][node])
        return summe

    def calcdeotx(self):
        """Calculate second layer of more like david would do."""
        x = {}
        for node in self.nodes:
            if str(node) == 'D':
                continue
            for neighbor in self.graph.neighbors(str(node)):
                if self.graph.nodes[str(node)]['EOTX'] > self.graph.nodes[neighbor]['EOTX']:
                    x[str(node) + neighbor] = self.z[str(node)] * self.graph.edges[str(node), neighbor]['weight'] * \
                                       self.calce(str(node), notnode=neighbor)
        bestlinks = [edge for edge, edgevalue in x.items() if edgevalue >= self.david]
        bestlinks = [(edge[0], edge[1], self.graph.edges[edge[0], edge[1]]['weight']) for edge in bestlinks]
        self.graph.remove_edges_from(bestlinks)
        eotx = self.geteotx()
        for node in self.nodes:
            node.setdeotx(eotx[str(node)])
            self.graph.nodes[str(node)]['DEOTX'] = eotx[str(node)]
        self.calc_tx_credit(david=True)
        self.graph.add_weighted_edges_from(bestlinks)

    def calce(self, node, notnode=None, david=False):
        """Calc the probability a packet is not received by any destination."""
        e = 1
        eotx = 'DEOTX' if david else 'EOTX'
        for neighbor in self.graph.neighbors(node):
            if self.graph.nodes[neighbor][eotx] < self.graph.nodes[node][eotx] and notnode != neighbor:
                e *= (1 - self.graph.edges[node, neighbor]['weight'])
        return e

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

        eotx = self.geteotx()
        for node in self.nodes:
            node.seteotx(eotx[str(node)])
            self.graph.nodes[str(node)]['EOTX'] = eotx[str(node)]

    # noinspection PyTypeChecker
    def calcres(self):
        """Calculate resilience of network based on used edges with no failure."""
        mincut = nx.minimum_edge_cut(self.graph, s='S', t='D')
        self.resilience[0] = 1 - (self.edgefailprob ** len(mincut))
        logging.info('Resilience for graph is {}'.format(self.resilience[0]))
        if self.interresting:
            intedges = [element for element in self.interresting if len(element) == 2]
            edgelist = [(element[0], element[1], self.graph.edges[element[0], element[1]]['weight'])
                        for element in intedges]
            resg = nx.Graph()
            resg.add_weighted_edges_from(edgelist)
            mincut = nx.minimum_edge_cut(resg, s='S', t='D')
            self.resilience[1] = 1 - (self.edgefailprob ** len(mincut))
            if self.david:
                logging.info('Resilience for MOREresilience is {}'.format(self.resilience[1]))
            elif self.sourceshift and self.own:
                logging.info('Resilience for MORELESS is {}'.format(self.resilience[1]))
            elif self.sourceshift:
                logging.info('Resilience for source shift is {}'.format(self.resilience[1]))
            elif self.own:
                logging.info('Resilience for own approach is {}'.format(self.resilience[1]))
            else:
                logging.info('Resilience for MORE is {}'.format(self.resilience[1]))

    def calc_tx_credit(self, david=False):
        """Calculate the amount of tx credit the receiver gets."""
        if david:
            l_n = {node: self.graph.nodes[node]['DEOTX'] for node in self.graph.nodes}
        else:
            l_n = {node: self.graph.nodes[node]['EOTX'] for node in self.graph.nodes}
        l_n = sorted(l_n.items(), key=lambda kv: kv[1])
        l_i = {nodename[0]: 0 for nodename in l_n if nodename[0] != 'D'}
        l_i['S'] = 1
        for idx, node in enumerate(list(l_i.keys())[::-1]):
            try:
                if david:
                    self.calcz(node, l_i[node], david)
                else:
                    self.z[node] = l_i[node] / (1 - self.calce(node))
            except ValueError:
                continue
            p = 1
            for idx2, j in enumerate(list(l_i.keys())[:(len(l_i.keys()) - idx - 1)]):
                try:
                    if idx2 == 0:
                        p *= (1 - self.graph.edges[node, 'D']['weight'])
                    else:
                        p *= (1 - self.graph.edges[node, list(l_i.keys())[idx2 - 1]]['weight'])
                except KeyError:
                    pass
                try:
                    l_i[j] += self.z[node] * p * self.graph.edges[node, j]['weight']
                except KeyError:
                    pass
        for node in self.nodes:
            if str(node) == 'D':
                continue
            somevalue = 0
            for neighbor in self.graph.neighbors(str(node)):
                if david:
                    if self.graph.nodes[neighbor]['DEOTX'] > node.getdeotx():
                        somevalue += self.z[neighbor] * self.graph.edges[str(node), neighbor]['weight']
                else:
                    if self.graph.nodes[neighbor]['EOTX'] > node.geteotx():
                        somevalue += self.z[neighbor] * self.graph.edges[str(node), neighbor]['weight']
            try:
                node.setcredit(self.z[str(node)] / somevalue)
            except ZeroDivisionError:
                pass

    def calcz(self, nodename, li, david):
        """Calculate metric like david does in MOREresilience."""
        z = []
        try:
            z.append(li / (1 - self.calce(nodename)))
        except ZeroDivisionError:        # In case removing a link broke the regular connection
            pass
        if david:
            try:
                z.append(li / (1 - self.calce(nodename, david=david)))
            except ZeroDivisionError:
                pass
        self.z[nodename] = max(z)

    def checkconnection(self):
        """Check for given list of potential failures if the graph is still connected."""
        lostlist = []
        for fail in self.interresting:
            if len(fail) == 1:
                edgelist = [(fail, neighbor, self.graph.edges[fail, neighbor]['weight'])
                            for neighbor in self.graph.neighbors(fail)]
            else:
                edgelist = [(fail[0], fail[1], self.graph.edges[fail[0], fail[1]]['weight'])]
            self.graph.remove_edges_from(edgelist)
            try:
                nx.shortest_path(self.graph, source='S', target='D')
            except nx.exception.NetworkXNoPath:
                lostlist.append(fail)
            self.graph.add_weighted_edges_from(edgelist)
        self.interresting = [fail for fail in self.interresting if fail not in lostlist]

    def checkduration(self):
        """Calculate duration since batch was started. To know there is no connection SD."""
        if len(self.batchhist) == 0:
            return True
        if self.maxduration:
            maxval = self.maxduration
        else:
            maxval = 100 * self.batchhist[0]
        if len(self.batchhist) == 1:
            if self.timestamp - self.batchhist[0] >= maxval:
                logging.info('Stopped batch after {} timesteps'.format(self.timestamp - self.batchhist[0]))
                return False
            else:
                return True
        else:
            if self.timestamp - self.batchhist[-1] >= maxval:
                logging.info('Stopped batch after {} timesteps'.format(self.timestamp - self.batchhist[-1]))
                return False
            else:
                return True

    def checkspecial(self, node, neighbor):
        """Return True if node should be able to send over special metric."""
        for invnei in self.graph.neighbors(neighbor):
            if self.graph.nodes[neighbor]['EOTX'] > self.graph.nodes[invnei]['EOTX']:   # Maybe a different way
                if invnei != str(node):     # Don't think your source is a different way
                    for invnode in self.nodes:
                        if str(invnode) == invnei:
                            if node.getbatch() >= invnode.getbatch():
                                return True
                            break
        return False

    def checkstate(self):
        """Node should stop sending if all neighbors have complete information."""
        for node in self.nodes:
            if node.getquiet():  # Do not check nodes twice
                continue
            allcomplete = node.isdone()  # Just check if node itself is done
            for neighbor in self.graph.neighbors(str(node)):
                if not allcomplete:
                    break
                for neighbornode in self.nodes:
                    if str(neighbornode) == neighbor:
                        allcomplete = (node.getbatch() == neighbornode.getbatch() and neighbornode.isdone())
                        # Every neighbor closer to destination has to be done at current batch
                        break
            if allcomplete:
                node.stopsending()
        for node in self.nodes:
            if node.getcredit() != float('inf') and node.isdone():
                for neighbor in self.graph.neighbors(str(node)):
                    for neighbornode in self.nodes:
                        if str(neighbornode) == neighbor:
                            if not neighbornode.isdone() and node.geteotx() > neighbornode.geteotx():
                                node.becomesource()

    def createnetwork(self, config=None, randcof=(10, 0.5)):
        """Create network using networkx library based on configuration given as dict."""
        if config is None:
            graph = nx.random_geometric_graph(randcof[0], randcof[1])
            # graph = nx.barabasi_albert_graph(randcof[0], 2)     # not sure which option for random graphes to take
            mapping = dict(zip(graph.nodes(), "SDABCEFGHIJKLMNOPQRTUVWXYZ"[:randcof[0]]))
            self.graph = nx.relabel_nodes(graph, mapping)
            self.pos = nx.kamada_kawai_layout(self.graph)
            for edge in list(self.graph.edges):
                self.graph.edges[edge]['weight'] = round(random.uniform(0.05, 0.95), 1)
        else:
            self.pos = config[1]
            configlist = []
            for edge in config[0]:
                configlist.append((edge['nodes'][0], edge['nodes'][1], edge['loss']))
            self.graph = nx.Graph()
            self.graph.add_weighted_edges_from(configlist)
        self.nodes = [components.Node(name=name, coding=self.coding, fieldsize=self.fieldsize, random=self.random)
                      for name in self.graph.nodes]
        for node in self.nodes:
            self.ranklist[str(node)] = []

    def drawtrash(self, kind=None):
        """Draw linear dependent packets over time and nodes. Do not use! Will move to plotter, at some time."""
        maxval, sumval = [], []
        width = self.batch * 2
        plt.figure(figsize=(width, 5))
        trashdict = {}
        amts = {ts: 0 for ts in range(self.timestamp)}
        cmap, colorcounter = plt.get_cmap('tab20'), 0
        for node in self.nodes:
            if str(node) != 'S':
                if kind == 'real':
                    trash, amount = node.getrealtrash(self.timestamp)
                else:
                    trash, amount = node.gettrash(self.timestamp)
                trashdict[str(node)] = (trash, amount)
                plt.bar(trash, amount, bottom=list(amts.values()), label=str(node), color=cmap(colorcounter), alpha=0.5)
                for trashtime, number in zip(trash, amount):
                    if trashtime in amts:
                        amts[trashtime] += number
                    else:
                        amts[trashtime] = number
                if len(amount):
                    maxval.append(max(amount))
                    sumval.append(sum(amount))
            colorcounter += 1
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('Amount of linear depended packets')
        plt.xlabel('Timestamp')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        # plt.yticks(range(1, max(amts.values()) + 1, 1))
        plt.title('Amount of linear dependent packets for each node.')
        # plt.grid(True)
        plt.tight_layout()
        own = 'own' if self.own else ''
        if kind == 'real':
            plt.savefig('{}/{}reallineardependent.pdf'.format(self.folder, own))
        else:
            plt.savefig('{}/{}lineardependent.pdf'.format(self.folder, own))
        plt.close()
        if kind == 'real':
            logging.info('{} linear dependent packets arrived from parents.'.format(sum(sumval)))
        else:
            logging.info('{} linear dependent packets arrived by overhearing children.'.format(sum(sumval)))

    def failall(self):
        """Kill one of all."""
        if self.prevfail is None:
            if len(self.interresting[0]) == 1:
                for node in self.nodes:
                    if str(node) == self.interresting[0]:
                        self.failnode(self.nodes.index(node))
                        break
            elif len(self.interresting[0]) == 2:
                try:
                    self.failedge(list(self.graph.edges).index((self.interresting[0][0], self.interresting[0][1])))
                except ValueError:
                    self.failedge(list(self.graph.edges).index((self.interresting[0][1], self.interresting[0][0])))
            else:
                logging.error('Something crazy in interesting list: {}'.format(self.interresting[0]))
        elif isinstance(self.prevfail, tuple):
            try:
                newidx = self.interresting.index(self.prevfail[0][0] + self.prevfail[0][1]) + 1
            except ValueError:
                newidx = self.interresting.index(self.prevfail[0][1] + self.prevfail[0][0]) + 1
            if len(self.interresting) > newidx:
                newfail = self.interresting[newidx]
                try:
                    self.failedge(list(self.graph.edges).index((newfail[0], newfail[1])))
                except ValueError:
                    self.failedge(list(self.graph.edges).index((newfail[1], newfail[0])))
            else:
                return False
        else:
            newfail = self.interresting[self.interresting.index(str(self.nodes[self.prevfail])) + 1]
            if isinstance(newfail, str) and len(newfail) > 1:
                try:
                    self.failedge(list(self.graph.edges).index((newfail[0], newfail[1])))
                except ValueError:
                    self.failedge(list(self.graph.edges).index((newfail[1], newfail[0])))
            else:
                for node in self.nodes:
                    if newfail == str(node):
                        self.failnode(self.nodes.index(node))
                        break
        return True

    def failnode(self, nodenum=None, node=None):
        """Kill a random node."""
        if self.prevfail is not None:
            if isinstance(self.prevfail, tuple):
                self.graph.edges[self.prevfail[0]]['weight'] = self.prevfail[1]
            else:
                self.nodes[self.prevfail].heal()
        if node is not None:
            for nodeinst in self.nodes:
                if str(nodeinst) == node:
                    nodenum = self.nodes.index(nodeinst)
                    break
        if nodenum is None:
            nodenum = random.randint(0, len(self.nodes) - 1)
        self.nodes[nodenum].fail()
        self.prevfail = nodenum
        logging.info('Node {} disabled'.format(str(self.nodes[nodenum])))

    def failedge(self, edgenum=None, edge=None):
        """Kill a random edge."""
        if self.prevfail is not None:
            if isinstance(self.prevfail, tuple):
                self.graph.edges[self.prevfail[0]]['weight'] = self.prevfail[1]
            else:
                # noinspection PyTypeChecker
                self.nodes[self.prevfail].heal()
        if edge is not None:
            try:
                edgenum = list(self.graph.edges).index(tuple(edge))
            except ValueError:
                edgenum = list(self.graph.edges).index((edge[1], edge[0]))
        if edgenum is None:
            edgenum = random.randint(0, len(self.graph.edges) - 1)
        nodes = list(self.graph.edges)[edgenum]
        self.prevfail = (nodes, self.graph.edges[nodes]['weight'])
        self.graph.edges[nodes]['weight'] = 0
        logging.info('Edge {} disabled'.format(nodes[0] + nodes[1]))

    def filterinterresting(self):
        """Get all edges and nodes, which do something if there is no failure.
        Just these should fail, except src/dst."""
        self.interresting = []
        usededges = [element for element in self.path['None']
                     if element[1] + element[0] not in self.path['None'] or element[0] > element[1]]
        for node in self.nodes:
            if str(node) not in 'SD' and node.getsent():
                self.interresting.append(str(node))
        self.interresting.extend(usededges)
        self.calcres()
        self.checkconnection()      # Fails who break the graph are also not interresting
        logging.debug('Interresting failures are: {}'.format(self.interresting))

    def geteotx(self):
        """Calculate EOTX for all nodes in network and return as dict."""
        eotx = {str(node): float('inf') for node in self.nodes}
        eotx_t = {str(node): 1 for node in self.nodes}
        eotx_p = eotx_t.copy()
        eotx['D'] = 0.
        q = {node: eotxvalue for node, eotxvalue in eotx.items()}
        while q:
            node = min(q, key=q.get)  # Calculate from destination to source
            del q[node]
            for neighbor in self.graph.neighbors(node):
                if neighbor not in q:
                    continue
                eotx_t[neighbor] += self.graph.edges[node, neighbor]['weight'] * eotx_p[neighbor] * eotx[node]
                eotx_p[neighbor] *= (1 - self.graph.edges[node, neighbor]['weight'])
                eotx[neighbor] = eotx_t[neighbor] / (1 - eotx_p[neighbor])
                q[neighbor] = eotx[neighbor]
        return eotx

    def getgraph(self):
        """Return graph."""
        return self.graph

    def getidentifier(self):
        """Get current identifier for airtime dict."""
        identifier = 'None'
        if isinstance(self.prevfail, int):
            identifier = str(self.nodes[self.prevfail])
        elif isinstance(self.prevfail, tuple):
            identifier = self.prevfail[0][0] + self.prevfail[0][1]
        return identifier

    def getready(self, jsonfile=None, randcof=(10, 0.5)):
        """Do the basic stuff to get ready."""
        while jsonfile is None:
            try:
                self.createnetwork(config=readconf(jsonfile), randcof=randcof)
                self.calceotx()
                self.calc_tx_credit()
            except ZeroDivisionError:
                logging.info('Found graph with no connection between S and D')
                continue
            logging.info('Created random graph successfully!')
            break
        else:
            self.createnetwork(config=readconf(jsonfile), randcof=randcof)
            self.calceotx()
            self.calc_tx_credit()
            logging.info('Created network from JSON successfully!')
        self.mcut = nx.minimum_edge_cut(self.graph, s='S', t='D')
        self.dijkstra = nx.shortest_path(self.graph, source='S', target='D', weight='weight')
        if self.david:
            self.calcdeotx()

    def getpath(self):
        """Get path of the packet which arrived successfully."""
        return self.path

    def getneeded(self):
        """Get the amount of time needed for first batch."""
        return self.batchhist[0]

    def newbatch(self):
        """Spawn new batch if old one is done."""
        if not self.done:
            logging.info('Old batch is not done yet')
        self.batch += 1
        self.done = False
        self.donedict = {}
        prevfail = None
        if isinstance(self.prevfail, int):
            prevfail = str(self.nodes[self.prevfail])
        elif isinstance(self.prevfail, tuple):
            prevfail = self.prevfail[0]
        else:
            self.filterinterresting()
        if len(self.batchhist):
            self.failhist[prevfail] = (self.timestamp - self.batchhist[-1], self.batch - 1)
        else:
            self.failhist['None'] = (self.timestamp, self.batch - 1)
        for node in self.nodes:
            if str(node) in 'SD':
                node.newbatch()
        if self.allfail:
            if not self.failall():  # Just false if last failure was tested
                return True  # Return done
        elif self.nodefail is not None:
            self.failnode(node=self.nodefail)
        elif self.edgefail is not None:
            self.failedge(edge=self.edgefail)
        else:
            return True
        self.airtime[self.getidentifier()] = {}
        self.path[self.getidentifier()] = {}
        self.batchhist.append(self.timestamp)

    def sendall(self):
        """All nodes send at same time."""
        for node in self.nodes:
            if str(node) != 'D' and node.getcredit() > 0. and not node.getquiet() and node.gethealth():
                self.broadcast(node)
                node.reducecredit()
                ident = self.getidentifier()
                if str(node) in self.airtime[ident].keys():
                    self.airtime[ident][str(node)].append(self.timestamp)
                else:
                    self.airtime[ident][str(node)] = [self.timestamp]

    def sendsel(self):
        """Just the selected amount of nodes send at one timeslot."""
        goodnodes = [  # goodnodes are nodes which are allowed to send
            node for node in self.nodes if node.getcredit() > 0. and str(node) != 'D' and not node.getquiet() and
            node.gethealth()]
        maxsend = self.sendam if len(goodnodes) > self.sendam else len(goodnodes)
        for _ in range(maxsend):
            k = random.randint(0, len(goodnodes) - 1)
            if str(goodnodes[k]) != 'D':
                self.broadcast(goodnodes[k])
                goodnodes[k].reducecredit()
                ident = self.getidentifier()
                if str(goodnodes[k]) in self.airtime[ident].keys():
                    self.airtime[ident][str(goodnodes[k])].append(self.timestamp)
                else:
                    self.airtime[ident][str(goodnodes[k])] = [self.timestamp]
            del goodnodes[k]

    def update(self):
        """Update one timestep."""
        if not self.done:
            if self.sourceshift:
                self.checkstate()
            if self.sendam:
                self.sendsel()
            else:
                self.sendall()
            for node in self.nodes:
                node.rcvpacket(self.timestamp)
                if str(node) == 'D':
                    self.done = node.isdone()
                if node.isdone() and str(node) not in self.donedict and self.batch == node.getbatch():
                    logging.debug('Node {} done at timestep {}'.format(str(node), self.timestamp))
                    self.donedict[str(node)] = self.timestamp
                self.ranklist[str(node)].append(node.getrank())
            self.timestamp += 1
            if not self.checkduration():
                return True
            return self.done
        else:
            return True

    def writelogs(self):
        """Write everything down which could be usefull."""
        with open('{}/airtime.json'.format(self.folder), 'w') as file:
            json.dump(self.airtime, file)
        with open('{}/path.json'.format(self.folder), 'w') as file:
            json.dump(self.path, file)
        with open('{}/ranklist.json'.format(self.folder), 'w') as file:
            json.dump(self.ranklist, file)
        config = {'json': self.json, 'coding': self.coding, 'fieldsize': self.fieldsize, 'sendam': self.sendam,
                  'own': self.own, 'failedge': self.edgefail, 'failnode': self.nodefail, 'failall': self.allfail,
                  'randconf': self.randcof, 'folder': self.folder, 'maxduration': self.maxduration,
                  'randomseed': self.random, 'sourceshift': self.sourceshift, 'david': self.david,
                  'resilience': self.resilience, 'path': list(self.dijkstra), 'mcut': list(self.mcut)}
        with open('{}/config.json'.format(self.folder), 'w') as file:
            json.dump(config, file)
        for kind in ['overhearing', 'real']:
            trashdict = {}
            for node in self.nodes:
                if str(node) != 'S':
                    if kind == 'real':
                        trash, amount = node.getrealtrash(self.timestamp)
                    else:
                        trash, amount = node.gettrash(self.timestamp)
                    trashdict[str(node)] = (trash, amount)
            with open('{}/{}trash.json'.format(self.folder, kind), 'w') as file:
                json.dump(trashdict, file)
        with open('{}/failhist.json'.format(self.folder), 'w') as file:
            failhist = {}
            for fail, ts in self.failhist.items():
                if isinstance(fail, tuple):
                    failhist[fail[0] + fail[1]] = ts
                else:
                    failhist[fail] = ts
            json.dump(failhist, file)
        if isinstance(self.prevfail, tuple):        # Repair graph before writing it down
            self.graph.edges[self.prevfail[0]]['weight'] = self.prevfail[1]
        elif isinstance(self.prevfail, int):
            # noinspection PyTypeChecker
            self.nodes[self.prevfail].heal()
        information = {'nodes': {node: {'x': position[0], 'y': position[1]} for node, position in self.pos.items()},
                       'links': [{'nodes': edge, 'loss': self.graph.edges[edge]['weight']}
                                 for edge in list(self.graph.edges)]}
        with open('{}/graph.json'.format(self.folder), 'w') as file:
            json.dump(information, file, indent=4, sort_keys=True)
        nodedict = {}
        for node in self.nodes:
            if self.david:
                nodedict[str(node)] = (node.geteotx(), node.getdeotx(), node.getcreditinc())
            else:
                nodedict[str(node)] = (node.geteotx(), node.getcreditinc())
        with open('{}/eotx.json'.format(self.folder), 'w') as file:
            json.dump(nodedict, file)
        if self.own:
            with open('{}/AAOWN.OWN'.format(self.folder), 'w') as file:
                file.write('OWN')
        if self.sourceshift:
            with open('{}/AASS.SS'.format(self.folder), 'w') as file:
                file.write('SS')
        if self.david:
            with open('{}/AADAVID.DAVID'.format(self.folder), 'w') as file:
                file.write('DAVID')
