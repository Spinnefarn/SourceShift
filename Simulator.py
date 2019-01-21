#!/usr/bin/python
# coding=utf-8
"""Main script, we will see what this will do."""
import components
import json
import networkx as nx
import random
import os
import logging


def readconf(jsonfile):
    """Read configuration file."""
    if jsonfile is not None:
        if os.path.exists(jsonfile):
            with open(jsonfile) as file:
                config = json.loads(file.read())
            pos = {node: [config['nodes'][node]['x'], config['nodes'][node]['y']] for node in config['nodes']}
            return config['links'], pos
        else:
            raise FileNotFoundError


class Simulator:
    """Round based simulator to simulate traffic in meshed network."""
    def __init__(self, jsonfile=None, coding=None, fieldsize=2, sendall=0, own=True, edgefail=None, nodefail=None,
                 allfail=False, randcof=(10, 0.5), folder='.', maxduration=0, randomseed=None, sourceshift=False,
                 newshift=False, david=False, edgefailprob=0.1, hops=0, optimal=False, trash=False, anchor=False):
        self.airtime = {'None': {}}
        self.anchor = anchor
        self.sourceshift = sourceshift
        self.newshift = newshift
        self.optimal = optimal
        self.david = david
        self.edgefail = edgefail
        self.nodefail = nodefail
        self.allfail = allfail
        self.edgefailprob = edgefailprob
        logging.debug('Random seed: '.format(randomseed))
        self.random = randomseed
        self.maxduration = maxduration
        self.eotxdict = {}
        self.config = {}
        self.trash = trash      # Log trash?
        self.graph = None
        self.folder = folder
        self.prevfail = None
        self.failhist = {}
        self.interesting = []
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
        self.getready(jsonfile=jsonfile, randcof=randcof, hops=hops)
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
                                ident = self.getidentifier()
                                if str(node) + neighbor not in self.path[ident]:
                                    self.path[ident][(str(node) + neighbor)] = 1
                                else:
                                    self.path[ident][(str(node) + neighbor)] += 1
                            break

    def calcairtime(self):
        """Calculate the amount of used airtime in total."""
        summe = 0
        ident = self.getidentifier()
        for node in self.airtime[ident].keys():
            summe += len(self.airtime[ident][node])
        return summe

    def calcanchor(self):
        """Calculate ANCHORS metric."""
        for node in self.graph.nodes:
            if node == 'D':
                self.graph.nodes[node]['Priority'] = 1.
            else:
                self.graph.nodes[node]['Priority'] = 1/len(self.graph.nodes)
            self.graph.nodes[node]['ForwarderList'] = []
        prevdict, currdict = {node: self.graph.nodes[node]['Priority'] for node in self.graph.nodes}, {}
        while prevdict != currdict:
            if currdict:
                prevdict = currdict.copy()
            for node in self.graph.nodes:
                if node == 'D':
                    continue
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor]['Priority'] > self.graph.nodes[node]['Priority'] and \
                            neighbor not in self.graph.nodes[node]['ForwarderList'] and neighbor != 'S':
                        self.graph.nodes[node]['ForwarderList'].append(neighbor)
                        a = 1.
                        for n in self.graph.nodes[node]['ForwarderList']:
                            try:
                                a *= self.graph.edges[n, node]['weight']
                            except KeyError:
                                pass
                        a = 1 - a
                        r = 0.
                        for n in self.graph.nodes[node]['ForwarderList']:
                            if n != 'D':
                                try:
                                    c = self.graph.edges[n, node]['weight']
                                except KeyError:
                                    continue
                                for m in self.graph.nodes[node]['ForwarderList']:
                                    if m != 'D' and self.graph.nodes[n]['Priority'] < self.graph.nodes[m]['Priority']:
                                        try:
                                            c *= self.graph.edges[n, m]['weight']
                                        except KeyError:
                                            continue
                                c *= 1 / self.graph.nodes[node]['Priority']
                                r += c
                        b = 1 + r
                        self.graph.nodes[node]['Priority'] = a / b
                        self.graph.nodes[node]['codingRate'] = a
            currdict = {node: self.graph.nodes[node]['Priority'] for node in self.graph.nodes}

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
        credit = self.calc_tx_credit(david=True)
        for node in self.nodes:
            try:
                node.setcredit(credit[str(node)])
            except KeyError:
                pass
        self.graph.add_weighted_edges_from(bestlinks)

    def calce(self, node, notnode=None, david=False):
        """Calc the probability a packet is not received by any destination."""
        e = 1
        eotx = 'DEOTX' if david else 'EOTX'
        for neighbor in self.graph.neighbors(node):
            if self.graph.nodes[neighbor][eotx] < self.graph.nodes[node][eotx] and notnode != neighbor:
                e *= (1 - self.graph.edges[node, neighbor]['weight'])
        return e

    def calceotx(self, fail='None'):
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
        for node in self.graph.nodes:
            self.graph.nodes[node]['EOTX'] = eotx[str(node)]
        self.eotxdict[fail] = eotx

    def calc_tx_credit(self, david=False):
        """Calculate the amount of tx credit the receiver gets."""
        credit = {}
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
        for node in self.graph.nodes:
            if str(node) == 'D':
                continue
            somevalue = 0
            for neighbor in self.graph.neighbors(node):
                if david:
                    if self.graph.nodes[neighbor]['DEOTX'] > self.graph.nodes[node]['EOTX']:
                        somevalue += self.z[neighbor] * self.graph.edges[node, neighbor]['weight']
                else:
                    if self.graph.nodes[neighbor]['EOTX'] > self.graph.nodes[node]['EOTX']:
                        somevalue += self.z[neighbor] * self.graph.edges[node, neighbor]['weight']
            try:
                credit[node] = self.z[node] / somevalue
            except ZeroDivisionError:
                pass
        return credit

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

    # noinspection PyTypeChecker
    def checkconnection(self, graph=None):
        """Check for given list of potential failures if the graph is still connected."""
        lostlist = []
        if graph is None:
            graph = self.graph
            interesting = self.interesting
        else:
            interesting = list(graph.edges())
            interesting.extend(list(graph.nodes()))
            interesting.remove('S')
            interesting.remove('D')
        for fail in interesting:
            if len(fail) == 1:
                edgelist = [(fail, neighbor, graph.edges[fail, neighbor]['weight'])
                            for neighbor in graph.neighbors(fail)]
            else:
                edgelist = [(fail[0], fail[1], graph.edges[fail[0], fail[1]]['weight'])]
            graph.remove_edges_from(edgelist)
            try:
                nx.shortest_path(graph, source='S', target='D')
            except nx.exception.NetworkXNoPath:
                lostlist.append(fail)
            graph.add_weighted_edges_from(edgelist)
        if graph == self.graph:
            self.interesting = [fail for fail in self.interesting if fail not in lostlist]
            self.resilience[0] = (1 - self.edgefailprob) ** len(lostlist)
            logging.info('Resilience for graph is {}'.format(self.resilience[0]))
        else:
            self.resilience[1] = (1 - self.edgefailprob) ** len(lostlist)
            if self.david:
                logging.info('Resilience for MOREresilience is {}'.format(self.resilience[1]))
            elif self.sourceshift and self.own:
                logging.info('Resilience for MORELESS is {}'.format(self.resilience[1]))
            elif self.sourceshift:
                logging.info('Resilience for source shift is {}'.format(self.resilience[1]))
            elif self.newshift:
                logging.info('Resilience for new shift is {}'.format(self.resilience[1]))
            elif self.own:
                logging.info('Resilience for own approach is {}'.format(self.resilience[1]))
            else:
                logging.info('Resilience for MORE is {}'.format(self.resilience[1]))

    def checkduration(self):
        """Calculate duration since batch was started. To know there is no connection SD."""
        if len(self.batchhist) == 0:
            return True
        if self.maxduration:
            maxval = self.maxduration
        else:
            maxval = 100 * self.batchhist[0]
        if self.timestamp >= maxval:
            logging.info('Stopped batch after {} timesteps'.format(self.timestamp - self.batchhist[0]))
            return False
        else:
            return True

    def checkhops(self, hops):
        """Find a combination of source and destination in given network with the wished amount of hops."""
        path = nx.shortest_path(self.graph)
        for source, destdict in path.items():
            for destination, route in destdict.items():
                if len(route) - 1 == hops:
                    if source == 'D':
                        mapping = {destination: 'D', 'D': 'S', 'S': destination}
                        self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)
                    elif destination == 'S':
                        mapping = {'S': 'D', source: 'S', 'D': source}
                        self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)
                    else:
                        if source != 'S':
                            mapping = dict(zip(source + 'S', 'S' + source))
                            self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)
                        if destination != 'D':
                            mapping = dict(zip(destination + 'D', 'D' + destination))
                            self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)

                    return False
        return True     # Recreate graph as no path with wished length could be found

    def checkspecial(self, node, neighbor):
        """Return True if node should be able to send over special metric."""
        for invnei in self.graph.neighbors(neighbor):
            if self.graph.nodes[neighbor]['EOTX'] > self.graph.nodes[invnei]['EOTX']:   # Maybe a different way
                if invnei != str(node):     # Don't think your source is a different way
                    for invnode in self.nodes:
                        if str(invnode) == invnei:
                            if node.getbatch() >= invnode.getbatch() and not invnode.isdone():
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
            if ((self.sourceshift and node.isdone()) or (self.newshift and node.getrank() > 0)) and str(node) != 'S':
                for neighbor in self.graph.neighbors(str(node)):
                    if self.david:
                        david = self.graph.nodes[str(node)]['DEOTX'] > self.graph.nodes[neighbor]['DEOTX']
                    else:
                        david = False
                    if self.graph.nodes[str(node)]['EOTX'] > self.graph.nodes[neighbor]['EOTX'] or david:
                        for neighbornode in self.nodes:
                            if str(neighbornode) == neighbor:
                                if self.sourceshift:
                                    if not neighbornode.isdone() or neighbornode.getbatch() < node.getbatch():
                                        node.becomesource()
                                if self.newshift:
                                    if neighbornode.getrank() < node.getrank() \
                                            or neighbornode.getbatch() < node.getbatch():
                                        node.becomesource()
                                break

    def checkanchor(self):
        """Implementing feedback for anchor."""
        for node in self.nodes:
            if str(node) != 'S':
                for neighbor in self.graph.neighbors(str(node)):
                    if self.graph.nodes[str(node)]['Priority'] < self.graph.nodes[neighbor]['Priority']:
                        for neighbornode in self.nodes:
                            if neighbor == str(neighbornode):
                                if neighbornode.getrank() > node.getrank() and \
                                        neighbornode.getbatch() >= node.getbatch():
                                    node.rmcredit()
                                break

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
        self.ranklist = {node: [] for node in self.graph.nodes}

    def createusedgraph(self):
        """Calculate resilience of network based on used edges with no failure."""
        intedges = [element for element in self.interesting if len(element) == 2]
        edgelist = [(element[0], element[1], self.graph.edges[element[0], element[1]]['weight'])
                    for element in intedges]
        resg = nx.Graph()
        resg.add_weighted_edges_from(edgelist)
        return resg

    def failall(self):
        """Kill one of all."""
        if not len(self.interesting):
            logging.info('Nothing interesting?!')
            return False
        if self.prevfail is None:
            if len(self.interesting[0]) == 1:
                for node in self.nodes:
                    if str(node) == self.interesting[0]:
                        self.failnode(self.nodes.index(node))
                        break
            elif len(self.interesting[0]) == 2:
                try:
                    self.failedge(list(self.graph.edges).index((self.interesting[0][0], self.interesting[0][1])))
                except ValueError:
                    self.failedge(list(self.graph.edges).index((self.interesting[0][1], self.interesting[0][0])))
            else:
                logging.error('Something crazy in interesting list: {}'.format(self.interesting[0]))
        elif isinstance(self.prevfail, tuple):
            try:
                newidx = self.interesting.index(self.prevfail[0][0] + self.prevfail[0][1]) + 1
            except ValueError:
                newidx = self.interesting.index(self.prevfail[0][1] + self.prevfail[0][0]) + 1
            if len(self.interesting) > newidx:
                newfail = self.interesting[newidx]
                try:
                    self.failedge(list(self.graph.edges).index((newfail[0], newfail[1])))
                except ValueError:
                    self.failedge(list(self.graph.edges).index((newfail[1], newfail[0])))
            else:
                return False
        else:
            newfail = self.interesting[self.interesting.index(str(self.nodes[self.prevfail])) + 1]
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
        self.interesting = []
        usededges = [element for element in self.path['None']
                     if element[1] + element[0] not in self.path['None'] or element[0] > element[1]]
        for node in self.nodes:
            if str(node) not in 'SD' and node.getsent():
                self.interesting.append(str(node))
        self.interesting.extend(usededges)
        self.checkconnection(graph=self.createusedgraph())
        self.checkconnection()      # Fails who break the graph are also not interresting
        logging.debug('Interresting failures are: {}'.format(self.interesting))

    def geteotx(self):
        """Calculate EOTX for all nodes in network and return as dict."""
        eotx = {str(node): float('inf') for node in self.graph.nodes}
        eotx_t = {str(node): 1 for node in self.graph.nodes}
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

    def getready(self, jsonfile=None, randcof=(10, 0.5), hops=0):
        """Do the basic stuff to get ready."""
        while jsonfile is None:
            try:
                self.createnetwork(config=readconf(jsonfile), randcof=randcof)
                if hops and self.checkhops(hops):
                    continue
                self.calceotx()
                credit = self.calc_tx_credit()
            except ZeroDivisionError:
                logging.info('Found graph with no connection between S and D')
                continue
            except nx.exception.NetworkXException:
                logging.info(str(self.graph.nodes))
                logging.info(str(self.graph.edges))
                continue
            logging.info('Created random graph successfully!')
            break
        else:
            self.createnetwork(config=readconf(jsonfile), randcof=randcof)
            self.calceotx()
            credit = self.calc_tx_credit()
            logging.info('Created network from JSON successfully!')
        self.nodes = [components.Node(name=name, coding=self.coding, fieldsize=self.fieldsize, random=self.random,
                                      trash=self.trash) for name in self.graph.nodes]
        for node in self.nodes:
            try:
                node.seteotx(self.graph.nodes[str(node)]['EOTX'])
                if not self.newshift:
                    node.setcredit(credit[str(node)])
            except KeyError:
                pass
        self.eotxdict['None'] = self.geteotx()
        if self.anchor:
            self.calcanchor()
            for node in self.nodes:
                try:
                    node.setpriority(self.graph.nodes[str(node)]['Priority'])
                    node.setcredit(self.graph.nodes[str(node)]['codingRate'])
                except KeyError:
                    pass
            self.eotxdict['None'] = {node: self.graph.nodes[node]['Priority'] for node in self.graph.nodes}
        self.mcut = nx.minimum_edge_cut(self.graph, s='S', t='D')
        self.dijkstra = nx.shortest_path(self.graph, source='S', target='D', weight='weight')
        if self.david:
            try:
                self.calcdeotx()
            except ZeroDivisionError:
                pass

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
        if isinstance(prevfail, tuple):
            self.failhist[prevfail[0] + prevfail[1]] = (self.timestamp, self.batch - 1)
        else:
            if prevfail is None:
                self.failhist['None'] = (self.timestamp, self.batch - 1)
            else:
                self.failhist[prevfail] = (self.timestamp, self.batch - 1)
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
        if self.optimal:
            self.recalc()
        self.airtime[self.getidentifier()] = {}
        self.path[self.getidentifier()] = {}
        self.batchhist.append(self.timestamp)
        self.timestamp = 0

    def recalc(self):
        """Recalc metric for current failure."""
        if isinstance(self.prevfail, int):
            fail = str(self.nodes[self.prevfail])
        elif isinstance(self.prevfail, tuple):
            fail = self.prevfail[0][0] + self.prevfail[0][1]
        else:
            return
        if len(fail) == 1:
            edgelist = [(fail, neighbor, self.graph.edges[fail, neighbor]['weight'])
                        for neighbor in self.graph.neighbors(fail)]
        else:
            edgelist = [(fail[0], fail[1], self.graph.edges[fail[0], fail[1]]['weight'])]
        self.graph.remove_edges_from(edgelist)
        credit = {}
        try:
            self.calceotx(fail=fail)
            credit = self.calc_tx_credit()
        except ZeroDivisionError:
            logging.error('Failed to recalc EOTX or credit for {}'.format(edgelist))
            logging.error(str(self.graph.nodes))
            logging.error(str(self.graph.edges))
        for node in self.nodes:
            try:
                node.resetcredit()
                node.seteotx(self.graph.nodes[str(node)]['EOTX'])
                node.setcredit(credit[str(node)])
            except KeyError:
                pass
        self.graph.add_weighted_edges_from(edgelist)

    def sendall(self):
        """All nodes send at same time."""
        for node in self.nodes:
            if str(node) != 'D' and node.getcredit() > 0. and not node.getquiet() and node.gethealth():
                if node.getbatch() == self.batch:
                    self.broadcast(node)
                    node.reducecredit()
                    ident = self.getidentifier()
                    if str(node) in self.airtime[ident].keys():
                        self.airtime[ident][str(node)] += 1
                    else:
                        self.airtime[ident][str(node)] = 1

    def sendsel(self):
        """Just the selected amount of nodes send at one timeslot."""
        goodnodes = [  # goodnodes are nodes which are allowed to send
            node for node in self.nodes if node.getcredit() > 0. and str(node) != 'D' and not node.getquiet() and
            node.gethealth() and node.getbatch() == self.batch]
        maxsend = self.sendam if len(goodnodes) > self.sendam else len(goodnodes)
        for _ in range(maxsend):
            k = random.randint(0, len(goodnodes) - 1)
            if str(goodnodes[k]) != 'D':
                self.broadcast(goodnodes[k])
                goodnodes[k].reducecredit()
                ident = self.getidentifier()
                if str(goodnodes[k]) in self.airtime[ident].keys():
                    self.airtime[ident][str(goodnodes[k])] += 1
                else:
                    self.airtime[ident][str(goodnodes[k])] = 1
            del goodnodes[k]

    def update(self):
        """Update one timestep."""
        if not self.done:
            if self.sourceshift or self.newshift:
                self.checkstate()
            if self.anchor:
                self.checkanchor()
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
                self.ranklist[str(node)].append(node.getrank())       # Just for debugging
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
        # with open('{}/ranklist.json'.format(self.folder), 'w') as file:
        #    json.dump(self.ranklist, file)     # Just for debugging
        config = {'json': self.json, 'coding': self.coding, 'fieldsize': self.fieldsize, 'sendam': self.sendam,
                  'own': self.own, 'failedge': self.edgefail, 'failnode': self.nodefail, 'failall': self.allfail,
                  'randconf': self.randcof, 'folder': self.folder, 'maxduration': self.maxduration,
                  'randomseed': self.random, 'sourceshift': self.sourceshift, 'newshift': self.newshift,
                  'david': self.david, 'resilience': self.resilience, 'path': list(self.dijkstra),
                  'mcut': list(self.mcut), 'optimal': self.optimal}
        with open('{}/config.json'.format(self.folder), 'w') as file:
            json.dump(config, file, indent=4)
        if self.trash:
            for kind in ['overhearing', 'real']:
                trashdict = {}
                for node in self.nodes:
                    if str(node) != 'S':
                        if kind == 'real':
                            trash = node.getrealtrash(self.timestamp)
                        else:
                            trash = node.gettrash(self.timestamp)
                        trashdict[str(node)] = trash
                with open('{}/{}trash.json'.format(self.folder, kind), 'w') as file:
                    json.dump(trashdict, file)
        with open('{}/failhist.json'.format(self.folder), 'w') as file:
            json.dump(self.failhist, file)
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
        if self.own:
            with open('{}/AAOWN.OWN'.format(self.folder), 'w') as file:
                file.write('OWN')
        if self.sourceshift:
            with open('{}/AASS.SS'.format(self.folder), 'w') as file:
                file.write('SS')
        if self.newshift:
            with open('{}/AANS.SS'.format(self.folder), 'w') as file:
                file.write('NS')
        if self.optimal:
            with open('{}/AAOPT.SS'.format(self.folder), 'w') as file:
                file.write('OPT')
        if self.david:
            with open('{}/AADAVID.DAVID'.format(self.folder), 'w') as file:
                file.write('DAVID')
            daveotx = {str(node): node.geteotx() for node in self.nodes}
            self.eotxdict['david'] = daveotx
        with open('{}/eotx.json'.format(self.folder), 'w') as file:
            json.dump(self.eotxdict, file)
