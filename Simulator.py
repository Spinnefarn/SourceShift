#!/usr/bin/python
# coding=utf-8
"""Main script, we will see what this will do."""
import argparse
import components
import json
import networkx as nx
import matplotlib.pylab as plt
import random
import time
import os
import logging


def parse_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json',
                        dest='json',
                        type=str,
                        help='should contain network configuration',
                        default=None)
    parser.add_argument('-n', '--randomnodes',
                        dest='amount',
                        type=tuple,
                        nargs=2,
                        help='In case of no json given how many random nodes should be created and max dist for links.',
                        default=(10, 0.1))
    parser.add_argument('-c', '--coding',
                        dest='coding',
                        type=int,
                        help='Batch size for coded packets. Default None(without coding)',
                        default=8)
    parser.add_argument('-f', '--fieldsize',
                        dest='fieldsize',
                        type=int,
                        help='Fieldsize used for coding. 2 to the power of x, x∈(1, 4, 8, 16)',
                        default=1)
    parser.add_argument('-sa', '--sendamount',
                        dest='sendam',
                        type=int,
                        help='Amount of nodes allowed to send per timeslot.',
                        default=0)
    parser.add_argument('-o', '--own',
                        dest='own',
                        type=bool,
                        help='Use own approach or MORE.',
                        default=False)
    parser.add_argument('-fe', '--failedge',
                        dest='failedge',
                        type=bool,
                        help='Shut a random edge fail for higher batches.',
                        default=False)
    parser.add_argument('-fn', '--failnode',
                        dest='failnode',
                        type=bool,
                        help='Should a random node fail for higher batches.',
                        default=False)
    parser.add_argument('-fa', '--failall',
                        dest='failall',
                        help='Everything should fail(just one by time.',
                        default=False)
    parser.add_argument('-F', '--folder',
                        dest='folder',
                        type=str,
                        help='Folder where results should be placed in.',
                        default='.')
    parser.add_argument('-max', '--maxduration',
                        dest='maxduration',
                        type=int,
                        help='Maximum number time slots to wait until destination finishes.',
                        default=0)
    return parser.parse_args()


def readconf(jsonfile):
    """Read configuration file."""
    if jsonfile is not None and os.path.exists(jsonfile):
        with open(jsonfile) as file:
            config = json.loads(file.read())
        pos = {node: [config['nodes'][node]['x'], config['nodes'][node]['y']] for node in config['nodes']}
        return config['links'], pos


class Simulator:
    """Round based simulator to simulate traffic in meshed network."""
    def __init__(self, jsonfile=None, coding=None, fieldsize=2, sendall=0, own=True, edgefail=False, nodefail=False,
                 allfail=False, randcof=(10, 0.5), folder='.', maxduration=0, randomseed=None):
        self.airtime = {}
        self.edgefail = edgefail
        self.nodefail = nodefail
        self.allfail = allfail
        self.random = randomseed
        self.maxduration = maxduration
        self.config = {}
        self.graph = None
        self.folder = folder
        self.prevfail = None
        self.failhist = {}
        self.nodes = []
        self.ranklist = {}
        self.z = {}
        self.batch = 0
        self.sendam = sendall
        self.batchhist = []
        self.coding = coding
        self.fieldsize = fieldsize
        self.pos = None
        self.getready(jsonfile=jsonfile, randcof=randcof)
        self.done = False
        self.path = {0: {}}
        self.timestamp = 0
        self.own = own
        self.donedict = {}

    def broadcast(self, node):
        """Broadcast given packet to all neighbors."""
        packet = node.getcoded()
        for neighbor in list(self.graph.neighbors(str(node))):
            if self.graph.edges[str(node), neighbor]['weight'] > random.random():  # roll dice
                if neighbor != 'S':  # Source will not receive a packet, but still written down
                    for name in self.nodes:  # Add received Packet to buffer with coding
                        if str(name) == neighbor:
                            if name.gethealth():    # Broken nodes should not receive
                                special = self.checkspecial(node, neighbor) if self.own else False
                                name.buffpacket(batch=node.getbatch(), coding=packet, preveotx=node.geteotx(),
                                                special=special, ts=self.timestamp)
                                if (str(node), neighbor) not in self.path[node.getbatch()]:
                                    self.path[node.getbatch()][(str(node), neighbor)] = []
                                self.path[node.getbatch()][(str(node), neighbor)].append(self.timestamp)
                            break

    def calcairtime(self):
        """Calculate the amount of used airtime in total."""
        summe = 0
        for node in self.airtime.keys():
            summe += len(self.airtime[node])
        return summe

    def calce(self, node):
        """Calc the probability a packet is not received by any destination."""
        e = 1
        for neighbor in self.graph.neighbors(node):
            if self.graph.nodes[neighbor]['EOTX'] < self.graph.nodes[node]['EOTX']:
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

        eotx_t = {str(node): 1 for node in self.nodes}
        eotx_p = eotx_t.copy()
        q = {}
        for node in self.nodes:
            self.graph.nodes[str(node)]['EOTX'] = float('inf')
            q[str(node)] = node
        q['D'].seteotx(0.)
        self.graph.nodes['D']['EOTX'] = 0.
        while q:
            node, _ = min(q.items(), key=lambda x: x[1].geteotx())  # Calculate from destination to source
            del q[node]
            for neighbor in self.graph.neighbors(node):
                if neighbor not in q:
                    continue
                eotx_t[neighbor] += \
                    self.graph.edges[node, neighbor]['weight'] * eotx_p[neighbor] * self.graph.nodes[node]['EOTX']
                eotx_p[neighbor] *= (1 - self.graph.edges[node, neighbor]['weight'])
                self.graph.nodes[neighbor]['EOTX'] = eotx_t[neighbor] / (1 - eotx_p[neighbor])
                q[neighbor].seteotx(self.graph.nodes[neighbor]['EOTX'])

    def calc_tx_credit(self):
        """Calculate the amount of tx credit the receiver gets."""
        l_n = {node: self.graph.nodes[node]['EOTX'] for node in self.graph.nodes}
        l_n = sorted(l_n.items(), key=lambda kv: kv[1])
        l_i = {nodename[0]: 0 for nodename in l_n if nodename[0] != 'D'}
        l_i['S'] = 1
        for idx, node in enumerate(list(l_i.keys())[::-1]):
            self.z[node] = l_i[node] / (1 - self.calce(node))
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
                if self.graph.nodes[neighbor]['EOTX'] > node.geteotx():
                    somevalue += self.z[neighbor] * self.graph.edges[str(node), neighbor]['weight']
            try:
                node.setcredit(self.z[str(node)] / somevalue)
            except ZeroDivisionError:
                pass

    def checkduration(self):
        """Calculate duration since batch was started. To know there is no connection SD."""
        if len(self.batchhist) == 0:
            return True
        if self.maxduration:
            maxval = self.maxduration
        else:
            maxval = 20 * self.batchhist[0]
        if len(self.batchhist) == 1:
            if self.timestamp - self.batchhist[0] > maxval:
                logging.warning('Stopped batch after {} timesteps'.format(self.timestamp - self.batchhist[0]))
                return False
            else:
                return True
        else:
            if self.timestamp - self.batchhist[-1] > maxval:
                logging.warning('Stopped batch after {} timesteps'.format(self.timestamp - self.batchhist[-1]))
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
                            if not invnode.isdone():     # Don't get special if dst is done
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
                for neighbor in self.graph.neighbors(str(node)):
                    if self.checkspecial(node, neighbor):
                        for neighbornode in self.nodes:
                            if str(neighbornode) == neighbor:
                                neighbornode.becomesource()
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
            information = {'nodes': {node: {'x': position[0], 'y': position[1]} for node, position in self.pos.items()},
                           'links': [{'nodes': edge, 'loss': self.graph.edges[edge]['weight']}
                                     for edge in list(self.graph.edges)]}
            with open('{}/usedgraph.json'.format(self.folder), 'w') as file:
                json.dump(information, file, indent=4, sort_keys=True)
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
            self.airtime[str(node)] = []
            self.ranklist[str(node)] = []

    def drawfailes(self, failhist=None):
        """Draw batch duration over failed nodes/edges."""
        plt.figure(figsize=(10, 10))
        if failhist is not None:
            plt.bar(range(len(failhist)), list(failhist.values()), label='MORE')
            plt.bar(range(len(self.failhist)), list(self.failhist.values()), bottom=list(failhist.values()),
                    label='MORELESS')
        else:
            plt.bar(range(len(self.failhist)), list(self.failhist.values()), label='MORE')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('Needed airtime in timeslots')
        plt.xlabel('Failure')
        plt.title('Needed airtime over failures for different protocols')
        plt.xticks(range(len(self.failhist)), labels=self.failhist.keys(), rotation=90)
        plt.tight_layout()
        plt.savefig('{}/airtimefail.pdf'.format(self.folder))
        plt.close()
        return self.failhist

    def drawunused(self):
        """Draw initial graph."""
        plt.figure(figsize=(10, 10))
        nx.draw(self.graph, pos=self.pos, with_labels=True, node_size=1500, node_color="skyblue", node_shape="o",
                alpha=0.7, linewidths=4, font_size=25, font_color="red", font_weight="bold", width=2,
                edge_color="grey")
        labels = {x: round(y, 1) for x, y in nx.get_node_attributes(self.graph, 'EOTX').items()}
        nx.draw_networkx_labels(self.graph, pos=self.pos, labels=labels)
        nx.draw_networkx_edge_labels(self.graph, pos=self.pos, edge_labels=nx.get_edge_attributes(self.graph, 'weight'))
        plt.savefig('{}/graph.pdf'.format(self.folder))

    def drawused(self):
        """Highlight paths used in graph drawn in getready()."""
        self.drawunused()
        edgelist = []
        currgen = list(self.path.keys())[-1]
        for edge in self.path[currgen].keys():
            edgelist.extend([edge for _ in self.path[currgen][edge]])
        nx.draw_networkx_edges(self.graph, pos=self.pos, edgelist=edgelist, width=8, alpha=0.1,
                               edge_color='purple')
        if isinstance(self.prevfail, tuple):
            fail = self.prevfail[0]
        elif self.prevfail is None:
            fail = self.prevfail
        else:
            fail = str(self.nodes[self.prevfail])
        if self.own:
            plt.savefig('{}/usedgraphownfail{}.pdf'.format(self.folder, fail))
        else:
            plt.savefig('{}/usedgraphfail{}.pdf'.format(self.folder, fail))
        plt.close()

    def drawtrash(self, kind=None):
        """Draw linear dependent packets over time and nodes"""
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
            self.failnode(0)
        elif isinstance(self.prevfail, tuple):
            liste = list(self.graph.edges)
            idx = liste.index(self.prevfail[0])
            if idx + 1 < len(liste):
                self.failedge(idx + 1)
            else:
                return False
        else:
            if self.prevfail + 1 < len(self.nodes):
                self.failnode(self.prevfail + 1)
            else:
                self.failedge(0)
        return True

    def failnode(self, nodenum=None):
        """Kill a random node."""
        if self.prevfail is not None:
            if isinstance(self.prevfail, tuple):
                self.graph.edges[self.prevfail[0]]['weight'] = self.prevfail[1]
            else:
                self.nodes[self.prevfail].heal()
        if nodenum is None:
            nodenum = random.randint(0, len(self.nodes) - 1)
        self.nodes[nodenum].fail()
        self.prevfail = nodenum
        logging.info('Node {} disabled'.format(str(self.nodes[nodenum])))

    def failedge(self, edgenum=None):
        """Kill a random edge."""
        if self.prevfail is not None:
            if isinstance(self.prevfail, tuple):
                self.graph.edges[self.prevfail[0]]['weight'] = self.prevfail[1]
            else:
                # noinspection PyTypeChecker
                self.nodes[self.prevfail].heal()
        if edgenum is None:
            edgenum = random.randint(0, len(self.graph.edges) - 1)
        nodes = list(self.graph.edges)[edgenum]
        self.prevfail = (nodes, self.graph.edges[nodes]['weight'])
        self.graph.edges[nodes]['weight'] = 0
        logging.info('Edge {} disabled'.format(str(nodes)))

    def getgraph(self):
        """Return graph."""
        return self.graph

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
        self.drawunused()
        plt.close()

    def getpath(self):
        """Get path of the packet which arrived successfully."""
        return self.path

    def getneeded(self):
        """Get the amount of time needed for first batch."""
        return self.batchhist[0]

    def newbatch(self):
        """Spawn new batch if old one is done."""
        if not self.done:
            logging.warning('Old batch is not done yet')
        self.batch += 1
        self.done = False
        self.donedict = {}
        prevfail = None
        self.path[self.batch] = {}
        if isinstance(self.prevfail, int):
            prevfail = str(self.nodes[self.prevfail])
        elif isinstance(self.prevfail, tuple):
            prevfail = self.prevfail[0]
        if len(self.batchhist):
            self.failhist[prevfail] = self.timestamp - self.batchhist[-1]
        else:
            self.failhist['None'] = self.timestamp
        for node in self.nodes:
            if str(node) in 'SD':
                node.newbatch()
        if self.allfail:
            if not self.failall():  # Just false if last failure was tested
                return True  # Return done
        elif self.nodefail:
            self.failnode()
        elif self.edgefail:
            self.failedge()
        else:
            return True
        self.batchhist.append(self.timestamp)

    def sendall(self):
        """All nodes send at same time."""
        for node in self.nodes:
            if str(node) != 'D' and node.getcredit() > 0 and not node.getquiet() and node.gethealth():
                self.broadcast(node)
                node.reducecredit()
                self.airtime[str(node)].append(self.timestamp)

    def sendsel(self):
        """Just the selected amount of nodes send at one timeslot."""
        goodnodes = [  # Goodnode are nodes which are allowed to send
            node for node in self.nodes if ((node.getcredit() > 0) or (str(node) == 'S')) and not node.getquiet() and
            node.gethealth()]
        maxsend = self.sendam if len(goodnodes) > self.sendam else len(goodnodes)
        for _ in range(maxsend):
            k = random.randint(0, len(goodnodes) - 1)
            if str(goodnodes[k]) != 'D':
                self.broadcast(goodnodes[k])
                goodnodes[k].reducecredit()
                self.airtime[str(goodnodes[k])].append(self.timestamp)
            del goodnodes[k]

    def update(self):
        """Update one timestep."""
        if not self.done:
            if self.own:
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
                    logging.info('Node {} done at timestep {}'.format(str(node), self.timestamp))
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
            path = {}
            for generation in self.path:
                path[generation] = {}
                for link in self.path[generation]:
                    path[generation][link[0] + link[1]] = self.path[generation][link]
            json.dump(path, file)
        with open('{}/ranklist.json'.format(self.folder), 'w') as file:
            json.dump(self.ranklist, file)
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
            for key, value in self.failhist.items():
                if isinstance(key, tuple):
                    failhist[key[0] + key[1]] = value
                else:
                    failhist[key] = value
            json.dump(failhist, file)


if __name__ == '__main__':
    random.seed(1)
    llevel = logging.INFO
    logging.basicConfig(
        filename='main.log', level=llevel, format='%(asctime)s %(levelname)s\t %(message)s', filemode='w')
    args = parse_args()
    sim = Simulator(jsonfile=args.json, coding=args.coding, fieldsize=args.fieldsize, sendall=args.sendam, own=args.own,
                    edgefail=args.failedge, nodefail=args.failnode, allfail=args.failall, randcof=args.amount,
                    folder=args.folder)
    starttime = time.time()
    complete = False
    while not complete:
        beginbatch = time.time()
        done = False
        while not done:
            done = sim.update()
        logging.info('{:3.0f} Seconds needed'.format(time.time() - beginbatch))
        sim.drawused()
        complete = sim.newbatch()
    logging.info('{:3.0f} Seconds needed in total.'.format(time.time() - starttime))
    # sim.drawtrash()
    # sim.drawtrash('real')
    sim.drawfailes()
    sim.writelogs()
    with open('path.json', 'w') as f:
        newdata = {}
        for batch in sim.getpath():
            newdata[batch] = {}
            for key, value in sim.getpath()[batch].items():
                newdata[batch][str(key)] = value
        json.dump(newdata, f)
    logging.info('Total used airtime {}'.format(sim.calcairtime()))
