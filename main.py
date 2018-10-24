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
    parser.add_argument('-j', '--json',
                        dest='json',
                        type=str,
                        help='should contain network configuration',
                        default='graph.json')
    parser.add_argument('-c', '--coding',
                        dest='coding',
                        type=int,
                        help='Batch size for coded packets. Default None(without coding)',
                        default=8)
    parser.add_argument('-f', '--fieldsize',
                        dest='fieldsize',
                        type=int,
                        help='Fieldsize used for coding.',
                        default=2)
    parser.add_argument('-sa', '--sendamount',
                        dest='sendam',
                        type=int,
                        help='Amount of nodes allowed to send per timeslot.',
                        default=0)
    parser.add_argument('-o', '-o-wn',
                        dest='own',
                        type=bool,
                        help='Use own approach or MORE.',
                        default=True)
    return parser.parse_args()


def readconf(jsonfile):
    """Read configuration file."""
    with open(jsonfile) as f:
        config = json.loads(f.read())
    pos = {node: [config['nodes'][node]['x'], config['nodes'][node]['y']] for node in config['nodes']}
    return config['links'], pos


class Simulator:
    """Round based simulator to simulate traffic in meshed network."""

    def __init__(self, loglevel=logging.INFO, jsonfile=None, coding=None, fieldsize=2, sendall=0, own=True):
        self.airtime = {}
        self.config = {}
        self.graph = None
        self.nodes = []
        self.packets = []
        self.newpackets = []
        self.z = {}
        self.batch = 0
        self.sendam = sendall
        self.coding = coding
        self.fieldsize = fieldsize
        self.pos = None
        import logging
        logging.basicConfig(filename='simulator.log', level=loglevel, format='%(asctime)s %(message)s', filemode='w')
        self.getready(jsonfile)
        self.done = False
        self.path = {}
        self.timestamp = 0
        self.own = own

    def broadcast(self, node):
        """Broadcast given packet to all neighbors."""
        for neighbor in list(self.graph.neighbors(str(node))):
            if self.graph.edges[str(node), neighbor]['weight'] > random.random():   # roll dice
                if neighbor != 'S':    # Source will not receive a packet, but still written down
                    for name in self.nodes:         # Add received Packet to buffer with coding
                        if str(name) == neighbor:
                            special = False
                            for invnei in self.graph.neighbors(neighbor):
                                if self.graph.nodes[neighbor]['EOTX'] > self.graph.nodes[invnei]['EOTX']:
                                    if invnei != str(node) and self.own:
                                        special = True
                                        break
                            name.buffpacket(batch=node.getbatch(), coding=node.getcoded(), preveotx=node.geteotx(),
                                            special=special)
                            break
                if node.getbatch() not in self.path:
                    self.path[node.getbatch()] = {}
                if (str(node), neighbor) not in self.path[node.getbatch()]:
                    self.path[node.getbatch()][(str(node), neighbor)] = []
                self.path[node.getbatch()][(str(node), neighbor)].append(self.timestamp)

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
            node, value = min(q.items(), key=lambda x: x[1].geteotx())    # Calculate from destination to source
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
        l = {key[0]: 0 for key in l_n if key[0] != 'D'}
        l['S'] = 1
        for idx, node in enumerate(list(l.keys())[::-1]):
            self.z[node] = l[node] / (1 - self.calce(node))
            p = 1
            for idx2, j in enumerate(list(l.keys())[:(len(l.keys()) - idx - 1)]):
                try:
                    if idx2 == 0:
                        p *= (1 - self.graph.edges[node, 'D']['weight'])
                    else:
                        p *= (1 - self.graph.edges[node, list(l.keys())[idx2 - 1]]['weight'])
                except KeyError:
                    pass
                try:
                    l[j] += self.z[node] * p * self.graph.edges[node, j]['weight']
                except KeyError:
                    pass
        for node in self.nodes:
            if str(node) == 'D':
                continue
            value = 0
            for neighbor in self.graph.neighbors(str(node)):
                if self.graph.nodes[neighbor]['EOTX'] > node.geteotx():
                    value += self.z[neighbor] * self.graph.edges[str(node), neighbor]['weight']
            try:
                node.setcredit(self.z[str(node)]/value)
            except ZeroDivisionError:
                pass

    def createnetwork(self, config):
        """Create network using networkx library based on configuration given as dict."""
        self.pos = config[1]
        configlist = []
        for edge in config[0]:
            configlist.append((edge['nodes'][0], edge['nodes'][1], edge['loss']))
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from(configlist)
        self.nodes = [components.Node(name=name, coding=self.coding, fieldsize=self.fieldsize)
                      for name in self.graph.nodes]
        for node in self.nodes:
            self.airtime[str(node)] = []

    def drawunused(self):
        """Draw initial graph."""
        plt.figure(figsize=(10, 10))
        nx.draw(self.graph, pos=self.pos, with_labels=True, node_size=1500, node_color="skyblue", node_shape="o",
                alpha=0.7, linewidths=4, font_size=25, font_color="red", font_weight="bold", width=2,
                edge_color="grey")
        labels = {key: round(value, 1) for key, value in nx.get_node_attributes(self.graph, 'EOTX').items()}
        nx.draw_networkx_labels(self.graph, pos=self.pos, labels=labels)
        nx.draw_networkx_edge_labels(self.graph, pos=self.pos, edge_labels=nx.get_edge_attributes(self.graph, 'weight'))
        plt.savefig('graph.pdf')

    def drawused(self):
        """Highlight paths used in graph drawn in getready()."""
        self.drawunused()
        edgelist = []
        for batch in self.path:
            edgelist.extend(list(self.path[batch].keys()))
            for edge in self.path[batch].keys():
                edgelist.extend([edge for _ in self.path[batch][edge]])
        nx.draw_networkx_edges(self.graph, pos=self.pos, edgelist=edgelist, width=8, alpha=0.1,
                               edge_color='purple')
        plt.savefig('usedgraph.pdf')
        plt.close()

    def drawtrash(self):
        """Draw linear dependent packets over time and nodes"""
        maxval = []
        plt.figure(figsize=(10, 5))
        trashdict = {}
        amts = {ts: 0 for ts in range(self.timestamp)}
        cmap, i = plt.get_cmap('tab20'), 0
        for node in self.nodes:
            if str(node) != 'S':
                trash, amount = node.gettrash(self.timestamp)
                trashdict[str(node)] = (trash, amount)
                plt.bar(trash, amount, bottom=list(amts.values()), label=str(node), color=cmap(i), alpha=0.5)
                for key, number in zip(trash, amount):
                    if key in amts:
                        amts[key] += number
                    else:
                        amts[key] = number
                # plt.scatter(trash, amount, marker='x', label=str(node), alpha=0.5)
                if len(amount):
                    maxval.append(max(amount))
            i += 1
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('Amount of linear depended packets')
        plt.xlabel('Timestamp')
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.yticks(range(1, max(amts.values()) + 1, 1))
        plt.title('Amount of linear dependent packets for each node.')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('lineardependent.pdf')
        plt.close()

    def getgraph(self):
        """Return graph."""
        return self.graph

    def getready(self, jsonfile):
        """Do the basic stuff to get ready."""
        self.createnetwork(readconf(jsonfile))
        self.calceotx()
        self.calc_tx_credit()
        self.pos = nx.spring_layout(self.graph).copy()
        self.drawunused()
        plt.close()
        logging.info('Created network from JSON successfully!')

    def getpath(self):
        """Get path of the packet which arrived successfully."""
        return self.path

    def newbatch(self):
        """Spawn new batch if old one is done."""
        if not self.done:
            logging.warning('Old batch is not done yet')
        self.batch += 1
        self.done = False
        for node in self.nodes:
            if str(node) == 'D':
                node.newbatch()

    def sendall(self):
        """All nodes send at same time."""
        for node in self.nodes:
            if str(node) == 'S':
                self.broadcast(node)
                self.airtime[str(node)].append(self.timestamp)
            elif str(node) != 'D' and node.getcredit() > 0:
                self.broadcast(node)
                node.reducecredit()
                self.airtime[str(node)].append(self.timestamp)

    def sendsel(self):
        """Just the selected amount of nodes send at one timeslot."""
        goodnodes = [node for node in self.nodes if (node.getcredit() > 0) or (str(node) == 'S')]
        maxsend = self.sendam if len(goodnodes) > self.sendam else len(goodnodes)
        for _ in range(maxsend):
            k = random.randint(0, len(goodnodes) - 1)
            if str(goodnodes[k]) == 'S':
                self.broadcast(goodnodes[k])
                self.airtime[str(goodnodes[k])].append(self.timestamp)
            elif str(goodnodes[k]) != 'D':
                self.broadcast(goodnodes[k])
                goodnodes[k].reducecredit()
                self.airtime[str(goodnodes[k])].append(self.timestamp)
            del goodnodes[k]

    def update(self):
        """Update one timestep."""
        if not self.done:
            if self.sendam:
                self.sendsel()
            else:
                self.sendall()
            for node in self.nodes:
                if str(node) != 'S':
                    node.rcvpacket(self.timestamp)
                if str(node) == 'D':
                    self.done = node.isdone()
            self.timestamp += 1
            return self.done
        else:
            return True


if __name__ == '__main__':
    llevel = logging.INFO
    logging.basicConfig(
        filename='main.log', level=llevel, format='%(asctime)s %(levelname)s\t %(message)s', filemode='w')
    args = parse_args()
    sim = Simulator(loglevel=llevel, jsonfile=args.json, coding=args.coding, fieldsize=args.fieldsize,
                    sendall=args.sendam, own=args.own)
    for i in range(20):
        done = False
        while not done:
            done = sim.update()
        sim.newbatch()
    sim.drawused()
    sim.drawtrash()
    logging.info('Packet arrived at destination. {}'.format(sim.getpath()))
    logging.info('Total used airtime {}'.format(sim.calcairtime()))
