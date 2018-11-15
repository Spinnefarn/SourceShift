#!/usr/bin/python
# coding=utf-8
"""Will collect interesting logs from subfolders and draw plots."""
import matplotlib.pylab as p
import networkx as nx
import json
import os
import datetime
import statistics


def drawunused(net=None, pos=None):
    """Draw unused graph."""
    if net is None or pos is None:
        return
    nx.draw(net, pos=pos, with_labels=True, node_size=1500, node_color="skyblue", node_shape="o",
            alpha=0.7, linewidths=4, font_size=25, font_color="red", font_weight="bold", width=2,
            edge_color="grey")
    labels = {x: round(y, 1) for x, y in nx.get_node_attributes(net, 'EOTX').items()}
    nx.draw_networkx_labels(net, pos=pos, labels=labels)
    nx.draw_networkx_edge_labels(net, pos=pos, edge_labels=nx.get_edge_attributes(net, 'weight'))


def parseairtime(dic):
    """Parse given airtime dict."""
    plot, std = {}, {}
    firstfolder = list(dic.keys())[0]
    for fail in dic[firstfolder].keys():
        counter = []
        for folder in dic.keys():
            try:
                counter.append(sum([len(dic[folder][fail][node])
                                    for node in dic[folder][fail].keys()]))
            except KeyError:
                pass
        plot[fail] = statistics.mean(counter)
        if len(counter) > 1:
            std[fail] = statistics.stdev(counter)
        else:
            std[fail] = 0
    return plot, std


def parsefail(dic):
    """Parse given failhist dict."""
    plotlist, stdlist = {}, {}
    if len(dic) > 1:
        firstfolder = list(dic.keys())[0]
        for fail in dic[firstfolder].keys():
            counter = []
            for folder in dic.keys():
                try:
                    counter.append(dic[folder][fail])
                except KeyError:
                    pass
            plotlist[fail] = statistics.mean(counter)
            if len(counter) > 1:
                stdlist[fail] = statistics.stdev(counter)
            else:
                stdlist[fail] = 0
    else:
        plotlist = dic[list(dic.keys())[0]]
        stdlist = {key: 0 for key in plotlist.keys()}
    return plotlist, stdlist


def readairtime(folder):
    """Read logs from folder."""
    airtime, config, failhist = None, None, None
    if os.path.exists('{}/airtime.json'.format(folder)):
        with open('{}/airtime.json'.format(folder)) as file:
            airtime = json.loads(file.read())
    if os.path.exists('{}/config.json'.format(folder)):
        with open('{}/config.json'.format(folder)) as file:
            config = json.loads(file.read())
    return airtime, config


def readfailhist(folder):
    """Read logs from folder."""
    if os.path.exists('{}/failhist.json'.format(folder)) and os.path.exists('{}/config.json'.format(folder)):
        with open('{}/failhist.json'.format(folder)) as file:
            failhist = json.loads(file.read())
        with open('{}/config.json'.format(folder)) as file:
            config = json.loads(file.read())
        return failhist, config
    raise FileNotFoundError


def readgraph(folder):
    """Read logs from folder."""
    graph, path, eotx, failhist = None, None, None, None
    if os.path.exists('{}/path.json'.format(folder)):
        with open('{}/graph.json'.format(folder)) as file:
            graph = json.loads(file.read())
    if os.path.exists('{}/failhist.json'.format(folder)):
        with open('{}/path.json'.format(folder)) as file:
            path = json.loads(file.read())
    if os.path.exists('{}/eotx.json'.format(folder)):
        with open('{}/eotx.json'.format(folder)) as file:
            eotx = json.loads(file.read())
    if os.path.exists('{}/failhist.json'.format(folder)):
        with open('{}/failhist.json'.format(folder)) as file:
            failhist = json.loads(file.read())
    return graph, path, eotx, failhist


def plotairtime(mainfolder=None, folders=None):
    """Plot airtime for different failures."""
    if folders is None:
        quit(1)
    if mainfolder is None:
        mainfolder = ''
    p.figure(figsize=(20, 10))
    globconfig = {}
    incdicts = {}
    for folder in folders:
        airtime, config = readairtime('{}/{}'.format(mainfolder, folder))
        if airtime is None or config is None:
            print('Can not read log at {}! Continue'.format(folder))
            continue
        elif not globconfig:
            globconfig = config
        if config['own'] and config['sourceshift']:
            if 'MORELESS' not in incdicts.keys():
                incdicts['MORELESS'] = {}
            incdicts['MORELESS'][folder] = airtime
        elif config['own']:
            if 'own' not in incdicts.keys():
                incdicts['own'] = {}
            incdicts['own'][folder] = airtime
        elif config['sourceshift']:
            if 'ss' not in incdicts.keys():
                incdicts['ss'] = {}
            incdicts['ss'][folder] = airtime
        elif config['david']:
            if 'MOREresilience' not in incdicts.keys():
                incdicts['MOREresilience'] = {}
            incdicts['MOREresilience'][folder] = airtime
        else:
            if 'MORE' not in incdicts.keys():
                incdicts['MORE'] = {}
            incdicts['MORE'][folder] = airtime
    plots, std = {}, {}
    failures = set()
    for protocol, dic in incdicts.items():
        plots[protocol], std[protocol] = parseairtime(dic)
        failures |= {fail for fail in plots[protocol]}
    for fail in failures:
        for protocol in plots:
            if fail not in plots[protocol].keys():
                plots[protocol][fail] = plots[protocol]['None']
                std[protocol][fail] = std[protocol]['None']
    plotlist, stdlist = {}, {}
    for protocol in plots:
        plotlist[protocol] = [plots[protocol][key] for key in sorted(plots[protocol].keys())]
        stdlist[protocol] = [std[protocol][key] for key in sorted(std[protocol].keys())]
    width = 0.9/len(plots.keys())
    for protocol in plotlist:
        ind = [x + list(sorted(plots.keys())).index(protocol) * width for x in range(len(failures))]
        p.bar(ind, plotlist[protocol], width=width, label=protocol, yerr=stdlist[protocol],
              error_kw={'elinewidth': width/5})
    p.legend(loc='upper right')
    p.ylabel('Needed airtime in timeslots')
    p.xlabel('Failure')
    p.yscale('log')
    p.xlim([-1, len(failures)])
    p.title('Needed transmissions over failures for different protocols')
    ind = [x + width/2 for x in range(len(failures))]
    p.xticks(ind, labels=sorted(plots[list(plots.keys())[0]].keys()), rotation=90)
    p.tight_layout()
    p.savefig('{}/airtimefail.pdf'.format(mainfolder))
    p.close()


def plotfailhist(mainfolder=None, folders=None):
    """Plot time to finish transmission diagram to compare different simulations."""
    if folders is None:
        quit(1)
    if mainfolder is None:
        mainfolder = ''
    p.figure(figsize=(20, 10))
    failhist, config = {}, {}
    incdicts = {}
    for folder in folders:
        try:
            failhist, config = readfailhist('{}/{}'.format(mainfolder, folder))
        except FileNotFoundError:
            print('No logs found at {}/{}'.format(mainfolder, folder))
            continue
        if config['own'] and config['sourceshift']:
            if 'MORELESS' not in incdicts.keys():
                incdicts['MORELESS'] = {}
            incdicts['MORELESS'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['own']:
            if 'own' not in incdicts.keys():
                incdicts['own'] = {}
            incdicts['own'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['sourceshift']:
            if 'ss' not in incdicts.keys():
                incdicts['ss'] = {}
            incdicts['ss'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['david']:
            if 'MOREresilience' not in incdicts.keys():
                incdicts['MOREresilience'] = {}
            incdicts['MOREresilience'][folder] = {key:value[0] for key, value in failhist.items()}
        else:
            if 'MORE' not in incdicts.keys():
                incdicts['MORE'] = {}
            incdicts['MORE'][folder] = {key: value[0] for key, value in failhist.items()}
    failures = set()
    plots = {}
    std = {}
    for protocol, dic in incdicts.items():
        plots[protocol], std[protocol] = parsefail(dic)
        failures |= {fail for fail in plots[protocol]}
    for fail in failures:
        for protocol in plots:
            if fail not in plots[protocol].keys():
                plots[protocol][fail] = plots[protocol]['None']
                std[protocol][fail] = std[protocol]['None']
    plotlist, stdlist = {}, {}
    for protocol in plots:
        plotlist[protocol] = [plots[protocol][key] for key in sorted(plots[protocol].keys())]
        stdlist[protocol] = [std[protocol][key] for key in sorted(std[protocol].keys())]

    width = 0.9/len(plots.keys())
    for protocol in plotlist:
        ind = [x + list(sorted(plots.keys())).index(protocol) * width for x in range(len(failures))]
        p.bar(ind, plotlist[protocol], width=width, label=protocol, yerr=stdlist[protocol],
              error_kw={'elinewidth': width/5})
    p.legend(loc='upper right')
    p.yscale('log')
    p.ylabel('Needed Latency in timeslots')
    p.xlabel('Failure')
    # p.yscale('log')
    if config['maxduration']:
        p.ylim([0, config['maxduration']])
    else:
        p.ylim(bottom=0)
    p.xlim([-1, len(failures)])
    p.title('Needed timeslots over failures for different protocols')
    ind = [x + 0.5 for x in range(len(plots[list(plots.keys())[0]]))]
    p.xticks(ind, labels=sorted(plots[list(plots.keys())[0]].keys()), rotation=90)
    p.tight_layout()
    p.savefig('{}/timeslotfail.pdf'.format(mainfolder))
    p.close()


def plotgraph(folders=None):
    """Plots used and unused network graphs."""
    if folders is None:
        quit(1)
    p.figure(figsize=(10, 10))
    for folder in folders:
        configlist = []
        graph, path, eotx, failhist = readgraph(folder)
        if graph is None or path is None or eotx is None or failhist is None:
            print('Can not read log at {}! Continue'.format(folder))
            continue
        pos = {node: [graph['nodes'][node]['x'], graph['nodes'][node]['y']] for node in graph['nodes']}
        for edge in graph['links']:
            configlist.append((edge['nodes'][0], edge['nodes'][1], edge['loss']))
        net = nx.Graph()
        net.add_weighted_edges_from(configlist)
        for node in net.nodes:
            net.nodes[node]['EOTX'] = eotx[node][0]
        prevfail = None
        for fail in failhist.keys():
            if fail != 'None':
                if len(fail) == 2:
                    if prevfail is not None:
                        if len(prevfail[0]) == 1:
                            for key, value in prevfail[1].items():
                                net.edges[key]['weight'] = value
                        else:
                            net.edges[(prevfail[0][0], prevfail[0][1])]['weight'] = prevfail[1]
                    prevfail = (fail, net.edges[(fail[0], fail[1])]['weight'])
                    net.edges[(fail[0], fail[1])]['weight'] = 0
                else:
                    if prevfail is not None:
                        if len(prevfail[0]) == 1:
                            for key, value in prevfail[1].items():
                                net.edges[key]['weight'] = value
                        else:
                            net.edges[(prevfail[0][0], prevfail[0][1])]['weight'] = prevfail[1]
                    faildict = {}
                    for edge in list(net.edges):
                        if fail in edge:
                            faildict[edge] = net.edges[edge]['weight']
                            net.edges[edge]['weight'] = 0
                    prevfail = (fail, faildict)

            drawunused(net, pos)
            if fail == 'None':
                p.savefig('{}/graph.pdf'.format(folder))    # Save it once without purple
            alphaval = 1/max([len(value) for value in path[fail].values()])
            for edge in path[fail].keys():
                nx.draw_networkx_edges(net, pos=pos, edgelist=[(edge[0], edge[1])], width=8,
                                       alpha=len(path[fail][edge]) * alphaval, edge_color='purple')
            p.savefig('{}/graphfail{}.pdf'.format(folder, fail))
            p.clf()
    p.close()


if __name__ == '__main__':
    now = datetime.datetime.now()
    # date = int(str(now.year) + str(now.month) + str(now.day))
    date = '../Results/20181115'
    folderlist = []
    folderlst = []
    for i in range(5):
        # folderlist.append('{}/graph{}/test'.format(date, i))
        folderlist.extend(['{}/graph{}/test{}'.format(date, i, j) for j in range(40)])
        mfolder = '{}/graph{}'.format(date, i)
        # folderlst.append('test'.format(i))
        folderlst.extend(['test{}'.format(j) for j in range(40)])
        plotairtime(mfolder, folderlst)
        plotfailhist(mfolder, folderlst)
    # plotgraph(folders=folderlist)
