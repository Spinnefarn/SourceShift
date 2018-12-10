#!/usr/bin/python
# coding=utf-8
"""Will collect interesting logs from subfolders and draw plots."""
import matplotlib
matplotlib.rcParams['backend'] = 'pdf'
import matplotlib.pylab as p
p.rcParams['font.family'] = 'sans-serif'
p.rcParams['font.size'] = 12
import numpy as np
import networkx as nx
import json
import os
import datetime
import statistics
import logging
from scipy import stats


def drawunused(net=None, pos=None):
    """Draw unused graph."""
    if net is None or pos is None:
        return
    nx.draw(net, pos=pos, with_labels=True, node_size=1500, node_color="skyblue", node_shape="o",
            alpha=0.7, linewidths=4, font_size=25, font_color="red", font_weight="bold", width=2,
            edge_color="grey")
    labels = {x: round(y, 1) for x, y in nx.get_node_attributes(net, 'EOTX').items()}
    nx.draw_networkx_labels(net, pos=pos, labels=labels)
    nx.draw_networkx_edge_labels(net, pos=pos, edge_labels=nx.get_edge_attributes(net, 'weight'), font_size=12)


# noinspection PyTypeChecker
def getairtime(mainfolder=None, folders=None):
    """Get parsed values and do statistics with it."""
    if folders is None:
        quit(1)
    if mainfolder is None:
        mainfolder = ''
    globconfig = {}
    incdicts = {}
    for folder in folders:
        airtime, config = readairtime('{}/{}'.format(mainfolder, folder))
        if airtime is None or config is None:
            logging.warning('Can not read log at {}! Continue'.format(folder))
            continue
        elif not globconfig:
            globconfig = config
        if config['own'] and config['sourceshift']:
            if 'MORELESS' not in incdicts.keys():
                incdicts['MORELESS'] = {}
            incdicts['MORELESS'][folder] = airtime
        elif config['sourceshift'] and config['david']:
            if 'MORERESS' not in incdicts.keys():
                incdicts['MORERESS'] = {}
            incdicts['MORERESS'][folder] = airtime
        elif config['own']:
            if 'send back' not in incdicts.keys():
                incdicts['send back'] = {}
            incdicts['send back'][folder] = airtime
        elif config['sourceshift']:
            if 'source shift' not in incdicts.keys():
                incdicts['source shift'] = {}
            incdicts['source shift'][folder] = airtime
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
    return sorted(failures), plotlist, stdlist


# noinspection PyTypeChecker
def getairtimemode(mainfolder=None, folders=None, mode='perhop'):
    """Get parsed values and do statistics with it."""
    if folders is None:
        quit(1)
    if mainfolder is None:
        mainfolder = ''
    globconfig = {}
    incdicts = {}
    for folder in folders:
        airtime, config = readairtime('{}/{}'.format(mainfolder, folder))
        if airtime is None or config is None:
            logging.warning('Can not read log at {}! Continue'.format(folder))
            continue
        elif not globconfig:
            globconfig = config
        try:
            ident = len(config['path']) if mode == 'perhop' else len(config['mcut'])
        except KeyError:
            logging.warning('Old log found at {}/{}'.format(mainfolder, folder))
            continue
        if config['own'] and config['sourceshift']:
            if 'MORELESS' not in incdicts.keys():
                incdicts['MORELESS'] = {}
            if ident not in incdicts['MORELESS'].keys():
                incdicts['MORELESS'][ident] = {}
            incdicts['MORELESS'][ident][folder] = airtime
        elif config['sourceshift'] and config['david']:
            if 'MORERESS' not in incdicts.keys():
                incdicts['MORERESS'] = {}
            if ident not in incdicts['MORERESS'].keys():
                incdicts['MORERESS'][ident] = {}
            incdicts['MORERESS'][ident][folder] = airtime
        elif config['own']:
            if 'send back' not in incdicts.keys():
                incdicts['send back'] = {}
            if ident not in incdicts['send back'].keys():
                incdicts['send back'][ident] = {}
            incdicts['send back'][ident][folder] = airtime
        elif config['sourceshift']:
            if 'source shift' not in incdicts.keys():
                incdicts['source shift'] = {}
            if ident not in incdicts['source shift'].keys():
                incdicts['source shift'][ident] = {}
            incdicts['source shift'][ident][folder] = airtime
        elif config['david']:
            if 'MOREresilience' not in incdicts.keys():
                incdicts['MOREresilience'] = {}
            if ident not in incdicts['MOREresilience'].keys():
                incdicts['MOREresilience'][ident] = {}
            incdicts['MOREresilience'][ident][folder] = airtime
        else:
            if 'MORE' not in incdicts.keys():
                incdicts['MORE'] = {}
            if ident not in incdicts['MORE'].keys():
                incdicts['MORE'][ident] = {}
            incdicts['MORE'][ident][folder] = airtime
    stepset = set()
    plots, std = {}, {}
    for protocol, dic in incdicts.items():
        if protocol not in plots.keys():
            plots[protocol], std[protocol] = {}, {}
        for steps in dic.keys():
            plots[protocol][steps], std[protocol][steps] = parseairtime(dic[steps])
        stepset |= {key for key in plots[protocol].keys()}
    plotlist, stdlist = {}, {}
    plist, slist = {}, {}
    for protocol in plots:
        if protocol not in plotlist.keys():
            plotlist[protocol], stdlist[protocol] = {}, {}
        for steps in plots[protocol]:
            plotlist[protocol][steps] = statistics.mean(plots[protocol][steps].values())
            stdlist[protocol][steps] = statistics.mean(std[protocol][steps].values())
        plist[protocol] = [plotlist[protocol][key] for key in sorted(plotlist[protocol].keys())]
        slist[protocol] = [stdlist[protocol][key] for key in sorted(stdlist[protocol].keys())]
    return sorted(stepset), plist, slist


# noinspection PyTypeChecker
def getfailhist(mainfolder=None, folders=None):
    """Get parsed values and do statistics with it."""
    if folders is None:
        quit(1)
    if mainfolder is None:
        mainfolder = ''
    p.figure(figsize=(20, 10))
    incdicts, config = {}, {}
    for folder in folders:
        try:
            failhist, config = readfailhist('{}/{}'.format(mainfolder, folder))
        except FileNotFoundError:
            logging.warning('No logs found at {}/{}'.format(mainfolder, folder))
            continue
        if config['own'] and config['sourceshift']:
            if 'MORELESS' not in incdicts.keys():
                incdicts['MORELESS'] = {}
            incdicts['MORELESS'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['sourceshift'] and config['david']:
            if 'MORERESS' not in incdicts.keys():
                incdicts['MORERESS'] = {}
            incdicts['MORERESS'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['own']:
            if 'send back' not in incdicts.keys():
                incdicts['send back'] = {}
            incdicts['send back'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['sourceshift']:
            if 'source shift' not in incdicts.keys():
                incdicts['source shift'] = {}
            incdicts['source shift'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['david']:
            if 'MOREresilience' not in incdicts.keys():
                incdicts['MOREresilience'] = {}
            incdicts['MOREresilience'][folder] = {key: value[0] for key, value in failhist.items()}
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
    return sorted(failures), plotlist, stdlist, config


# noinspection PyTypeChecker
def getfailhistmode(mainfolder=None, folders=None, mode='perhop'):
    """Get parsed values and do statistics with it."""
    if folders is None:
        quit(1)
    if mainfolder is None:
        mainfolder = ''
    p.figure(figsize=(20, 10))
    incdicts, config = {}, {}
    for folder in folders:
        try:
            failhist, config = readfailhist('{}/{}'.format(mainfolder, folder))
        except FileNotFoundError:
            logging.warning('No logs found at {}/{}'.format(mainfolder, folder))
            continue
        try:
            ident = len(config['path']) if mode == 'perhop' else len(config['mcut'])
        except KeyError:
            logging.warning('Old log found at {}/{}'.format(mainfolder, folder))
            continue
        if config['own'] and config['sourceshift']:
            if 'MORELESS' not in incdicts.keys():
                incdicts['MORELESS'] = {}
            if ident not in incdicts['MORELESS'].keys():
                incdicts['MORELESS'][ident] = {}
            incdicts['MORELESS'][ident][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['sourceshift'] and config['david']:
            if 'MORERESS' not in incdicts.keys():
                incdicts['MORERESS'] = {}
            if ident not in incdicts['MORERESS'].keys():
                incdicts['MORERESS'][ident] = {}
            incdicts['MORERESS'][ident][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['own']:
            if 'send back' not in incdicts.keys():
                incdicts['send back'] = {}
            if ident not in incdicts['send back'].keys():
                incdicts['send back'][ident] = {}
            incdicts['send back'][ident][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['sourceshift']:
            if 'source shift' not in incdicts.keys():
                incdicts['source shift'] = {}
            if ident not in incdicts['source shift'].keys():
                incdicts['source shift'][ident] = {}
            incdicts['source shift'][ident][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['david']:
            if 'MOREresilience' not in incdicts.keys():
                incdicts['MOREresilience'] = {}
            if ident not in incdicts['MOREresilience'].keys():
                incdicts['MOREresilience'][ident] = {}
            incdicts['MOREresilience'][ident][folder] = {key: value[0] for key, value in failhist.items()}
        else:
            if 'MORE' not in incdicts.keys():
                incdicts['MORE'] = {}
            if ident not in incdicts['MORE'].keys():
                incdicts['MORE'][ident] = {}
            incdicts['MORE'][ident][folder] = {key: value[0] for key, value in failhist.items()}
    stepset = set()
    plots = {}
    std = {}
    for protocol, dic in incdicts.items():
        if protocol not in plots.keys():
            plots[protocol], std[protocol] = {}, {}
        for steps in dic.keys():
            plots[protocol][steps], std[protocol][steps] = parsefail(dic[steps])
        stepset |= {key for key in plots[protocol].keys()}
    plotlist, stdlist = {}, {}
    plist, slist = {}, {}
    for protocol in plots:
        if protocol not in plotlist.keys():
            plotlist[protocol], stdlist[protocol] = {}, {}
        for steps in plots[protocol]:
            plotlist[protocol][steps] = statistics.mean(plots[protocol][steps].values())
            stdlist[protocol][steps] = statistics.mean(std[protocol][steps].values())
        plist[protocol] = [plotlist[protocol][key] for key in sorted(plotlist[protocol].keys())]
        slist[protocol] = [stdlist[protocol][key] for key in sorted(stdlist[protocol].keys())]
    return sorted(stepset), plist, slist, config


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


def parseaircdf(mainfolder, folders, mode='regular'):
    """Read and parse logs for plotting as cdf."""
    incdicts, globconfig = {}, {}
    for folder in folders:
        airtime, config = readairtime('{}/{}'.format(mainfolder, folder))
        if airtime is None or config is None:
            logging.warning('Can not read log at {}/{}! Continue'.format(mainfolder, folder))
            continue
        try:
            if config['own'] and config['sourceshift']:
                if 'MORELESS' not in incdicts.keys():
                    incdicts['MORELESS'] = {}
                incdicts['MORELESS'][folder] = airtime
            elif config['sourceshift'] and config['david']:
                if 'MORERESS' not in incdicts.keys():
                    incdicts['MORERESS'] = {}
                incdicts['MORERESS'][folder] = airtime
            elif config['own']:
                if 'send back' not in incdicts.keys():
                    incdicts['send back'] = {}
                incdicts['send back'][folder] = airtime
            elif config['sourceshift']:
                if 'source shift' not in incdicts.keys():
                    incdicts['source shift'] = {}
                incdicts['source shift'][folder] = airtime
            elif config['david']:
                if 'MOREresilience' not in incdicts.keys():
                    incdicts['MOREresilience'] = {}
                incdicts['MOREresilience'][folder] = airtime
            else:
                if 'MORE' not in incdicts.keys():
                    incdicts['MORE'] = {}
                incdicts['MORE'][folder] = airtime
            if not globconfig:
                globconfig = config
        except KeyError:
            logging.warning('Uncomplete config in {}'.format(folder))
    failure = set()
    for protocol in incdicts.keys():
        for folder in incdicts[protocol]:
            failure |= set(incdicts[protocol][folder].keys())
    plots = {}
    for protocol in incdicts:
        plots[protocol] = {}
        for fail in failure:
            counter = []
            for folder in incdicts[protocol].keys():
                try:
                    counter.append(sum([len(incdicts[protocol][folder][fail][node])
                                        for node in incdicts[protocol][folder][fail].keys()]))
                except KeyError:
                    pass
            plots[protocol][fail] = counter
    if mode == 'regular':
        plotlist = {}
        for protocol in plots:
            plotlist[protocol] = []
            for fail in failure:
                if fail in plots[protocol].keys():
                    plotlist[protocol].extend(plots[protocol][fail])
                else:
                    plotlist[protocol].extend(plots[protocol]['None'])
        return plotlist
    else:
        gainlist = {}
        for protocol in plots:
            gainlist[protocol] = []
            for fail in failure:
                if fail in plots[protocol].keys() and len(plots[protocol][fail]):
                    proval = statistics.mean(plots[protocol][fail])
                else:
                    proval = statistics.mean(plots[protocol]['None'])
                if fail in plots['MORE'].keys() and len(plots['MORE'][fail]):
                    moreval = statistics.mean(plots['MORE'][fail])
                else:
                    moreval = statistics.mean(plots['MORE']['None'])
                gainlist[protocol].append(((proval / moreval) - 1) * 100)
        return gainlist


def parsefailcdf(mainfolder, folders, mode='regular'):
    """Read and parse logs for plotting as cdf."""
    incdicts = {}
    for folder in folders:
        try:
            failhist, config = readfailhist('{}/{}'.format(mainfolder, folder))
        except FileNotFoundError:
            logging.warning('No logs found at {}/{}'.format(mainfolder, folder))
            continue
        try:
            if config['own'] and config['sourceshift']:
                if 'MORELESS' not in incdicts.keys():
                    incdicts['MORELESS'] = {}
                incdicts['MORELESS'][folder] = {key: value[0] for key, value in failhist.items()}
            elif config['david'] and config['sourceshift']:
                if 'MORERESS' not in incdicts.keys():
                    incdicts['MORERESS'] = {}
                incdicts['MORERESS'][folder] = {key: value[0] for key, value in failhist.items()}
            elif config['own']:
                if 'send back' not in incdicts.keys():
                    incdicts['send back'] = {}
                incdicts['send back'][folder] = {key: value[0] for key, value in failhist.items()}
            elif config['sourceshift']:
                if 'source shift' not in incdicts.keys():
                    incdicts['source shift'] = {}
                incdicts['source shift'][folder] = {key: value[0] for key, value in failhist.items()}
            elif config['david']:
                if 'MOREresilience' not in incdicts.keys():
                    incdicts['MOREresilience'] = {}
                incdicts['MOREresilience'][folder] = {key: value[0] for key, value in failhist.items()}
            else:
                if 'MORE' not in incdicts.keys():
                    incdicts['MORE'] = {}
                incdicts['MORE'][folder] = {key: value[0] for key, value in failhist.items()}
        except KeyError:
            logging.warning('Uncomplete config in {}'.format(folder))
    failure = set()
    for protocol in incdicts.keys():
        for folder in incdicts[protocol]:
            failure |= set(incdicts[protocol][folder].keys())
    plots = {}
    for protocol in incdicts.keys():
        plots[protocol] = {}
        if len(incdicts[protocol]) > 1:
            for fail in failure:
                plots[protocol][fail] = []
                counter = []
                for folder in incdicts[protocol].keys():
                    try:
                        counter.append(incdicts[protocol][folder][fail])
                    except KeyError:
                        pass
                if len(counter):
                    plots[protocol][fail].extend(counter)
        else:
            for fail in failure:
                try:
                    plots[protocol][fail] = [incdicts[protocol][list(incdicts[protocol].keys())[0]][fail]]
                except KeyError:
                    pass

    if mode == 'regular':
        plotlist = {}
        for protocol in plots:
            plotlist[protocol] = []
            for fail in failure:
                if fail in plots[protocol].keys():
                    plotlist[protocol].extend(plots[protocol][fail])
                else:
                    plotlist[protocol].extend(plots[protocol]['None'])
        return plotlist
    else:
        gainlist = {}
        for protocol in plots:
            gainlist[protocol] = []
            for fail in failure:
                if fail in plots[protocol].keys() and len(plots[protocol][fail]):
                    proval = statistics.mean(plots[protocol][fail])
                else:
                    proval = statistics.mean(plots[protocol]['None'])
                if fail in plots['MORE'].keys() and len(plots['MORE'][fail]):
                    moreval = statistics.mean(plots['MORE'][fail])
                else:
                    moreval = statistics.mean(plots['MORE']['None'])
                gainlist[protocol].append(((proval / moreval) - 1) * 100)
        return gainlist


def parsetrash(mainfolder, folders, mode='real'):
    """Read and parse trash logs for plotting as cdf."""
    incdicts = {}
    for folder in folders:
        try:
            trash, config = readtrash('{}/{}'.format(mainfolder, folder), mode=mode)
        except FileNotFoundError:
            logging.warning('No trash logs found at {}/{}'.format(mainfolder, folder))
            continue
        try:
            if config['own'] and config['sourceshift']:
                if 'MORELESS' not in incdicts.keys():
                    incdicts['MORELESS'] = {}
                incdicts['MORELESS'][folder] = trash
            elif config['david'] and config['sourceshift']:
                if 'MORERESS' not in incdicts.keys():
                    incdicts['MORERESS'] = {}
                incdicts['MORERESS'][folder] = trash
            elif config['own']:
                if 'send back' not in incdicts.keys():
                    incdicts['send back'] = {}
                incdicts['send back'][folder] = trash
            elif config['sourceshift']:
                if 'source shift' not in incdicts.keys():
                    incdicts['source shift'] = {}
                incdicts['source shift'][folder] = trash
            elif config['david']:
                if 'MOREresilience' not in incdicts.keys():
                    incdicts['MOREresilience'] = {}
                incdicts['MOREresilience'][folder] = trash
            else:
                if 'MORE' not in incdicts.keys():
                    incdicts['MORE'] = {}
                incdicts['MORE'][folder] = trash
        except KeyError:
            logging.warning('Uncomplete config in {}'.format(folder))
    plotlist = {}
    nodes = {}
    for protocol in incdicts.keys():
        plotlist[protocol] = {}
        nodes[protocol] = set()
        for folder in incdicts[protocol].keys():
            nodes[protocol].update(incdicts[protocol][folder])
            for node in incdicts[protocol][folder].keys():
                for timestamp in incdicts[protocol][folder][node].keys():
                    if int(timestamp) not in plotlist[protocol].keys():
                        plotlist[protocol][int(timestamp)] = incdicts[protocol][folder][node][timestamp]
                    else:
                        plotlist[protocol][int(timestamp)] += incdicts[protocol][folder][node][timestamp]
    N = 10
    retlist = {}
    for protocol in plotlist:
        retlist[protocol] = {}
        values = np.convolve(plotlist[protocol].values(), np.ones((N,))/N, mode='valid')
        retlist[protocol] = {key: value for key, value in enumerate(values)}
    return retlist


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


def readtrash(folder, mode='real'):
    """Read trash log."""
    trash, config = None, None
    if os.path.exists('{}/{}trash.json'.format(folder, mode)):
        with open('{}/{}trash.json'.format(folder, mode)) as file:
            trash = json.loads(file.read())
    if os.path.exists('{}/config.json'.format(folder, mode)):
        with open('{}/config.json'.format(folder, mode)) as file:
            config = json.loads(file.read())
    if trash is None or config is None:
        raise FileNotFoundError
    else:
        return trash, config


def plotairtime(mainfolder=None, folders=None):
    """Plot airtime for different failures."""
    p.figure(figsize=(8.45, 6.18))
    failures, plotlist, stdlist = getairtime(mainfolder, folders)
    width = 0.9 / len(plotlist.keys())
    for protocol in sorted(plotlist.keys()):
        ind = [x + list(sorted(plotlist.keys())).index(protocol) * width for x in range(len(failures))]
        p.bar(ind, plotlist[protocol], width=width, label=protocol, yerr=stdlist[protocol],
              error_kw={'elinewidth': width / 3})
    p.legend(loc='best')
    p.ylabel('Needed airtime in timeslots')
    p.xlabel('Failure')
    p.yscale('log')
    p.xlim([-1, len(failures)])
    p.title('Needed transmissions over failures for different protocols')
    ind = [x + width / 2 for x in range(len(failures))]
    p.xticks(ind, labels=failures, rotation=90)
    p.grid(True)
    p.tight_layout()
    p.savefig('{}/airfailbar.pdf'.format(mainfolder))
    p.close()


def plotaircdf(mainfolder=None, folders=None):
    """Plot airtime CDF."""
    if folders is None and mainfolder is not None:
        folders = []
        cmain = [folder for folder in os.listdir(mainfolder) if os.path.isdir('{}/{}'.format(mainfolder, folder))]
        for subfolder in cmain:
            folders.extend(['{}/{}'.format(subfolder, subsubfolder)
                            for subsubfolder in os.listdir('{}/{}'.format(mainfolder, subfolder))
                            if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder))])
    if mainfolder is None:
        mainfolder = ''
    plotlist = parseaircdf(mainfolder, folders)
    for mode in ['regular', 'close']:
        if mode == 'regular':
            p.figure(figsize=(4.2, 6.18))
        else:
            p.figure(figsize=(6.18, 4.2))
        for protocol in sorted(plotlist.keys()):
            p.plot(sorted(plotlist[protocol]), np.linspace(0, 1, len(plotlist[protocol])), label=protocol, alpha=0.8)
        p.title('Used airtime per protocol')
        p.ylabel('Fraction of Airtime')
        p.xlabel('Airtime in transmissions')
        # p.ylim([0.8, 1])
        # p.xlim(left=0)
        p.xscale('log')
        p.xticks(rotation=90)
        p.grid(True)
        p.legend(loc='best')
        p.tight_layout()
        if mode == 'regular':
            p.ylim([0, 1])
            p.savefig('{}/airtimecdf.pdf'.format(mainfolder))
        else:
            p.ylim([0.9, 1])
            p.savefig('{}/airclosecdf.pdf'.format(mainfolder))
        p.close()


def plotlatcdf(mainfolder=None, folders=None):
    """Plot latency CDF."""
    if folders is None and mainfolder is not None:
        folders = []
        cmain = [folder for folder in os.listdir(mainfolder) if os.path.isdir('{}/{}'.format(mainfolder, folder))]
        for subfolder in cmain:
            folders.extend(['{}/{}'.format(subfolder, subsubfolder)
                            for subsubfolder in os.listdir('{}/{}'.format(mainfolder, subfolder))
                            if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder))])
    if mainfolder is None:
        mainfolder = ''
    plotlist = parsefailcdf(mainfolder, folders)
    for mode in ['regular', 'close']:
        if mode == 'regular':
            p.figure(figsize=(4.2, 6.18))
        else:
            p.figure(figsize=(6.18, 4.2))
        for protocol in sorted(plotlist.keys()):
            p.plot(sorted(plotlist[protocol]), np.linspace(0, 1, len(plotlist[protocol])), label=protocol, alpha=0.8)
        p.title('Required latency per protocol')
        p.ylabel('Fraction of Latency')
        p.xlabel('Latency in timeslots')
        # p.ylim([0.8, 1])
        # p.xlim(left=0)
        p.xscale('log')
        p.xticks(rotation=90)
        p.grid(True)
        p.legend(loc='best')
        p.tight_layout()
        if mode == 'regular':
            p.ylim([0, 1])
            p.savefig('{}/latencycdf.pdf'.format(mainfolder))
        else:
            p.ylim([0.9, 1])
            p.savefig('{}/latclosecdf.pdf'.format(mainfolder))
        p.close()


def plotfailhist(mainfolder=None, folders=None):
    """Plot time to finish transmission diagram to compare different simulations."""
    failures, plotlist, stdlist, config = getfailhist(mainfolder, folders)
    width = 0.9 / len(plotlist.keys())
    p.figure(figsize=(8.45, 6.18))
    for protocol in sorted(plotlist.keys()):
        ind = [x + list(sorted(plotlist.keys())).index(protocol) * width for x in range(len(failures))]
        p.bar(ind, plotlist[protocol], width=width, label=protocol, yerr=stdlist[protocol],
              error_kw={'elinewidth': width / 3})
    p.legend(loc='best')
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
    ind = [x + 0.5 for x in range(len(failures))]
    p.xticks(ind, labels=failures, rotation=90)
    p.grid(True)
    p.tight_layout()
    p.savefig('{}/latfailbar.pdf'.format(mainfolder))
    p.close()


def plotgain(mainfolder=None, folders=None):
    """Plot gain in latency."""
    for mode in ['airtime', 'latency']:
        if mode == 'latency':
            failures, plotlist, stdlist, config = getfailhist(mainfolder, folders)
        else:
            failures, plotlist, stdlist = getairtime(mainfolder, folders)
        gainlist = {}
        for protocol in plotlist:
            if protocol != 'MORE':
                try:
                    gainlist[protocol] = statistics.mean([((plotlist[protocol][j] / plotlist['MORE'][j]) - 1) * 100
                                                          for j in range(len(failures))])
                except KeyError:
                    logging.info('Not able to plot gain')
                    return      # Could happen in case of uncomplete logs like when plotting while simulating
        p.figure(figsize=(5, 5))
        width = 0.9 / len(plotlist.keys())
        # for protocol in gainlist:
        #    ind = [x + list(sorted(plotlist.keys())).index(protocol) * width for x in range(len(failures))]
        p.bar(range(len(gainlist.keys())), gainlist.values(), width=width)
        # p.legend(loc='best')
        if mode == 'latency':
            p.title('Latency compared to MORE')
        else:
            p.title('Airtime compared to MORE')
        p.ylim([-100, 100])
        p.ylabel('Mean relative difference [%]')
        # p.xlim([-1, len(failures)])
        # p.xlabel('Approach')
        p.grid()
        # ind = [x + width / 2 for x in range(len(failures))]
        p.xticks(range(len(gainlist.keys())), labels=gainlist.keys())
        p.tight_layout()
        if mode == 'latency':
            p.savefig('{}/latencygain.pdf'.format(mainfolder))
        else:
            p.savefig('{}/airtimegain.pdf'.format(mainfolder))
        p.close()


def plotgaincdf(mainfolder=None):
    """Plot gain in latency and airtime as cdf."""
    if mainfolder is None:
        return
    folders = []
    cmain = [folder for folder in os.listdir(mainfolder) if os.path.isdir('{}/{}'.format(mainfolder, folder))]
    for subfolder in cmain:
        folders.extend(['{}/{}'.format(subfolder, subsubfolder)
                        for subsubfolder in os.listdir('{}/{}'.format(mainfolder, subfolder))
                        if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder))])
    for mode in ['airtime', 'latency']:
        if mode == 'latency':
            gainlist = parsefailcdf(mainfolder, folders, mode='gain')
        else:
            gainlist = parseaircdf(mainfolder, folders, mode='gain')
        p.figure(figsize=(4.2, 6.18))
        for protocol in sorted(gainlist.keys()):
            p.plot(sorted(gainlist[protocol]), np.linspace(0, 1, len(gainlist[protocol])), label=protocol, alpha=0.8)
        if mode == 'latency':
            p.title('Difference in latency compared to MORE')
        else:
            p.title('Difference in airtime compared to MORE')
        p.ylabel('Fraction of Gain')
        p.xlabel('Difference in percent')
        p.ylim([0, 1])
        # p.ylim([0.8, 1])
        # p.xlim(left=0)
        # p.xscale('log')
        p.grid(True)
        p.legend(loc='best')
        p.tight_layout()
        if mode == 'latency':
            p.savefig('{}/latgaincdf.pdf'.format(mainfolder))
        else:
            p.savefig('{}/airgaincdf.pdf'.format(mainfolder))
        p.close()


def plotgraph(folders=None):
    """Plots used and unused network graphs."""
    if folders is None:
        quit(1)
    p.figure(figsize=(6.18, 6.18))  # Plot graph \textwidth
    # p.figure(figsize=(3, 3))          # Plot graph to use in subfigure 0.5 * \textwidth
    for folder in folders:
        configlist = []
        graph, path, eotx, failhist = readgraph(folder)
        if graph is None or path is None or eotx is None or failhist is None:
            logging.warning('Can not read log at {}! Continue'.format(folder))
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
                p.savefig('{}/graph.pdf'.format(folder))  # Save it once without purple
            alphaval = 1 / max([len(value) for value in path[fail].values()])
            for edge in path[fail].keys():
                nx.draw_networkx_edges(net, pos=pos, edgelist=[(edge[0], edge[1])], width=8,
                                       alpha=len(path[fail][edge]) * alphaval, edge_color='purple')
            p.savefig('{}/graphfail{}.pdf'.format(folder, fail))
            p.clf()
    p.close()


def plotperhop(mainfolder=None, kind='perhop'):
    """Plot fancy graphs per hop count."""
    if mainfolder is None:
        return
    folders = []
    cmain = [folder for folder in os.listdir(mainfolder) if os.path.isdir('{}/{}'.format(mainfolder, folder))]
    for subfolder in cmain:
        folders.extend(['{}/{}'.format(subfolder, subsubfolder)
                        for subsubfolder in os.listdir('{}/{}'.format(mainfolder, subfolder))
                        if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder))])
    for mode in ['airtime', 'latency']:
        if mode == 'latency':
            steps, plotlist, stdlist, config = getfailhistmode(mainfolder, folders, mode=kind)
        else:
            steps, plotlist, stdlist = getairtimemode(mainfolder, folders, mode=kind)
        if not plotlist:
            continue
        width = 0.9 / len(plotlist.keys())
        p.figure(figsize=(8.45, 6.18))
        for protocol in sorted(plotlist.keys()):
            ind = [x + list(sorted(plotlist.keys())).index(protocol) * width for x in range(len(plotlist[protocol]))]
            p.bar(ind, plotlist[protocol], width=width, label=protocol, yerr=stdlist[protocol])
            # error_kw={'elinewidth': width/3})
        p.legend(loc='best')
        p.yscale('log')
        p.ylabel('Needed Latency in timeslots')
        if kind == 'perhop':
            p.xlabel('Minimum amount of hops between Source and Destination')
        else:
            p.xlabel('Mincut')
        p.yscale('log')
        p.ylim(bottom=0)
        p.xlim([-1, len(steps)])
        p.title('Needed {} over different networks and protocols'.format(mode))
        ind = [x + 0.5 for x in range(len(steps))]
        p.xticks(ind, labels=steps)
        p.grid(True)
        p.tight_layout()
        name = 'hop' if kind == 'perhop' else 'mcut'
        if mode == 'airtime':
            p.savefig('{}/air{}bar.pdf'.format(mainfolder, name))
        else:
            p.savefig('{}/lat{}bar.pdf'.format(mainfolder, name))
        p.close()


def plotqq(mainfolder=None):
    """Plot QQDiagramm."""
    if mainfolder is None or len(mainfolder) != 2:
        return
    plotlist = []
    for x in mainfolder:
        folders = []
        cmain = [folder for folder in os.listdir(x) if os.path.isdir('{}/{}'.format(x, folder))]
        for subfolder in cmain:
            folders.extend(['{}/{}'.format(subfolder, subsubfolder)
                            for subsubfolder in os.listdir('{}/{}'.format(x, subfolder))
                            if os.path.isdir('{}/{}/{}'.format(x, subfolder, subsubfolder))])
        plotlist.append(parsefailcdf(x, folders))

    # p.figure(figsize=(4.2, 6.18))
    for protocol in sorted(plotlist[0].keys()):
        # result = stats.mannwhitneyu(plotlist[0][protocol], plotlist[1][protocol])
        # result = stats.ranksums(plotlist[0][protocol], plotlist[1][protocol])
        logging.info('KS 2Samp for {} is {}'.format(protocol,   # dist unequal if p < 1% | equal if p > 10%
                                                    stats.ks_2samp(plotlist[0][protocol], plotlist[1][protocol])))
        # p.plot(sorted(plotlist[0][protocol]), np.linspace(0, 1, len(plotlist[0][protocol])),
        #       label=protocol, alpha=0.8)
    # p.title('Needed latency per protocol')
    # p.ylabel('Fraction of Latency')
    # p.xlabel('Latency in timeslots')
    # p.ylim([0.8, 1])
    # p.xlim(left=0)
    # p.xscale('log')
    # p.xticks(rotation=90)
    # p.grid(True)
    # p.legend(loc='best')
    # p.tight_layout()
    # p.ylim([0, 1])
    # p.savefig('latencycdf.pdf')
    # p.close()


def plottrash(mainfolder=None):
    """Plot cdf over incomming trash."""
    if mainfolder is None:
        return
    folders, plotlist = [], {}
    cmain = [folder for folder in os.listdir(mainfolder) if os.path.isdir('{}/{}'.format(mainfolder, folder))]
    for subfolder in cmain:
        folders.extend(['{}/{}'.format(subfolder, subsubfolder)
                        for subsubfolder in os.listdir('{}/{}'.format(mainfolder, subfolder))
                        if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder))])
    for mode in ['real', 'overhearing']:
        try:
            plotlist = parsetrash(mainfolder, folders, mode=mode)
        except AttributeError:
            logging.warning('Old log format in {}'.format(mainfolder))
            return
        p.figure(figsize=(6.18, 6.18))
        for protocol in sorted(plotlist.keys()):
            p.plot(list(plotlist[protocol].keys()), list(plotlist[protocol].values()), label=protocol, alpha=0.5)
        p.title('Incomming trash per protocol')
        p.ylabel('Total amount of trash per time slots')
        p.xlabel('Time slot')
        p.xlim(left=0)
        p.ylim(bottom=0)
        p.yscale('log')
        p.grid(True)
        p.legend(loc='best')
        p.tight_layout()
        p.savefig('{}/{}trashcdf.pdf'.format(mainfolder, mode))
        p.close()


if __name__ == '__main__':
    logging.basicConfig(filename='plotlog.log', level=logging.DEBUG, filemode='w')
    now = datetime.datetime.now()
    # date = str(int(str(now.year) + str(now.month) + str(now.day)))
    date = '../2018126'
    folderlist = []
    for i in range(1):
        folderlst = []
        # folderlist.append('{}/graph{}/test'.format(date, i))
        folderlist.extend(['{}/graph{}/test{}'.format(date, i, j) for j in range(100)])
        mfolder = '{}/graph{}'.format(date, i)
        # folderlst.append('test'.format(i))
        folderlst.extend(['test{}'.format(j) for j in range(10)])
        # plotaircdf('{}/graph{}'.format(date, i), folderlst)
        # plotlatcdf('{}/graph{}'.format(date, i), folderlst)
        # plotairtime(mfolder, folderlst)
        # plotgain(mfolder, folderlst)
        # plotfailhist(mfolder, folderlst)
    # plotaircdf(date)
    # plotlatcdf(date)
    # plotgaincdf(date)
    # plotperhop(date)
    # plotperhop(date, kind='mcut')
    # plotqq(['../expdav', '../expnodav'])
    plottrash(date)
    # plotgraph(folders=folderlist)
