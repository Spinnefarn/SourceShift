#!/usr/bin/python
# coding=utf-8
"""Will collect interesting logs from subfolders and draw plots."""

import matplotlib.pylab as p
import numpy as np
import networkx as nx
import json
import os
import datetime
import statistics
import logging
from tex import setup
from scipy import stats


def drawunused(net=None, pos=None):
    """Draw unused graph."""
    if net is None or pos is None:
        return
    # nx.draw(net, pos=pos, with_labels=True, node_size=1500, node_color="skyblue", node_shape="o",
    #        alpha=0.7, linewidths=4, font_size=25, font_color="red", font_weight="bold", width=2,
    nx.draw(net, pos=pos, with_labels=True, node_size=200, node_color="white", node_shape="o",
            alpha=0.9, linewidths=4, font_size=12, font_color="black", font_weight="bold", width=2,
            edge_color="grey")
    # labels = {x: round(y, 1) for x, y in nx.get_node_attributes(net, 'EOTX').items()}
    # nx.draw_networkx_labels(net, pos=pos, labels=labels)
    nx.draw_networkx_edge_labels(net, pos=pos, edge_labels=nx.get_edge_attributes(net, 'weight'), font_size=10)


# noinspection PyTypeChecker
def getairtime(mainfolder=None, folders=None, plotfail='all'):
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
        elif config['sourceshift'] and config['moreres']:
            if 'MORERESS' not in incdicts.keys():
                incdicts['MORERESS'] = {}
            incdicts['MORERESS'][folder] = airtime
        elif config['own']:
            if 'Send Aback' not in incdicts.keys():
                incdicts['Send Aback'] = {}
            incdicts['Send Aback'][folder] = airtime
        elif config['sourceshift']:
            if 'Source Shift' not in incdicts.keys():
                incdicts['Source Shift'] = {}
            incdicts['Source Shift'][folder] = airtime
        elif config['nomore']:
            if 'NOMORE' not in incdicts.keys():
                incdicts['NOMORE'] = {}
            incdicts['NOMORE'][folder] = airtime
        elif config['moreres']:
            if 'MOREresilience' not in incdicts.keys():
                incdicts['MOREresilience'] = {}
            incdicts['MOREresilience'][folder] = airtime
        elif config['optimal']:
            if 'Optimal' not in incdicts.keys():
                incdicts['Optimal'] = {}
            incdicts['Optimal'][folder] = airtime
        elif config['ANChOR']:
            if 'ANChOR' not in incdicts.keys():
                incdicts['ANChOR'] = {}
            incdicts['ANChOR'][folder] = airtime
        else:
            if 'MORE' not in incdicts.keys():
                incdicts['MORE'] = {}
            incdicts['MORE'][folder] = airtime
    plots, std = {}, {}
    failures = set()
    for protocol, dic in incdicts.items():
        plots[protocol], std[protocol] = parseairtime(dic, plotfail=plotfail)
        failures |= {fail for fail in plots[protocol]}
    for fail in failures:
        for protocol in plots:
            if not plots[protocol][fail]:
                plots[protocol][fail] = plots[protocol]['None']
                std[protocol][fail] = std[protocol]['None']
    plotlist, stdlist = {}, {}
    for protocol in plots:
        plotlist[protocol] = [plots[protocol][key] for key in sorted(plots[protocol].keys())]
        stdlist[protocol] = [std[protocol][key] for key in sorted(std[protocol].keys())]
    return sorted(failures), plotlist, stdlist


# noinspection PyTypeChecker
def getairtimemode(mainfolder=None, folders=None, mode='perhop', plotfail='all', box=False):
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
        elif config['sourceshift'] and config['moreres']:
            if 'MORERESS' not in incdicts.keys():
                incdicts['MORERESS'] = {}
            if ident not in incdicts['MORERESS'].keys():
                incdicts['MORERESS'][ident] = {}
            incdicts['MORERESS'][ident][folder] = airtime
        elif config['own']:
            if 'Send Aback' not in incdicts.keys():
                incdicts['Send Aback'] = {}
            if ident not in incdicts['Send Aback'].keys():
                incdicts['Send Aback'][ident] = {}
            incdicts['Send Aback'][ident][folder] = airtime
        elif config['sourceshift']:
            if 'Source Shift' not in incdicts.keys():
                incdicts['Source Shift'] = {}
            if ident not in incdicts['Source Shift'].keys():
                incdicts['Source Shift'][ident] = {}
            incdicts['Source Shift'][ident][folder] = airtime
        elif config['nomore']:
            if 'NOMORE' not in incdicts.keys():
                incdicts['NOMORE'] = {}
            if ident not in incdicts['NOMORE'].keys():
                incdicts['NOMORE'][ident] = {}
            incdicts['NOMORE'][ident][folder] = airtime
        elif config['moreres']:
            if 'MOREresilience' not in incdicts.keys():
                incdicts['MOREresilience'] = {}
            if ident not in incdicts['MOREresilience'].keys():
                incdicts['MOREresilience'][ident] = {}
            incdicts['MOREresilience'][ident][folder] = airtime
        elif config['optimal']:
            if 'Optimal' not in incdicts.keys():
                incdicts['Optimal'] = {}
            if ident not in incdicts['Optimal'].keys():
                incdicts['Optimal'][ident] = {}
            incdicts['Optimal'][ident][folder] = airtime
        elif config['ANChOR']:
            if 'ANChOR' not in incdicts.keys():
                incdicts['ANChOR'] = {}
            if ident not in incdicts['ANChOR'].keys():
                incdicts['ANChOR'][ident] = {}
            incdicts['ANChOR'][ident][folder] = airtime
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
            plots[protocol][steps], std[protocol][steps] = parseairtime(dic[steps], plotfail=plotfail)
        stepset |= {key for key in plots[protocol].keys()}
    plotlist, stdlist = {}, {}
    plist, slist = {}, {}
    for protocol in plots:
        if protocol not in plotlist.keys():
            plotlist[protocol], stdlist[protocol] = {}, {}
        for steps in plots[protocol]:
            if box:
                if steps not in plotlist[protocol].keys():
                    plotlist[protocol][steps] = []
                for fail in plots[protocol][steps].keys():
                    plotlist[protocol][steps].append(plots[protocol][steps][fail])
            else:
                plotlist[protocol][steps] = statistics.mean(plots[protocol][steps].values())
                stdlist[protocol][steps] = statistics.mean(std[protocol][steps].values())
        if not box:
            plist[protocol] = [plotlist[protocol][key] for key in sorted(plotlist[protocol].keys())]
            slist[protocol] = [stdlist[protocol][key] for key in sorted(stdlist[protocol].keys())]
        else:
            if protocol not in plist.keys():
                plist[protocol] = []
            if mode != 'perhop':
                for step in plotlist[protocol].keys():
                    plist[protocol].extend(plotlist[protocol][step])
            else:
                plist[protocol] = plotlist[protocol]
    return sorted(stepset), plist, slist


# noinspection PyTypeChecker
def getfailhist(mainfolder=None, folders=None, plotfail='all'):
    """Get parsed values and do statistics with it."""
    if folders is None:
        quit(1)
    if mainfolder is None:
        mainfolder = ''
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
        elif config['sourceshift'] and config['moreres']:
            if 'MORERESS' not in incdicts.keys():
                incdicts['MORERESS'] = {}
            incdicts['MORERESS'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['own']:
            if 'Send Aback' not in incdicts.keys():
                incdicts['Send Aback'] = {}
            incdicts['Send Aback'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['sourceshift']:
            if 'Source Shift' not in incdicts.keys():
                incdicts['Source Shift'] = {}
            incdicts['Source Shift'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['nomore']:
            if 'NOMORE' not in incdicts.keys():
                incdicts['NOMORE'] = {}
            incdicts['NOMORE'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['moreres']:
            if 'MOREresilience' not in incdicts.keys():
                incdicts['MOREresilience'] = {}
            incdicts['MOREresilience'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['optimal']:
            if 'Optimal' not in incdicts.keys():
                incdicts['Optimal'] = {}
            incdicts['Optimal'][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['ANChOR']:
            if 'ANChOR' not in incdicts.keys():
                incdicts['ANChOR'] = {}
            incdicts['ANChOR'][folder] = {key: value[0] for key, value in failhist.items()}
        else:
            if 'MORE' not in incdicts.keys():
                incdicts['MORE'] = {}
            incdicts['MORE'][folder] = {key: value[0] for key, value in failhist.items()}
    failures = set()
    plots = {}
    std = {}
    for protocol, dic in incdicts.items():
        plots[protocol], std[protocol] = parsefail(dic, plotfail=plotfail)
        failures |= {fail for fail in plots[protocol]}
    for fail in failures:
        for protocol in plots:
            if not plots[protocol][fail]:
                plots[protocol][fail] = plots[protocol]['None']
                std[protocol][fail] = std[protocol]['None']
    plotlist, stdlist = {}, {}
    for protocol in plots:
        plotlist[protocol] = [plots[protocol][key] for key in sorted(plots[protocol].keys())]
        stdlist[protocol] = [std[protocol][key] for key in sorted(std[protocol].keys())]
    return sorted(failures), plotlist, stdlist, config


# noinspection PyTypeChecker
def getfailhistmode(mainfolder=None, folders=None, mode='perhop', plotfail='all', box=False):
    """Get parsed values and do statistics with it."""
    if folders is None:
        quit(1)
    if mainfolder is None:
        mainfolder = ''
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
        elif config['sourceshift'] and config['moreres']:
            if 'MORERESS' not in incdicts.keys():
                incdicts['MORERESS'] = {}
            if ident not in incdicts['MORERESS'].keys():
                incdicts['MORERESS'][ident] = {}
            incdicts['MORERESS'][ident][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['own']:
            if 'Send Aback' not in incdicts.keys():
                incdicts['Send Aback'] = {}
            if ident not in incdicts['Send Aback'].keys():
                incdicts['Send Aback'][ident] = {}
            incdicts['Send Aback'][ident][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['sourceshift']:
            if 'Source Shift' not in incdicts.keys():
                incdicts['Source Shift'] = {}
            if ident not in incdicts['Source Shift'].keys():
                incdicts['Source Shift'][ident] = {}
            incdicts['Source Shift'][ident][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['nomore']:
            if 'NOMORE' not in incdicts.keys():
                incdicts['NOMORE'] = {}
            if ident not in incdicts['NOMORE'].keys():
                incdicts['NOMORE'][ident] = {}
            incdicts['NOMORE'][ident][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['moreres']:
            if 'MOREresilience' not in incdicts.keys():
                incdicts['MOREresilience'] = {}
            if ident not in incdicts['MOREresilience'].keys():
                incdicts['MOREresilience'][ident] = {}
            incdicts['MOREresilience'][ident][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['optimal']:
            if 'Optimal' not in incdicts.keys():
                incdicts['Optimal'] = {}
            if ident not in incdicts['Optimal'].keys():
                incdicts['Optimal'][ident] = {}
            incdicts['Optimal'][ident][folder] = {key: value[0] for key, value in failhist.items()}
        elif config['ANChOR']:
            if 'ANChOR' not in incdicts.keys():
                incdicts['ANChOR'] = {}
            if ident not in incdicts['ANChOR'].keys():
                incdicts['ANChOR'][ident] = {}
            incdicts['ANChOR'][ident][folder] = {key: value[0] for key, value in failhist.items()}
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
            bmode = 'box' if box else 'regular'
            plots[protocol][steps], std[protocol][steps] = parsefail(dic[steps], plotfail=plotfail, mode=bmode)
        stepset |= {key for key in plots[protocol].keys()}
    plotlist, stdlist = {}, {}
    plist, slist = {}, {}
    for protocol in plots:
        if protocol not in plotlist.keys():
            plotlist[protocol], stdlist[protocol] = {}, {}
        for steps in plots[protocol]:
            if box:
                if steps not in plotlist[protocol].keys():
                    plotlist[protocol][steps] = []
                for fail in plots[protocol][steps].keys():
                    plotlist[protocol][steps].extend(plots[protocol][steps][fail])
            else:
                plotlist[protocol][steps] = statistics.mean(plots[protocol][steps].values())
                stdlist[protocol][steps] = statistics.mean(std[protocol][steps].values())
        if not box:
            plist[protocol] = [plotlist[protocol][key] for key in sorted(plotlist[protocol].keys())]
            slist[protocol] = [stdlist[protocol][key] for key in sorted(stdlist[protocol].keys())]
        else:
            if protocol not in plist.keys():
                plist[protocol] = []
            if mode != 'perhop':
                for step in plotlist[protocol].keys():
                    plist[protocol].extend(plotlist[protocol][step])
            else:
                plist[protocol] = plotlist[protocol]
    return sorted(stepset), plist, slist, config


# noinspection PyTypeChecker
def getopt(mainfolder=None, folders=None, plotfail='all'):
    """Get parsed values and do statistics with it."""
    if folders is None:
        quit(1)
    if mainfolder is None:
        mainfolder = ''
    globconfig = {}
    incdicts = {}
    for folder in folders:
        airtime, config = readairtime('{}/{}'.format(mainfolder, folder))
        eotx = readeotx('{}/{}'.format(mainfolder, folder))
        if airtime is None or config is None or eotx is None:
            logging.warning('Can not read log at {}! Continue'.format(folder))
            continue
        if not globconfig:
            globconfig = config
        if config['own'] and config['sourceshift']:
            if 'MORELESS' not in incdicts.keys():
                incdicts['MORELESS'] = {}
            incdicts['MORELESS'][folder] = airtime
        elif config['sourceshift'] and config['moreres']:
            if 'MORERESS' not in incdicts.keys():
                incdicts['MORERESS'] = {}
            incdicts['MORERESS'][folder] = airtime
        elif config['own']:
            if 'Send Aback' not in incdicts.keys():
                incdicts['Send Aback'] = {}
            incdicts['Send Aback'][folder] = airtime
        elif config['sourceshift']:
            if 'Source Shift' not in incdicts.keys():
                incdicts['Source Shift'] = {}
            incdicts['Source Shift'][folder] = airtime
        elif config['nomore']:
            if 'NOMORE' not in incdicts.keys():
                incdicts['NOMORE'] = {}
            incdicts['NOMORE'][folder] = airtime
        elif config['moreres']:
            if 'MOREresilience' not in incdicts.keys():
                incdicts['MOREresilience'] = {}
            incdicts['MOREresilience'][folder] = airtime
        elif config['optimal']:
            if 'Optimal' not in incdicts.keys():
                incdicts['Optimal'] = {}
            incdicts['Optimal'][folder] = airtime
        elif config['ANChOR']:
            if 'ANChor' not in incdicts.keys():
                incdicts['ANChOR'] = {}
            incdicts['ANChOR'][folder] = airtime
        else:
            if 'MORE' not in incdicts.keys():
                incdicts['MORE'] = {}
            incdicts['MORE'][folder] = airtime
    plots = {}
    for protocol, dic in incdicts.items():
        plots[protocol] = {}
        for folder, content in dic.items():
            for fail, nodes in content.items():
                if plotfail != 'all' and fail != plotfail:
                    continue
                for node, airtime in nodes.items():
                    if node not in plots[protocol].keys():
                        plots[protocol][node] = [airtime / globconfig['coding']]
                    else:
                        plots[protocol][node].append(airtime / globconfig['coding'])
    plotlist = {}
    for protocol in sorted(plots.keys()):
        plotlist[protocol] = {}
        for node in sorted(plots[protocol].keys()):
            plotlist[protocol][node] = statistics.mean(plots[protocol][node]) / len(incdicts[protocol])
    return plotlist


def parseairtime(dic, plotfail='all', mode='regular'):
    """Parse given airtime dict."""
    plot, std = {}, {}
    firstfolder = list(dic.keys())[0]
    for fail in dic[firstfolder].keys():
        if plotfail != 'all' and fail != plotfail:
            continue
        counter = []
        for folder in dic.keys():
            try:
                counter.append(sum([dic[folder][fail][node] for node in dic[folder][fail].keys()]))
            except KeyError:
                pass
        if mode == 'regular':
            plot[fail] = statistics.mean(counter)
        else:
            if fail not in plot.keys():
                plot[fail] = []
            plot[fail].extend(counter)
        if len(counter) > 1:
            std[fail] = statistics.stdev(counter)
        else:
            std[fail] = 0
    return plot, std


def parsefail(dic, plotfail='all', mode='regular'):
    """Parse given failhist dict."""
    plotlist, stdlist = {}, {}
    if len(dic) > 1:
        firstfolder = list(dic.keys())[0]
        for fail in dic[firstfolder].keys():
            if plotfail != 'all' and fail != plotfail:
                continue
            counter = []
            for folder in dic.keys():
                try:
                    counter.append(dic[folder][fail])
                except KeyError:
                    pass
            if mode == 'regular':
                plotlist[fail] = statistics.mean(counter)
            else:
                if fail not in plotlist.keys():
                    plotlist[fail] = []
                plotlist[fail].extend(counter)
            if len(counter) > 1:
                stdlist[fail] = statistics.stdev(counter)
            else:
                stdlist[fail] = 0
    else:
        plotlist = dic[list(dic.keys())[0]]
        stdlist = {key: 0 for key in plotlist.keys()}
    return plotlist, stdlist


def parseaircdf(mainfolder, folders, mode='regular', plotfail='all'):
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
            elif config['sourceshift'] and config['moreres']:
                if 'MORERESS' not in incdicts.keys():
                    incdicts['MORERESS'] = {}
                incdicts['MORERESS'][folder] = airtime
            elif config['own']:
                if 'Send Aback' not in incdicts.keys():
                    incdicts['Send Aback'] = {}
                incdicts['Send Aback'][folder] = airtime
            elif config['sourceshift']:
                if 'Source Shift' not in incdicts.keys():
                    incdicts['Source Shift'] = {}
                incdicts['Source Shift'][folder] = airtime
            elif config['nomore']:
                if 'NOMORE' not in incdicts.keys():
                    incdicts['NOMORE'] = {}
                incdicts['NOMORE'][folder] = airtime
            elif config['moreres']:
                if 'MOREresilience' not in incdicts.keys():
                    incdicts['MOREresilience'] = {}
                incdicts['MOREresilience'][folder] = airtime
            elif config['optimal']:
                if 'Optimal' not in incdicts.keys():
                    incdicts['Optimal'] = {}
                incdicts['Optimal'][folder] = airtime
            elif config['ANChOR']:
                if 'ANChOR' not in incdicts.keys():
                    incdicts['ANChOR'] = {}
                incdicts['ANChOR'][folder] = airtime
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
            if plotfail != 'all' and fail != plotfail:
                continue
            counter = []
            for folder in incdicts[protocol].keys():
                try:
                    counter.append(sum([incdicts[protocol][folder][fail][node]
                                        for node in incdicts[protocol][folder][fail].keys()]))
                except KeyError:
                    pass
            plots[protocol][fail] = counter
    if mode == 'regular':
        plotlist = {}
        for protocol in plots:
            plotlist[protocol] = []
            for fail in failure:
                if fail in plots[protocol].keys() and plots[protocol][fail]:
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


def parsefailcdf(mainfolder, folders, mode='regular', plotfail='all'):
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
            elif config['moreres'] and config['sourceshift']:
                if 'MORERESS' not in incdicts.keys():
                    incdicts['MORERESS'] = {}
                incdicts['MORERESS'][folder] = {key: value[0] for key, value in failhist.items()}
            elif config['own']:
                if 'Send Aback' not in incdicts.keys():
                    incdicts['Send Aback'] = {}
                incdicts['Send Aback'][folder] = {key: value[0] for key, value in failhist.items()}
            elif config['sourceshift']:
                if 'Source Shift' not in incdicts.keys():
                    incdicts['Source Shift'] = {}
                incdicts['Source Shift'][folder] = {key: value[0] for key, value in failhist.items()}
            elif config['nomore']:
                if 'NOMORE' not in incdicts.keys():
                    incdicts['NOMORE'] = {}
                incdicts['NOMORE'][folder] = {key: value[0] for key, value in failhist.items()}
            elif config['moreres']:
                if 'MOREresilience' not in incdicts.keys():
                    incdicts['MOREresilience'] = {}
                incdicts['MOREresilience'][folder] = {key: value[0] for key, value in failhist.items()}
            elif config['optimal']:
                if 'Optimal' not in incdicts.keys():
                    incdicts['Optimal'] = {}
                incdicts['Optimal'][folder] = {key: value[0] for key, value in failhist.items()}
            elif config['ANChOR']:
                if 'ANChOR' not in incdicts.keys():
                    incdicts['ANChOR'] = {}
                incdicts['ANChOR'][folder] = {key: value[0] for key, value in failhist.items()}
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
                if plotfail != 'all' and fail != plotfail:
                    continue
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
                if fail in plots[protocol].keys() and plots[protocol][fail]:
                    plotlist[protocol].extend(plots[protocol][fail])
                else:
                    plotlist[protocol].extend(plots[protocol]['None'])
        return plotlist
    else:
        gainlist = {}
        for protocol in plots:
            gainlist[protocol] = []
            for fail in failure:
                if fail in plots[protocol].keys() and plots[protocol][fail]:
                    proval = statistics.mean(plots[protocol][fail])
                else:
                    proval = statistics.mean(plots[protocol]['None'])
                if fail in plots['MORE'].keys() and plots['MORE'][fail]:
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
            elif config['moreres'] and config['sourceshift']:
                if 'MORERESS' not in incdicts.keys():
                    incdicts['MORERESS'] = {}
                incdicts['MORERESS'][folder] = trash
            elif config['own']:
                if 'Send Aback' not in incdicts.keys():
                    incdicts['Send Aback'] = {}
                incdicts['Send Aback'][folder] = trash
            elif config['sourceshift']:
                if 'Source Shift' not in incdicts.keys():
                    incdicts['Source Shift'] = {}
                incdicts['Source Shift'][folder] = trash
            elif config['moreres']:
                if 'MOREresilience' not in incdicts.keys():
                    incdicts['MOREresilience'] = {}
                incdicts['MOREresilience'][folder] = trash
            elif config['optimal']:
                if 'Optimal' not in incdicts.keys():
                    incdicts['Optimal'] = {}
                incdicts['Optimal'][folder] = trash
            elif config['ANChOR']:
                if 'ANChOR' not in incdicts.keys():
                    incdicts['ANChOR'] = {}
                incdicts['ANChOR'][folder] = trash
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
                        plotlist[protocol][int(timestamp)] = \
                            incdicts[protocol][folder][node][timestamp] / len(incdicts[protocol])
                    else:
                        plotlist[protocol][int(timestamp)] += \
                            incdicts[protocol][folder][node][timestamp] / len(incdicts[protocol])
    return plotlist


def readairtime(folder):
    """Read logs from folder."""
    airtime, config, failhist = None, None, None
    if os.path.exists('{}/airtime.json'.format(folder)):
        with open('{}/airtime.json'.format(folder)) as file:
            try:
                airtime = json.loads(file.read())
            except json.decoder.JSONDecodeError:
                pass
    if os.path.exists('{}/config.json'.format(folder)):
        with open('{}/config.json'.format(folder)) as file:
            try:
                config = json.loads(file.read())
                if 'david' in config.keys():
                    config['moreres'] = config['david']
                if 'newshift' in config.keys():
                    config['nomore'] = config['newshift']
            except json.decoder.JSONDecodeError:
                pass
    return airtime, config


def readeotx(folder):
    """Read logs from folder."""
    eotx = None
    if os.path.exists('{}/eotx.json'.format(folder)):
        with open('{}/eotx.json'.format(folder)) as file:
            try:
                eotx = json.loads(file.read())
            except json.decoder.JSONDecodeError:
                pass
    return eotx


def readfailhist(folder):
    """Read logs from folder."""
    failhist, config = None, None
    if os.path.exists('{}/failhist.json'.format(folder)) and os.path.exists('{}/config.json'.format(folder)):
        with open('{}/failhist.json'.format(folder)) as file:
            try:
                failhist = json.loads(file.read())
            except json.decoder.JSONDecodeError:
                pass
        with open('{}/config.json'.format(folder)) as file:
            try:
                config = json.loads(file.read())
                if 'david' in config.keys():
                    config['moreres'] = config['david']
                if 'newshift' in config.keys():
                    config['nomore'] = config['newshift']
            except json.decoder.JSONDecodeError:
                pass
        return failhist, config
    raise FileNotFoundError


def readgraph(folder):
    """Read logs from folder."""
    graph, path, eotx, failhist = None, None, None, None
    if os.path.exists('{}/path.json'.format(folder)):
        with open('{}/graph.json'.format(folder)) as file:
            try:
                graph = json.loads(file.read())
            except json.decoder.JSONDecodeError:
                pass
    if os.path.exists('{}/failhist.json'.format(folder)):
        with open('{}/path.json'.format(folder)) as file:
            try:
                path = json.loads(file.read())
            except json.decoder.JSONDecodeError:
                pass
    if os.path.exists('{}/eotx.json'.format(folder)):
        with open('{}/eotx.json'.format(folder)) as file:
            try:
                eotx = json.loads(file.read())
            except json.decoder.JSONDecodeError:
                pass
    if os.path.exists('{}/failhist.json'.format(folder)):
        with open('{}/failhist.json'.format(folder)) as file:
            try:
                failhist = json.loads(file.read())
            except json.decoder.JSONDecodeError:
                pass
    return graph, path, eotx, failhist


def readtrash(folder, mode='real'):
    """Read trash log."""
    trash, config = None, None
    if os.path.exists('{}/{}trash.json'.format(folder, mode)):
        with open('{}/{}trash.json'.format(folder, mode)) as file:
            try:
                trash = json.loads(file.read())
            except json.decoder.JSONDecodeError:
                pass
    if os.path.exists('{}/config.json'.format(folder, mode)):
        with open('{}/config.json'.format(folder, mode)) as file:
            try:
                config = json.loads(file.read())
                if 'david' in config.keys():
                    config['moreres'] = config['david']
                if 'newshift' in config.keys():
                    config['nomore'] = config['newshift']
            except json.decoder.JSONDecodeError:
                pass
    if trash is None or config is None:
        raise FileNotFoundError
    else:
        return trash, config


def plotairtime(mainfolder=None, folders=None):
    """Plot airtime for different failures."""
    # p.figure(figsize=(8.45, 6.18))
    p.figure()
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
    p.title('Needed transmissions over failures for different routing schemes')
    ind = [x + width / 2 for x in range(len(failures))]
    p.xticks(ind, labels=failures, rotation=90)
    p.grid(True)
    p.tight_layout()
    p.savefig('{}/airfailbar.pdf'.format(mainfolder))
    p.close()


def plotaircdf(mainfolder=None, folders=None, plotfail='all'):
    """Plot airtime CDF."""
    if folders is None and mainfolder is not None:
        folders = []
        cmain = [folder for folder in os.listdir(mainfolder) if os.path.isdir('{}/{}'.format(mainfolder, folder))]
        for subfolder in cmain:
            folders.extend(['{}/{}'.format(subfolder, subsubfolder)
                            for subsubfolder in os.listdir('{}/{}'.format(mainfolder, subfolder))
                            if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder)) and
                            subsubfolder != 'test'])
    if mainfolder is None:
        mainfolder = ''
    plotlist = parseaircdf(mainfolder, folders, plotfail=plotfail)
    for mode in ['regular', 'close']:
        # p.figure(figsize=(6.18, 4.2))
        # p.figure(figsize=(4.2, 6))
        p.figure()
        dashes = {'Optimal': [4, 2], 'ANChOR': [2, 2], 'MORE': [4, 1, 1, 1], 'Source Shift': [3, 1],
                  'MOREresilience': [1, 1, 2, 1]}
        cmap = p.cm.get_cmap('tab10')
        clist = [cmap(j) for j in range(len(plotlist.keys()))]
        colors = {name: color for name, color in zip(sorted(plotlist.keys()), clist)}
        protocols = [x for x in ['ANChOR', 'Source Shift', 'Optimal', 'MOREresilience', 'MORE'] if x in plotlist.keys()]
        for protocol in protocols:
            # if protocol in ['ANChOR', 'MORE', 'MORELESS', 'MOREresilience', 'NOMORE', 'Optimal']:
            if protocol in ['ANChOR', 'MORE', 'MOREresilience']:
                line = p.plot(sorted(plotlist[protocol]), np.linspace(0, 1, len(plotlist[protocol])), '--',
                              color=colors[protocol], label=protocol, alpha=0.8)
            elif protocol in ['Source Shift']:
                line = p.plot(sorted(plotlist[protocol]), np.linspace(0, 1, len(plotlist[protocol])), '--',
                              color=colors[protocol], label='SourceShift', alpha=0.8)
            elif protocol in ['Optimal']:
                line = p.plot(sorted(plotlist[protocol]), np.linspace(0, 1, len(plotlist[protocol])), '--',
                              color=colors[protocol], label='AMORE', alpha=0.8)
            line[0].set_dashes(dashes[protocol])
        # if plotfail == 'all':
        #    p.title('Airtime per routing scheme')
        # else:
        #    p.title('Airtime for {} failure'.format(plotfail))
        p.ylabel('Fraction')
        p.xlabel('Airtime in transmissions')
        # p.ylim([0.8, 1])
        p.xlim([40, 100000])
        # p.xlim([40, 1000])
        p.xscale('log')
        # p.xlim([40, 200])
        p.grid(True, which='both')
        # p.minorticks_off()
        # p.xticks(range(40, 200, 50), range(40, 200, 50), rotation=90)
        p.legend(loc='lower right')
        p.tight_layout()
        if mode == 'regular':
            p.ylim([0, 1])
            p.savefig('{}/air{}cdf.pdf'.format(mainfolder, plotfail))
        else:
            p.ylim([0.9, 1])
            p.savefig('{}/air{}closecdf.pdf'.format(mainfolder, plotfail))
        p.close()


def plotlatcdf(mainfolder=None, folders=None, plotfail='all'):
    """Plot latency CDF."""
    if folders is None and mainfolder is not None:
        folders = []
        cmain = [folder for folder in os.listdir(mainfolder) if os.path.isdir('{}/{}'.format(mainfolder, folder))]
        for subfolder in cmain:
            folders.extend(['{}/{}'.format(subfolder, subsubfolder)
                            for subsubfolder in os.listdir('{}/{}'.format(mainfolder, subfolder))
                            if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder)) and
                            subsubfolder != 'test'])
    if mainfolder is None:
        mainfolder = ''
    plotlist = parsefailcdf(mainfolder, folders, plotfail=plotfail)
    for mode in ['regular', 'close']:
        # p.figure(figsize=(6.18, 4.2))
        # p.figure(figsize=(4.2, 6))
        p.figure()
        for protocol in sorted(plotlist.keys()):
            # if protocol in ['ANChOR', 'MORE', 'MORELESS', 'MOREresilience', 'NOMORE', 'Optimal']:
            if protocol in ['ANChOR', 'MORE', 'MOREresilience', 'Source Shift', 'Optimal']:
                p.plot(sorted(plotlist[protocol]), np.linspace(0, 1, len(plotlist[protocol])), label=protocol,
                       alpha=0.8)
        # p.axvline(x=5000, linestyle='--', color='black')
        # if plotfail == 'all':
        #    p.title('Latency per routing scheme')
        # else:
        #    p.title('Latency for {} failure'.format(plotfail))
        p.ylabel('Fraction')
        p.xlabel('Latency in time slots')
        # p.ylim([0.8, 1])
        p.xscale('log')
        # p.xlim([10, 100000])
        # p.xlim([40, 2000])
        # p.xlim([40, 140])
        # p.yticks(rotation=90)
        p.grid(True)
        # p.minorticks_off()
        # p.xticks(range(40, 140, 20), range(40, 140, 20), rotation=90)
        p.tight_layout()
        if mode == 'regular':
            # p.legend(loc='lower right')
            p.legend(loc='best')
            p.ylim([0, 1])
            p.savefig('{}/lat{}cdf.pdf'.format(mainfolder, plotfail))
        else:
            # p.legend(loc='lower left')
            p.legend(loc='best')
            p.ylim([0.9, 1])
            p.savefig('{}/lat{}closecdf.pdf'.format(mainfolder, plotfail))
        p.close()


def plotfailhist(mainfolder=None, folders=None):
    """Plot time to finish transmission diagram to compare different simulations."""
    failures, plotlist, stdlist, config = getfailhist(mainfolder, folders)
    width = 0.9 / len(plotlist.keys())
    # p.figure(figsize=(8.45, 6.18))
    p.figure()
    for protocol in sorted(plotlist.keys()):
        ind = [x + list(sorted(plotlist.keys())).index(protocol) * width for x in range(len(failures))]
        p.bar(ind, plotlist[protocol], width=width, label=protocol, yerr=stdlist[protocol],
              error_kw={'elinewidth': width / 3})
    p.legend(loc='best')
    p.yscale('log')
    p.ylabel('Latency in timeslots')
    p.xlabel('Failure')
    if config['maxduration']:
        p.ylim(top=config['maxduration'])
    p.xlim([-1, len(failures)])
    p.title('Timeslots over failures for different routing schemes')
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
                    return  # Could happen in case of uncomplete logs like when plotting while simulating
        # p.figure(figsize=(5, 5))
        p.figure()
        width = 0.9 / len(plotlist.keys())
        # for protocol in gainlist:
        #    ind = [x + list(sorted(plotlist.keys())).index(protocol) * width for x in range(len(failures))]
        p.bar(range(len(gainlist.keys())), gainlist.values(), width=width)
        # p.legend(loc='best')
        # if mode == 'latency':
        #    p.title('Latency compared to MORE')
        # else:
        #    p.title('Airtime compared to MORE')
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
    """Plot gain in latency for airtime as cdf."""
    if mainfolder is None:
        return
    folders = []
    cmain = [folder for folder in os.listdir(mainfolder) if os.path.isdir('{}/{}'.format(mainfolder, folder))]
    for subfolder in cmain:
        folders.extend(['{}/{}'.format(subfolder, subsubfolder)
                        for subsubfolder in os.listdir('{}/{}'.format(mainfolder, subfolder))
                        if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder)) and
                        subsubfolder != 'test'])
    for mode in ['airtime', 'latency']:
        if mode == 'latency':
            gainlist = parsefailcdf(mainfolder, folders, mode='gain')
        else:
            gainlist = parseaircdf(mainfolder, folders, mode='gain')
        # p.figure(figsize=(6.18, 4.2))
        p.figure()
        dashes = {'Optimal': [4, 2], 'ANChOR': [2, 2], 'MORE': [4, 1, 1, 1], 'Source Shift': [3, 1],
                  'MOREresilience': [1, 1, 2, 1]}
        cmap = p.cm.get_cmap('tab10')
        clist = [cmap(j) for j in range(len(gainlist.keys()))]
        colors = {name: color for name, color in zip(sorted(gainlist.keys()), clist)}
        protocols = [x for x in ['Source Shift', 'Optimal', 'ANChOR', 'MOREresilience', 'MORE'] if x in gainlist.keys()]
        for protocol in protocols:
            if protocol in ['ANChOR', 'MORE', 'MOREresilience', 'Source Shift']:
                line = p.plot(sorted(gainlist[protocol]), np.linspace(0, 1, len(gainlist[protocol])), '--',
                              color=colors[protocol], label=protocol, alpha=0.8)
            elif protocol in ['Optimal']:
                line = p.plot(sorted(gainlist[protocol]), np.linspace(0, 1, len(gainlist[protocol])), '--',
                              color=colors[protocol], label='SorceShift', alpha=0.8)
            elif protocol in ['Optimal']:
                line = p.plot(sorted(gainlist[protocol]), np.linspace(0, 1, len(gainlist[protocol])), '--',
                              color=colors[protocol], label='AMORE', alpha=0.8)
            line[0].set_dashes(dashes[protocol])
        # if mode == 'latency':
        #    p.title('Difference in latency compared to MORE')
        # else:
        #    p.title('Difference in airtime compared to MORE')
        p.ylabel('Fraction')
        p.xlabel('Difference in percent')
        p.ylim([0, 1])
        # p.ylim([0.8, 1])
        p.xlim([-100, 200])
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
    # p.figure(figsize=(6.18, 6.18))  # Plot graph \textwidth
    p.figure()  # Plot graph to use in subfigure 0.5 * \textwidth
    for folder in folders:
        graph, path, eotx, failhist = readgraph(folder)
        if graph is None or path is None or eotx is None or failhist is None:
            logging.warning('Can not read log at {}! Continue'.format(folder))
            continue
        pos = {node: [graph['nodes'][node]['x'], graph['nodes'][node]['y']] for node in graph['nodes']}
        configlist = [(edge['nodes'][0], edge['nodes'][1], edge['loss']) for edge in graph['links']]
        net = nx.Graph()
        net.add_weighted_edges_from(configlist)
        prevfail = None
        for fail in failhist.keys():
            if fail in eotx.keys():
                for node in net.nodes:  # Update EOTX to the one of the current failure if there is such
                    net.nodes[node]['EOTX'] = eotx[fail][node]
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
            alphaval = 1 / max([value for value in path[fail].values()])
            for edge in path[fail].keys():
                nx.draw_networkx_edges(net, pos=pos, edgelist=[(edge[0], edge[1])], width=5,
                                       # nx.draw_networkx_edges(net, pos=pos, edgelist=[(edge[0], edge[1])], width=8,
                                       alpha=path[fail][edge] * alphaval, edge_color='purple')
            p.savefig('{}/graphfail{}.pdf'.format(folder, fail))
            p.clf()
    p.close()


def plotopt(mainfolder=None, plotfail='all'):
    """Plot achieved airtime and optimal one per node."""
    if mainfolder is None:
        return
    folders = []
    cmain = [folder for folder in os.listdir(mainfolder) if os.path.isdir('{}/{}'.format(mainfolder, folder))]
    for subfolder in cmain:
        folders.extend(['{}/{}'.format(subfolder, subsubfolder)
                        for subsubfolder in os.listdir('{}/{}'.format(mainfolder, subfolder))
                        if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder)) and
                        subsubfolder != 'test'])
    plotlist = getopt(mainfolder, folders, plotfail=plotfail)
    width = 0.9 / len(plotlist.keys())
    # p.figure(figsize=(20, 8.4))
    p.figure()
    for protocol in plotlist.keys():
        ind = [x + list(sorted(plotlist.keys())).index(protocol) * width for x in range(len(plotlist[protocol]))]
        p.bar(ind, plotlist[protocol].values(), width=width, label=protocol)
    p.legend(loc='best')
    p.yscale('log')
    p.ylabel('Airtime per node')
    p.xlabel('Nodes')
    ind = [x + 0.5 for x in range(len(list(plotlist[list(plotlist.keys())[0]])))]
    p.xticks(ind, labels=list(plotlist[list(plotlist.keys())[0]].keys()))
    p.title('Airtime over nodes, optimal value with failure knowledge and achieved values')
    p.grid(True)
    p.tight_layout()
    p.savefig('{}/airopt{}bar.pdf'.format(mainfolder, plotfail))
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
                        if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder)) and
                        subsubfolder != 'test'])
    for mode in ['Airtime', 'Latency']:
        if mode == 'Latency':
            steps, plotlist, stdlist, config = getfailhistmode(mainfolder, folders, mode=kind, box=True)
        else:
            steps, plotlist, stdlist = getairtimemode(mainfolder, folders, mode=kind, box=True)
        if not plotlist:
            continue
        cmap = p.cm.get_cmap('tab10')
        for fliers in [True]:
            # p.figure(figsize=(8.45, 6.18))
            p.figure()
            newlist = {}
            hoplist = set()
            for protocol in plotlist.keys():
                if protocol in ['ANChOR', 'MORE', 'MOREresilience', 'Source Shift', 'Optimal']:
                    if hoplist:
                        hoplist &= set(plotlist[protocol].keys())
                    else:
                        hoplist = set(plotlist[protocol].keys())
            for protocol in sorted(plotlist.keys()):
                if protocol in ['ANChOR', 'MORE', 'MOREresilience', 'Source Shift', 'Optimal']:
                    for hop in sorted(plotlist[protocol].keys()):
                        if hop in hoplist:
                            if protocol not in newlist.keys():
                                newlist[protocol] = []
                            newlist[protocol].append(plotlist[protocol][hop])
            boxlist = []
            poslist = [x * 7 + 1 for x in range(10)]
            for protocol in newlist.keys():
                boxlist.append(p.boxplot(newlist[protocol], positions=poslist, showfliers=fliers))
                poslist = [x + 1 for x in poslist]
            for bp in boxlist:
                for l, box in enumerate(bp['boxes']):
                    box.set(color=cmap(l))
                for l, median in enumerate(bp['medians']):
                    median.set(color=cmap(l))
                for l, flier in enumerate(bp['fliers']):
                    flier.set(marker='x', color=cmap(l), alpha=0.3)
            p.yscale('log')
            p.legend([box["boxes"][0] for box in boxlist],
                     ['ANChOR', 'MORE', 'MORELESS', 'MOREresilience', 'NOMORE', 'Optimal'], loc='best')
            if mode == 'Airtime':
                p.ylabel('Airtime in transmissions')
            else:
                p.ylabel('Latency in timeslots')
            if kind == 'perhop':
                p.xlabel('Minimum amount of hops between Source and Destination')
            else:
                p.xlabel('Mincut')
            # p.ylim(bottom=0)
            p.xlim([0, 70])
            p.title('{} over different networks and routing schemes'.format(mode))
            # ind = [x + 0.5 for x in range(len(steps))]
            # p.xticks(ind, labels=steps)
            p.grid(True)
            p.tight_layout()
            name = 'hop' if kind == 'perhop' else 'mcut'
            if mode == 'Airtime':
                p.savefig('{}/air{}bar.pdf'.format(mainfolder, name))
            else:
                p.savefig('{}/lat{}bar.pdf'.format(mainfolder, name))
            p.close()


def plotbox(mainfolder=None):
    """Do boxplots."""
    if mainfolder is None:
        return
    folders = []
    cmain = [folder for folder in os.listdir(mainfolder) if os.path.isdir('{}/{}'.format(mainfolder, folder))]
    for subfolder in cmain:
        folders.extend(['{}/{}'.format(subfolder, subsubfolder)
                        for subsubfolder in os.listdir('{}/{}'.format(mainfolder, subfolder))
                        if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder)) and
                        subsubfolder != 'test'])
    # p.figure(figsize=(8.45, 6.18))
    p.figure()
    for mode in ['Latency', 'Airtime']:
        if mode == 'Latency':
            steps, plotlist, stdlist, config = getfailhistmode(mainfolder, folders, mode='all', box=True)
        else:
            steps, plotlist, stdlist = getairtimemode(mainfolder, folders, mode='all', box=True)
        if not plotlist:
            continue
        cmap = p.cm.get_cmap('tab10')
        for fliers in [True, False]:
            bp = p.boxplot([plotlist[protocol] for protocol in sorted(plotlist.keys()) if
                            protocol in ['ANChOR', 'MORE', 'MOREresilience', 'Source Shift', 'Optimal']],
                           showfliers=fliers)
            for l, box in enumerate(bp['boxes']):
                box.set(color=cmap(l))
            for l, median in enumerate(bp['medians']):
                median.set(color=cmap(l))
            for l, flier in enumerate(bp['fliers']):
                flier.set(marker='x', color=cmap(l), alpha=0.3)
            p.yscale('log')
            if mode == 'Airtime':
                p.ylabel('Airtime in transmissions')
            else:
                p.ylabel('Latency in time slots')
            # p.title('{} over routing schemes'.format(mode))
            p.xticks(range(1, 6 + 1),
                     labels=sorted(['ANChOR', 'MORE', 'MOREresilience', 'Source Shift', 'Optimal']), rotation=45)
            # if fliers:
            #    p.ylim(top=1000)
            #    p.yticks(range(100, 1100, 100), range(100, 1100, 100))
            # else:
            #    p.ylim(top=400)
            #    p.yticks(range(100, 450, 50), range(100, 450, 50))
            p.grid(True)
            # p.minorticks_off()
            # p.ylim(top=100000)
            p.tight_layout()
            if mode == 'Airtime':
                if fliers:
                    p.savefig('{}/airflierbox.pdf'.format(mainfolder))
                else:
                    p.savefig('{}/airbox.pdf'.format(mainfolder))
            else:
                if fliers:
                    p.savefig('{}/latflierbox.pdf'.format(mainfolder))
                else:
                    p.savefig('{}/latbox.pdf'.format(mainfolder))
            p.clf()
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
                            if os.path.isdir('{}/{}/{}'.format(x, subfolder, subsubfolder)) and subsubfolder != 'test'])
        plotlist.append(parsefailcdf(x, folders))

    # p.figure(figsize=(4.2, 6.18))
    for protocol in sorted(plotlist[0].keys()):
        # result = stats.mannwhitneyu(plotlist[0][protocol], plotlist[1][protocol])
        # result = stats.ranksums(plotlist[0][protocol], plotlist[1][protocol])
        logging.info('KS 2Samp for {} is {}'.format(protocol,  # dist unequal if p < 1% | equal if p > 10%
                                                    stats.anderson_ksamp(
                                                        [plotlist[0][protocol], plotlist[1][protocol]])))
        # p.plot(sorted(plotlist[0][protocol]), np.linspace(0, 1, len(plotlist[0][protocol])),
        #       label=protocol, alpha=0.8)
    # p.ylabel('Fraction')
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
                        if os.path.isdir('{}/{}/{}'.format(mainfolder, subfolder, subsubfolder)) and
                        subsubfolder != 'test'])
    for mode in ['real', 'overhearing']:
        try:
            plotlist = parsetrash(mainfolder, folders, mode=mode)
        except AttributeError:
            logging.warning('Old log format in {}'.format(mainfolder))
            return
        # p.figure(figsize=(6.18, 6.18))
        p.figure()
        for protocol in sorted(plotlist.keys()):
            p.plot(list(plotlist[protocol].keys()), list(plotlist[protocol].values()), label=protocol, alpha=0.8)
        p.title('Linear dependent packets per routing scheme')
        p.ylabel('Total amount of linear dependend packets per time slots')
        p.xlabel('Time slot')
        # p.xlim(left=0)
        # p.ylim(bottom=0)
        p.yscale('log')
        p.xscale('log')
        p.grid(True)
        p.legend(loc='best')
        p.tight_layout()
        p.savefig('{}/{}trash.pdf'.format(mainfolder, mode))
        p.close()


if __name__ == '__main__':
    setup()
    logging.basicConfig(filename='plotlog.log', level=logging.INFO, filemode='w')
    now = datetime.datetime.now()
    # date = str(int(str(now.year) + str(now.month) + str(now.day)))
    date = '../2019223sa1'
    folderlist = []
    for i in range(50):
        folderlst = []
        # folderlist.append('{}/graph{}/test'.format(date, i))
        folderlist.extend(['{}/graph{}/test{}'.format(date, i, j) for j in range(10)])
        mfolder = '{}/graph{}'.format(date, i)
        # folderlst.append('test'.format(i))
        folderlst.extend(['test{}'.format(j) for j in range(10)])
        # plotaircdf('{}/graph{}'.format(date, i), folderlst)
        # plotlatcdf('{}/graph{}'.format(date, i), folderlst)
        # plotairtime(mfolder, folderlst)
        # plotgain(mfolder, folderlst)
        # plotfailhist(mfolder, folderlst)
    #    plotopt(date)
    #    plotopt(date, plotfail='None')
    plotaircdf(date, plotfail='None')
    plotlatcdf(date, plotfail='None')
    plotaircdf(date)
    plotlatcdf(date)
    plotgaincdf(date)
    # plotperhop(date)
    # plotperhop(date, kind='mcut')
    # plotqq(['../2019212', '../2019212a'])
    # plottrash(date)
    # plotgraph(folders=folderlist)
    # plotgraph(date)
    plotbox(date)
