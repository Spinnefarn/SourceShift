#!/usr/bin/python
# coding=utf-8
"""Main script, this will supervise the simulation."""
import json
import datetime
import os
from Simulator import Simulator
import plotter
import random
import logging
import time
from multiprocessing import Process, cpu_count, active_children


def runsim(sim=None):
    """Run simulation."""
    if sim is None:
        return
    starttime = time.time()
    complete = False
    while not complete:
        beginbatch = time.time()
        done = False
        while not done:
            done = sim.update()
        logging.info('{:3.0f} Seconds needed'.format(time.time() - beginbatch))
        complete = sim.newbatch()
    sim.writelogs()
    logging.info('{:3.0f} Seconds needed in total.'.format(time.time() - starttime))


def startsim(config):
    """Manage simulation based on arguments."""
    randomseed = random.randint(0, 1000000)
    sim = Simulator(jsonfile=config['json'], coding=config['coding'], fieldsize=config['fieldsize'],
                    sendall=config['sendam'], own=config['own'], edgefail=config['failedge'],
                    nodefail=config['failnode'], allfail=config['failall'], randcof=config['randconf'],
                    folder=config['folder'], maxduration=config['maxduration'], randomseed=randomseed,
                    sourceshift=config['sourceshift'], nomore=config['nomore'], moreres=config['moreres'],
                    hops=config['hops'], optimal=config['optimal'], anchor=config['anchor'])
    logging.info('Start simulator {}'.format(config['folder']))
    runsim(sim=sim)


def launchsubp(config):
    """Launch subprocess."""
    p = Process(target=startsim, args=(config,))
    p.start()
    logging.info('Startet simulation process with {}'.format(config))
    processes.append(p)


def cleanfolder(folder):
    """Make sure folder exist and is empty."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif folder != '.':
        filelist = [f for f in os.listdir(folder)]
        for f in filelist:
            os.remove(os.path.join(folder, f))


def setmode(config, count):
    """Set config mode."""
    # mode = ['oss', 'o', 'no', 'ss', 'moreresss', 'moreres', 'm', 'opt', 'anchor']
    mode = ['ss', 'moreres', 'm', 'opt', 'anchor']
    try:
        count = int(count) % len(mode)
    except (TypeError, ValueError):
        count = 0
    config['own'], config['sourceshift'], config['nomore'], config['moreres'], config['optimal'], config['anchor'] = \
        False, False, False, 0.0, False, False
    if mode[count] == 'm':
        pass
    elif mode[count] == 'o':
        config['own'] = True
    elif mode[count] == 'ss':
        config['sourceshift'] = True
    elif mode[count] == 'no':
        config['nomore'] = True
    elif mode[count] == 'oss':
        config['own'], config['sourceshift'] = True, True
    elif mode[count] == 'moreresss':
        config['sourceshift'], config['moreres'] = True, 1.0
    elif mode[count] == 'moreres':
        config['moreres'] = 1.0
    elif mode[count] == 'opt':
        config['optimal'] = True
    elif mode[count] == 'anchor':
        config['anchor'] = True
    return config


def plotall(mfolder, counter , liste):
    """Create a process to do the plots."""
    # plotter.plotairtime('{0}/graph{1}'.format(mfolder, counter), liste)
    # plotter.plotfailhist('{0}/graph{1}'.format(mfolder, counter), liste)
    # plotter.plotgain('{0}/graph{1}'.format(mfolder, counter), liste)
    # plotter.plotaircdf('{0}/graph{1}'.format(mfolder, counter), liste)
    # plotter.plotlatcdf('{0}/graph{1}'.format(mfolder, counter), liste)
    plotter.plotaircdf(mfolder)
    plotter.plotlatcdf(mfolder)
    plotter.plotaircdf(mfolder, plotfail='None')
    plotter.plotlatcdf(mfolder, plotfail='None')
    # plotter.plotperhop(mfolder)
    # plotter.plotperhop(mfolder, kind='mcut')
    # plotter.plottrash(mfolder)
    plotter.plotbox(mfolder)
    plotter.plotgraph(['{0}/graph{1}/test'.format(mfolder, counter)])  # Just plot each graph once
    # plotter.plotgraph(['{0}/graph{1}/{2}'.format(mfolder, counter, folder) for folder in liste[:8]])


if __name__ == '__main__':
    if os.path.exists('main.log'):
        os.remove('main.log')
    llevel = logging.INFO
    logging.basicConfig(
        filename='main.log',
        level=llevel,
        format='%(asctime)s %(processName)s\t %(levelname)s\t %(message)s',
        filemode='w')
    now = datetime.datetime.now()
    date = str(now.year) + str(now.month) + str(now.day)
    plot, plotconf = None, None
    randomnumber = None
    processes = []
    for i in range(3):
        logging.info('Created new graph at graph{}'.format(i))
        confdict = {'json': None, 'randconf': (20, 0.3), 'coding': 42, 'fieldsize': 16,
                    'sendam': 0, 'own': False, 'failedge': None, 'failnode': None,
                    'failall': False, 'folder': '{}/graph{}/test'.format(date, i), 'maxduration': 0,
                    'random': None, 'sourceshift': False, 'nomore': False,
                    'moreres': 0.0, 'hops': (i % 9) + 1, 'optimal': False, 'anchor': False}
        cleanfolder(confdict['folder'])
        launchsubp(confdict)
        while not os.path.exists(confdict['folder'] + '/graph.json'):
            time.sleep(0.01)
        with open('{}/graph{}/test/failhist.json'.format(date, i)) as file:
            failhist = json.loads(file.read())
        if randomnumber is None:
            randomnumber = random.randint(0, 1000000)
        random.seed(randomnumber)
        logging.info('Randomseed = ' + str(randomnumber))
        folderlist = ['test{}'.format(i) for i in range(63)]     # Should be much bigger than number of available cores
        try:
            for element in folderlist:
                cleanfolder('{}/graph{}/{}'.format(date, i, element))
                confdict['json'] = '{}/graph{}/test/graph.json'.format(date, i)
                confdict['folder'] = '{}/graph{}/{}'.format(date, i, element)
                try:
                    x = int(element[-2:])
                except ValueError:
                    x = int(element[-1])
                confdict = setmode(confdict, x)
                confdict['maxduration'] = 100000
                while True:
                    if cpu_count() > len(active_children()):
                        try:
                            launchsubp(confdict)
                            break
                        except OSError:
                            pass
                    time.sleep(1)
        except KeyboardInterrupt:
            print('Got KeyboardInterrupt!')
            break
        dellist = []
        for j in range(len(processes)):
            if not processes[j].is_alive():
                dellist.append(i)
        while dellist:
            try:
                number = dellist.pop()
                del processes[number]
            except IndexError:
                break
        if plot is not None and plot.is_alive():
            plot.join()
        if plotconf is not None:
            plot = Process(target=plotall, args=plotconf)   # Start next graph bevore plotting old one
            plot.start()                                    # No need for last process to finish
        plotconf = date, i, folderlist
    if plot is not None and plot.is_alive():
        plot.join()
    logging.info('Waiting for last plots.')
    if plotconf is not None:
        plot = Process(target=plotall, args=plotconf)
        plot.start()
        plot.join()
    logging.info('Everything done')
