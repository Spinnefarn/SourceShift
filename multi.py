#!/usr/bin/python
# coding=utf-8
"""Main script, this will supervise the simulation."""
import argparse
import json
import datetime
import os
from Simulator import Simulator
import plotter
import random
import logging
import time
from multiprocessing import Process, cpu_count, active_children


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
                        default=(20, 0.3))
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
    parser.add_argument('-s', '--sourceshift',
                        dest='sourceshift',
                        type=bool,
                        help='Enable source shifting',
                        default=False)
    parser.add_argument('-m', '--multiprocessing',
                        dest='multiprocessing',
                        type=bool,
                        help='Turn on multiprocessing, default on.',
                        default=False)
    parser.add_argument('-fe', '--failedge',
                        dest='failedge',
                        type=str,
                        nargs=2,
                        help='Which edge should fail?',
                        default=None)
    parser.add_argument('-fn', '--failnode',
                        dest='failnode',
                        type=str,
                        help='Which node should fail?.',
                        default=None)
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
    parser.add_argument('-r', '--random',
                        dest='random',
                        type=int,
                        help='Specify a seed to reduce randomness.',
                        default=None)
    return parser.parse_args()


def runsim(config):
    """Run simulation based on arguments."""
    randomseed = random.randint(0, 1000000)
    sim = Simulator(jsonfile=config['json'], coding=config['coding'], fieldsize=config['fieldsize'],
                    sendall=config['sendam'], own=config['own'], edgefail=config['failedge'],
                    nodefail=config['failnode'], allfail=config['failall'], randcof=config['randconf'],
                    folder=config['folder'], maxduration=config['maxduration'], randomseed=randomseed,
                    sourceshift=config['sourceshift'])
    logging.info('Start simulator {}'.format(config['folder']))
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
    with open('{}/config.json'.format(config['folder']), 'w') as f:
        json.dump(config, f)


def launchsubp(config):
    """Launch subprocess."""
    p = Process(target=runsim, args=(config,))
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


def setmode(config, number):
    """Set config mode."""
    mode = ['m', 'o', 'ss', 'oss', 'david']
    try:
        number = int(number) % len(mode)
    except (TypeError, ValueError):
        number = 0
    if mode[number] == 'm':
        config['own'], config['sourceshift'], config['david'] = False, False, 0.0
    elif mode[number] == 'o':
        config['own'], config['sourceshift'], config['david'] = True, False, 0.0
    elif mode[number] == 'ss':
        config['own'], config['sourceshift'], config['david'] = False, True, 0.0
    elif mode[number] == 'oss':
        config['own'], config['sourceshift'], config['david'] = True, True, 0.0
    elif mode[number] == 'david':
        config['own'], config['sourceshift'], config['david'] = False, False, 1.0
    return config


def plotall(mfolder, counter, liste):
    """Create a process to do the plots."""
    plotter.plotgraph(['{0}/graph{1}/test'.format(mfolder, counter)])
    plotter.plotgraph(['{0}/graph{1}/{2}'.format(mfolder, counter, folder) for folder in liste])
    plotter.plotairtime('{0}/graph{1}'.format(mfolder, counter), liste)
    plotter.plotfailhist('{0}/graph{1}'.format(mfolder, counter), liste)
    plotter.plotgain('{0}/graph{1}'.format(mfolder, counter), liste)


if __name__ == '__main__':
    args = parse_args()
    llevel = logging.INFO
    logging.basicConfig(
        filename='main.log', level=llevel, format='%(asctime)s %(processName)s\t %(levelname)s\t %(message)s',
        filemode='w')
    now = datetime.datetime.now()
    date = str(now.year) + str(now.month) + str(now.day)
    plot = None
    for i in range(5):
        logging.info('Created new graph at graph{}'.format(i))
        confdict = {'json': args.json, 'randconf': args.amount, 'coding': args.coding, 'fieldsize': args.fieldsize,
                    'sendam': args.sendam, 'own': args.own, 'failedge': args.failedge, 'failnode': args.failnode,
                    'failall': False, 'folder': '{}/graph{}/test'.format(date, i), 'maxduration': args.maxduration,
                    'random': args.random, 'sourceshift': args.sourceshift}
        cleanfolder(confdict['folder'])
        runsim(confdict)
        with open('{}/graph{}/test/failhist.json'.format(date, i)) as file:
            failhist = json.loads(file.read())
        if args.random is None:
            randomnumber = random.randint(0, 1000000)
        else:
            randomnumber = args.random
        random.seed(randomnumber)
        confdict = {'json': args.json, 'randconf': args.amount, 'coding': args.coding, 'fieldsize': args.fieldsize,
                    'sendam': args.sendam, 'own': args.own, 'failedge': args.failedge, 'failnode': args.failnode,
                    'failall': True, 'folder': args.folder, 'maxduration': args.maxduration,
                    'random': randomnumber, 'sourceshift': args.sourceshift}
        logging.info('Randomseed = ' + str(randomnumber))
        folderlist = ['test{}'.format(i) for i in range(40)]
        processes = []
        try:
            for element in folderlist:
                cleanfolder('{}/graph{}/{}'.format(date, i, element))
                confdict['json'] = '{}/graph{}/test/graph.json'.format(date, i)
                confdict['folder'] = '{}/graph{}/{}'.format(date, i, element)
                confdict = setmode(confdict, element[-1])
                confdict['maxduration'] = 20 * failhist['None'][0]
                while True:
                    if cpu_count() > len(active_children()):
                        launchsubp(confdict)
                        break
                    else:
                        time.sleep(1)
        except KeyboardInterrupt:
            pass
        for process in processes:
            process.join()
        if plot is not None and plot.is_alive():
            plot.join()
        plot = Process(target=plotall, args=(date, i, folderlist))
        plot.start()
    if plot is not None and plot.is_alive():
        plot.join()
    logging.info('Everything done')
