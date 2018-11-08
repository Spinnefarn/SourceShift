#!/usr/bin/python
# coding=utf-8
"""Main script, this will supervise the simulation."""
import argparse
import json
import time
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
    parser.add_argument('-m', '--multiprocessing',
                        dest='multiprocessing',
                        type=bool,
                        help='Turn on multiprocessing, default on.',
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
    parser.add_argument('-r', '--random',
                        dest='random',
                        type=int,
                        help='Specify a seed to reduce randomness.',
                        default=None)
    return parser.parse_args()


def runsim(config):
    """Run simulation based on arguments."""
    sim = Simulator(jsonfile=config['json'], coding=config['coding'], fieldsize=config['fieldsize'],
                    sendall=config['sendam'], own=config['own'], edgefail=config['failedge'],
                    nodefail=config['failnode'], allfail=config['failall'], randcof=config['randconf'],
                    folder=config['folder'], maxduration=config['maxduration'], randomseed=config['random'])
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
    sim.writelogs()
    with open('{}/config.json'.format(config['folder']), 'w') as file:
        json.dump(config, file)


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
        filelist = [file for file in os.listdir(folder)]
        for file in filelist:
            os.remove(os.path.join(folder, file))


if __name__ == '__main__':
    args = parse_args()
    if args.random is None:
        randomnumber = random.randint(0, 1000000)
    else:
        randomnumber = args.random
    random.seed(randomnumber)
    confdict = {'json': args.json, 'randconf': args.amount, 'coding': args.coding, 'fieldsize': args.fieldsize,
                'sendam': args.sendam, 'own': args.own, 'failedge': args.failedge, 'failnode': args.failnode,
                'failall': args.failall, 'folder': args.folder, 'maxduration': args.maxduration, 'random': randomnumber}
    llevel = logging.INFO
    logging.basicConfig(
        filename='main.log', level=llevel, format='%(asctime)s %(levelname)s\t %(message)s',
        filemode='w')
    logging.info('Randomseed = ' + str(randomnumber))
    now = datetime.datetime.now()
    date = int(str(now.year) + str(now.month) + str(now.day))
    folderlist = ['{}/test{}'.format(date, i) for i in range(2000)]
    processes = []
    try:
        for element in folderlist:
            cleanfolder(element)
            confdict['folder'] = element
            confdict['own'] = not confdict['own']
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
    plotter.plotfailhist(date, folderlist)
    logging.info('Everything done')
