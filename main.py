#!/usr/bin/python
# coding=utf-8
"""Main script, this will supervise the simulation."""
import argparse
import json
import time
import os
from Simulator import Simulator
import random


def parse_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json',
                        dest='json',
                        type=str,
                        help='should contain network configuration',
                        default=None)
    parser.add_argument('-n', '--randomnodes',
                        dest='randomnodes',
                        type=float,
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


if __name__ == '__main__':
    args = parse_args()
    if args.random is None:
        randomnumber = random.randint(0, 1000000)
    else:
        randomnumber = args.random
    random.seed(randomnumber)
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    elif args.folder != '.':
        filelist = [file for file in os.listdir(args.folder)]
        for file in filelist:
            os.remove(os.path.join(args.folder, file))
    import logging
    llevel = logging.INFO
    logging.basicConfig(
        filename='{}/main.log'.format(args.folder), level=llevel, format='%(asctime)s %(levelname)s\t %(message)s',
        filemode='w')
    logging.info('Randomseed = ' + str(randomnumber))
    sim = Simulator(jsonfile=args.json, coding=args.coding, fieldsize=args.fieldsize,
                    sendall=args.sendam, own=args.own, edgefail=args.failedge, nodefail=args.failnode,
                    allfail=args.failall, randcof=args.randomnodes, folder=args.folder,
                    maxduration=args.maxduration, randomseed=randomnumber)

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
    with open('{}/path.json'.format(args.folder), 'w') as f:
        newdata = {}
        for batch in sim.getpath():
            newdata[batch] = {}
            for key, value in sim.getpath()[batch].items():
                newdata[batch][str(key)] = value
        json.dump(newdata, f)
    logging.info('Total used airtime {}'.format(sim.calcairtime()))
