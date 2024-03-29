#!/usr/bin/python
# coding=utf-8
"""Main script, this will supervise the simulation."""
import argparse
import os
from Simulator import Simulator
import random
from multi import runsim


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
                        default=16)
    parser.add_argument('-f', '--fieldsize',
                        dest='fieldsize',
                        type=int,
                        help='Fieldsize used for coding. 2 to the power of x, x∈(1, 4, 8, 16)',
                        default=16)
    parser.add_argument('-sa', '--sendamount',
                        dest='sendam',
                        type=int,
                        help='Amount of nodes allowed to send per timeslot.',
                        default=0)
    parser.add_argument('-a', '--anchor',
                        dest='anchor',
                        type=bool,
                        help='Use anchor or MORE.',
                        default=False)
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
    parser.add_argument('-s', '--sourceshift',
                        dest='sourceshift',
                        type=bool,
                        help='Enable source shifting',
                        default=False)
    parser.add_argument('-opt', '--optimal',
                        dest='optimal',
                        type=bool,
                        help='Use MORE but recalculate in case of failure.',
                        default=False)
    parser.add_argument('-no', '--nomore',
                        dest='nomore',
                        type=bool,
                        help='Enable new shifting, means enhanced version of source shift',
                        default=False)
    parser.add_argument('-fn', '--failnode',
                        dest='failnode',
                        type=str,
                        help='Which node should fail?.',
                        default=None)
    parser.add_argument('-fa', '--failall',
                        dest='failall',
                        type=bool,
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
    parser.add_argument('-d', '--moreres',
                        dest='moreres',
                        type=float,
                        help='Use Davids protocol MOREresilience. Value should be float value as limit.',
                        default=0.0)
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
    llevel = logging.DEBUG
    logging.basicConfig(
        filename='{}/main.log'.format(args.folder), level=llevel, format='%(asctime)s %(levelname)s\t %(message)s',
        filemode='w')
    logging.info('Randomseed = ' + str(randomnumber))
    sim = Simulator(jsonfile=args.json, coding=args.coding, fieldsize=args.fieldsize,
                    sendall=args.sendam, own=args.own, edgefail=args.failedge, nodefail=args.failnode,
                    allfail=args.failall, randcof=args.randomnodes, folder=args.folder,
                    maxduration=args.maxduration, randomseed=randomnumber, sourceshift=args.sourceshift,
                    nomore=args.nomore, moreres=args.moreres, hops=5, optimal=args.optimal, anchor=args.anchor)
    runsim(sim=sim)
    logging.info('Total used airtime {}'.format(sim.calcairtime()))
