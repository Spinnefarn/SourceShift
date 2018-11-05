#!/usr/bin/python
# coding=utf-8
"""Main script, this will supervise the simulation."""
import argparse
import json
import time
from Simulator import Simulator


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
                        default=(20, 0.5))
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
    return parser.parse_args()


if __name__ == '__main__':
    import logging
    llevel = logging.INFO
    args = parse_args()
    logging.basicConfig(
        filename='{}/main.log'.format(args.folder), level=llevel, format='%(asctime)s %(levelname)s\t %(message)s',
        filemode='w')
    failes = None
    generated = False
    for ownbool in [False, True]:
        if not generated or args.json is not None:
            sim = Simulator(jsonfile=args.json, coding=args.coding, fieldsize=args.fieldsize,
                            sendall=args.sendam, own=ownbool, multiprocessing=args.multiprocessing,
                            edgefail=args.failedge, nodefail=args.failnode, allfail=args.failall, randcof=args.amount,
                            folder=args.folder)
        else:
            sim = Simulator(jsonfile='usedgraph.json', coding=args.coding, fieldsize=args.fieldsize,
                            sendall=args.sendam, own=ownbool, multiprocessing=args.multiprocessing,
                            edgefail=args.failedge, nodefail=args.failnode, allfail=args.failall, randcof=args.amount,
                            folder=args.folder)
            pass
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
        sim.drawtrash()
        sim.drawtrash('real')
        if failes is None:
            failes = sim.drawfailes()
        else:
            sim.drawfailes(failes)
        generated = True
    with open('{}/path.json'.format(args.folder), 'w') as f:
        newdata = {}
        for batch in sim.getpath():
            newdata[batch] = {}
            for key, value in sim.getpath()[batch].items():
                newdata[batch][str(key)] = value
        json.dump(newdata, f)
    logging.info('Total used airtime {}'.format(sim.calcairtime()))
