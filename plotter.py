#!/usr/bin/python
# coding=utf-8
"""Will collect interesting logs from subfolders and draw plots."""
import matplotlib.pylab as p
import json
import os


def readinformation(folder):
    """Read logs from folder."""
    if os.path.exists('{}/failhist.json'.format(folder)) and os.path.exists('{}/config.json'.format(folder)):
        with open('{}/failhist.json'.format(folder)) as file:
            failhist = json.loads(file.read())
        with open('{}/config.json'.format(folder)) as file:
            config = json.loads(file.read())
        return failhist, config
    raise FileNotFoundError


def plot(folders=None):
    """Plot airtime diagram to compare different simulations."""
    if folders is None:
        quit(1)
    p.figure(figsize=(20, 10))
    failhist, config = {}, {}
    for folder in folders:
        failhist, config = readinformation(folder)
        label = 'MORELESS' if config['own'] else 'MORE'
        p.bar(range(len(failhist)), list(failhist.values()), label=label, alpha=0.5)
    p.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    p.ylabel('Needed airtime in timeslots')
    p.xlabel('Failure')
    if config['maxduration']:
        p.ylim([0, config['maxduration']])
    else:
        p.yscale('log')
    p.xlim([-1, len(failhist)])
    p.title('Needed airtime over failures for different protocols')
    p.xticks(range(len(failhist)), labels=failhist.keys(), rotation=90)
    p.tight_layout()
    p.savefig('airtimefail.pdf')
    p.close()


if __name__ == '__main__':
    plot(['test7', 'test6'])
