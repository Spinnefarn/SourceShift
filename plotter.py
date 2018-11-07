#!/usr/bin/python
# coding=utf-8
"""Will collect interesting logs from subfolders and draw plots."""
import matplotlib.pylab as p
import json
import os
import statistics


def readinformation(folder):
    """Read logs from folder."""
    if os.path.exists('{}/failhist.json'.format(folder)) and os.path.exists('{}/config.json'.format(folder)):
        with open('{}/failhist.json'.format(folder)) as file:
            failhist = json.loads(file.read())
        with open('{}/config.json'.format(folder)) as file:
            config = json.loads(file.read())
        return failhist, config
    raise FileNotFoundError


def plotfailhist(folders=None):
    """Plot airtime diagram to compare different simulations."""
    if folders is None:
        quit(1)
    p.figure(figsize=(20, 10))
    failhist, config = {}, {}
    moredict, morelessdict = {}, {}
    for folder in folders:
        failhist, config = readinformation(folder)
        if config['own']:
            morelessdict[folder] = failhist
        else:
            moredict[folder] = failhist
    moreplot, morestd = {}, {}
    if len(moredict) > 1:
        firstfolder = list(moredict.keys())[0]
        for fail in moredict[firstfolder].keys():
            counter = []
            for folder in moredict.keys():
                counter.append(moredict[folder][fail])
            moreplot[fail] = statistics.mean(counter)
            morestd[fail] = statistics.stdev(counter)
    else:
        moreplot = moredict
    morelessplot, morelessstd = {}, {}
    if len(morelessdict) > 1:
        firstfolder = list(morelessdict.keys())[0]
        for fail in morelessdict[firstfolder].keys():
            counter = []
            for folder in morelessdict.keys():
                counter.append(morelessdict[folder][fail])
            morelessplot[fail] = statistics.mean(counter)
            morelessstd[fail] = statistics.stdev(counter)
    else:
        morelessplot = morelessdict

    p.bar(range(len(moreplot)), list(moreplot.values()), label='MORE', alpha=0.5, yerr=morestd.values(), ecolor='blue')
    p.bar(range(len(morelessplot)), list(morelessplot.values()), label='MORELESS', alpha=0.5, yerr=morelessstd.values(),
          ecolor='yellow')
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
    plotfailhist(['test4', 'test5', 'test7', 'test6'])
