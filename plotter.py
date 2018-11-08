#!/usr/bin/python
# coding=utf-8
"""Will collect interesting logs from subfolders and draw plots."""
import matplotlib.pylab as p
import json
import os
import datetime
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


def plotfailhist(date=None, folders=None):
    """Plot airtime diagram to compare different simulations."""
    if folders is None:
        quit(1)
    if date is None:
        date = ''
    p.figure(figsize=(20, 10))
    failhist, config = {}, {}
    moredict, morelessdict = {}, {}
    for folder in folders:
        failhist, config = readinformation('{}/{}'.format(date, folder))
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
                try:
                    counter.append(moredict[folder][fail])
                except KeyError:
                    pass
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
                try:
                    counter.append(morelessdict[folder][fail])
                except KeyError:
                    pass
            morelessplot[fail] = statistics.mean(counter)
            morelessstd[fail] = statistics.stdev(counter)
    else:
        morelessplot = morelessdict
    for fail in morelessplot:
        if fail not in moreplot:
            moreplot[fail] = moreplot['None']
            morestd[fail] = morestd['None']
    for fail in moreplot:
        if fail not in morelessplot:
            morelessplot[fail] = morelessplot['None']
            morelessstd[fail] = morelessstd['None']
    moreplotlist = [moreplot[key] for key in sorted(moreplot.keys())]
    morestdlist = [morestd[key] for key in sorted(morestd.keys())]
    morelessplotlist = [morelessplot[key] for key in sorted(morelessplot.keys())]
    morelessstdlist = [morelessstd[key] for key in sorted(morelessstd.keys())]
    p.bar(range(len(moreplotlist)), moreplotlist, label='MORE', alpha=0.5, yerr=morestdlist, ecolor='blue')
    p.bar(range(len(morelessplotlist)), morelessplotlist, label='MORELESS', alpha=0.5, yerr=morelessstdlist,
          ecolor='yellow')
    p.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    p.ylabel('Needed airtime in timeslots')
    p.xlabel('Failure')
    p.yscale('log')
    if config['maxduration']:
        p.ylim([0, config['maxduration']])

    p.xlim([-1, len(moreplot)])
    p.title('Needed airtime over failures for different protocols')
    p.xticks(range(len(moreplot)), labels=sorted(moreplot.keys()), rotation=90)
    p.tight_layout()
    p.savefig('{}/airtimefail.pdf'.format(date))
    p.close()


if __name__ == '__main__':
    now = datetime.datetime.now()
    date = int(str(now.year) + str(now.month) + str(now.day))
    plotfailhist(date, ['test{}'.format(i) for i in range(200)])
