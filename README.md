This project will contain a network graph simulator based on networkx. It will be able to simulate (re)coding
The written thesis will be at latex.spinnefarn.de 

**components.py**
    - contains the cass Node
    - necessary for Simulator
    - representation of node within the network
    - depends on numpy/kodo/os
 
**main.py**
    - Runs a simulator
    - Made to supervise the simulator to work
    - Use seed for reproducable results
    - Saves logs in to given folder
    - Does not plot, use plotter manually
    - Creates main.log for most importent events
    - depends on argparse/time/os/random
    - Parameter:
        - a/anchor - use Genias approach ANChOR
        - c/coding - batch size for coding
        - d/moreres - use Davids approach MOREresilience
        - f/fieldsize - field size for coding
        - F/folder - specify a folder where the results should be placed
        - fe/failedge - specify an edge which should not work in the simulated graph
        - fn/failnode - specify an node which should not work in the simulated graph
        - fa/failall - if True every node and edge in the graph will not work for one simulation. Will run many simulations
        - j/json - Path to JSON contains network setup
        - max/maxduration - Specify a maximum duration one simulation should take. Int value in timesteps
        - n/randomnodes - If theire is no given jsonfile to read the graph from it will gets a random graph based on 
        this values. Default are (20, 0.3) 20 nodes connected if closer than 0.3
        - no/nomore - Use own routing approach, called NOMORE, based on ExOR
        - o/own - Use own attempt to increase resilience
        - opt/optimal - Use MORE but recalculate for every failure in order to generate a optimal upper bound
        - r/random - Give a seed to reproduce randomness
        - s/sourceshift - Enable source shifting
        - sa/sendamount - Specify the amount of nodes allowed to send per time slot
 
**multi.py**
    - Accepts same parameters as main.py
    - Plots within a dedicated subprocess if wished
    - Creates main.log for most important events
    - Creates folder, with current date as default name, subfolders for each graph, subsubfolders for each simulation
    - Runs different simulators each in one subprocess, one per CPU core
    - Runs different routing approaches one after an other to compare them later
    - Amount of simulations per graph and graphs at all can be customized
 
**plotter.py**
    - collects logs from supfolders to plot diagrams of a whole measurenment with multiple simulations
    - plots graph with used edges marked purple
    - plots lots of different graph
    - depends on matplotlib/numpy/networkx/json/os/datetime/statistics/logging/scipy
    - can be used from multi to plot automated
    - can be startet seperately to plot specific diagrams
 
**Simulator.py**
    - Contains the actual simulator
    - Should be used by main or multi
    - depends on json/networkx/random/os/logging
