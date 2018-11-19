This project will contain a network graph simulator based on networkx. It will be able to simulate (re)coding
The written thesis will be at latex.spinnefarn.de 

components.py
    - contains the cass Node
    - necessary for Simulator
    
main.py
    - Runs a simulator
    - Made to supervise the simulator to work
    - Parameter:
        - c/coding - batch size for coding
        - d/david - use Davids approach MOREresilience
        - f/fieldsize - field size for coding
        - F/folder - specify a folder where the results should be placed
        - fe/failedge - specify an edge which should not work in the simulated graph
        - fn/failnode - specify an node which should not work in the simulated graph
        - fa/failall - if True every node and edge in the graph will not work for one simulation. Will run many simulations
        - j/json - Path to JSON contains network setup
        - max/maxduration - Specify a maximum duration one simulation should take. Int value in timesteps
        - n/randomnodes - If theire is no given jsonfile to read the graph from it will gets a random graph based on 
        this values. Default are (20, 0.3) 20 nodes connected if closer than 0.3
        - o/own - Use own attempt to increase resilience
        - r/random - Give a seed to reproduce randomness
        - s/sourceshift - Enable source shifting
        - sa/sendamount - Specify the amount of nodes allowed to send per time slot
        
multi.py
    - Runs lots of simulators in different processes with slighly different configuration
    
plotter.py
    - collects logs from supfolders to plot diagrams of a whole measurenment with multiple simulations
    - plots graph with used edges marked purple
    - plots airtime graph
    - plots latency graph
    
Simulator.py
    - Contains the actual simulator
    - Should be used by main or multi
