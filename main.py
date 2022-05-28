import numpy as np

from systems.sys_276BPR3            import Sys_276BPR3
from controllers.ctr_276BPR3_CEC    import Ctr_276BPR3_CEC
from simulators.sim_276BPR3         import Sim_276BPR3
from visualizers.vis_276BPR3        import Vis_276BPR3
from testsuites.tst_Baseline        import Tst_Baseline

if __name__ == '__main__':
    '''
    to use: 
    import desired system model, controller, simulator, visualizer, and test suite from modules above,
    then run the test suite with args below.
    '''

    # instantiate test suite
    tester = Tst_Baseline(sys = Sys_276BPR3,
                          ctr = Ctr_276BPR3_CEC,
                          sim = Sim_276BPR3,
                          vis = Vis_276BPR3)

    # user define initial condition, duration, and sim options
    xi = np.array([[1.5],[0],[np.pi/2]])
    duration = 120
    noise = True

    # run sim, generate visualizations if verbose
    tester.run(xi,duration,noise,verbose=True)