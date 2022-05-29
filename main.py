import numpy as np

from systems.sys_ActiveCruiseControl import activeCruiseControl
from controllers.ctr_CLF_ACC         import Controller_CLF_ACC
from simulators.sim_IVP              import Sim_SolIVP
from visualizers.vis_PlotTime        import Vis_PlotTime
from testsuites.tst_Baseline         import Tst_Baseline

if __name__ == '__main__':
    '''
    to use: 
    import desired system model, controller, simulator, visualizer, and test suite from modules above,
    then run the test suite with args below.
    '''

    # instantiate test suite
    tester = Tst_Baseline(sys = activeCruiseControl,
                          ctr = Controller_CLF_ACC,
                          sim = Sim_SolIVP,
                          vis = Vis_PlotTime)

    # user define initial condition, duration, and sim options
    xi = np.array([[2],[-1]])
    duration = 50
    noise = True

    # run sim, generate visualizations if verbose
    tester.run(xi,duration,noise,verbose=True)