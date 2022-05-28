# Project 3: Infinite-Horizon Stochastic Optimal Control
I put a bit more effort into this code, since it is more closely related to my research and other MAE coursework, and I'd like to reuse it later. The file structure and some class structure was moderately inspired by [this nonlinear control repo](https://github.com/MIT-REALM/neural_clbf), which I've been studying from to develop neural control lyapunov/barrier functions. **No code was copied**, and the technical matter is only tangentially related. Everything in this submission was written by me, from scratch, specifically for this project (except for the provided utils, of course).

## File structure:
```
submisison
│-  README.md
│-  main.py          <--- runs code as described in report
│
└───systems
│   │-  sys_Parents.py      <--- abstract parent classes
│   │-  sys_DiffDrive.py    <--- system used in report
│   │-  ...                 <--- other systems, beyond project scope
│   
└───controllers
│   │-  ctr_Parents.py      <--- abstract parent classes
│   │-  ctr_CEC.py          <--- controller used in report part 1
│   │-  ctr_GPI.py          <--- controller used in report part 2
│   │-  ...                 <--- other controllers, beyond project scope
│
└───simulators
│   │-  sim_Parents.py      <--- abstract parent classes
│   │-  sim_276BPR3.py      <--- baseline sim used in report
│   │-  ...                 <--- other sims, used where referenced
│
└───visualizers
│   │-  vis_Parents.py      <--- abstract parent classes
│   │-  vis_276BPR3.py      <--- baseline vis used in report
│   │-  ...                 <--- other vis, used where referenced
│
└───testsuites
│   │-  tst_Parents.py      <--- abstract parent classes
│   │-  tst_Baseline.py     <--- baseline test suite used in report
│   │-  ...                 <--- other test suites, used where referenced
│
└───utilities
    │-  utils.py            <--- the provided utils file
```

## Instructions:

main.py is straightforward to use. it does the following:
* picks a dynamic system from /systems, specifically the diff drive
  * this contains the drift f(x), g(x), w(x) info, as well as control bounds.
* picks a controller from /controllers, specifically CEC or GPI
  * this uses info from the dynamic system to calculate the control policy
* picks a simulator from /simulators, specifically the baseline sim
  * this uses the system and controller to simulate the system through time
* picks a visualizer from /visualizers, specifically the baseline vis
  * this renders the sim data for the user to review

For details on how the interior functions work, please review the code, which is extensively commented.