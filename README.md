# Toof Control Package
General nonlinear control playground, built to prevent redundant coding during research, and also as good practice. There are probably other, better packages for similar applications.

## File structure:
```
ToofControlPackage
│-  README.md
│-  main_test.py        <--- tests a controller on a system
│-  main_train.py       <--- trains a neural controller for a system
│
└───systems
│   │-  sys_Parents.py      <--- abstract parent classes
│   │-  ...                 <--- systems, add more per abc
│   
└───controllers
│   │-  ctr_Parents.py      <--- abstract parent classes
│   │-  ...                 <--- controllers, add more per abc
│   └───trainedNetworks
│       │-  nnTrainer.py        <--- neural lyapunov trainer class
│       │-  ~~~~~~.pth          <--- saved pytorch networks
│
└───simulators
│   │-  sim_Parents.py      <--- abstract parent classes
│   │-  ...                 <--- simulators, add more per abc
│
└───visualizers
│   │-  vis_Parents.py      <--- abstract parent classes
│   │-  ...                 <--- visualizers, add more per abc
│
└───testsuites
│   │-  tst_Parents.py      <--- abstract parent classes
│   │-  ...                 <--- test suites, add more per abc
│
└───utilities
    │-  ...                 <--- miscellaneous helper functions
```
