# Toof Control Package

This is just a controls playground, built to test various reseaarch interests. 

Current focus is on Control Lyapunov Functions and Control Barrier Functions. 


References: [neural_clbf](https://github.com/MIT-REALM/neural_clbf), [cbf_opt](https://github.com/stonkens/cbf_opt), [hj_reachability](https://github.com/StanfordASL/hj_reachability).

## File structure:
This project is not set up as a package yet, so all files assume they are being run from the top level.
```
ToofControlPackage
│-  README.md
│-  main_train.py       <--- trains a neural CLF for a system
│-  main_test.py        <--- tests a controller on a system
│-  ...                 <--- misc, OK to ignore
│
└───systems
│   │-  sys_Parents.py            <--- abstract parent classes
│   │-  ...                       <--- systems, add per abc
│   
└───controllers
│   │-  ctr_Parents.py            <--- abstract parent classes
│   │-  ...                       <--- controllers, add per abc
│   └───trainedNetworks
│       │-  nnTrainer.py          <--- neural lyapunov trainer
│       │-  ~~~~~~.pth            <--- saved pytorch networks
│
└───simulators
│   │-  sim_Parents.py      <--- abstract parent classes
│   │-  ...                 <--- simulators, add per abc
│
└───visualizers
│   │-  vis_Parents.py      <--- abstract parent classes
│   │-  ...                 <--- visualizers, add per abc
│
└───utilities
    │-  ...                 <--- miscellaneous helper functions
```
