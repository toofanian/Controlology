# ECE 228 Project Repo

This branch is a subset of the ToofControlPackage, used for direct reference with UCSD ECE228 Course Project: Neural Control Lyapunov Functions.

This branch will be killed on 06/22/22 at 11:59pm.

Work here is mostly derivative, borrowing heavily from [neural_clbf](https://github.com/MIT-REALM/neural_clbf) and its associated paper.

## File structure:
This project is not set up as a package yet, so all files assume they are being run from the top level.
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
│   │-  ctr_nCLF.py         <--- neural lyapunov controller
│   │-  ...                 <--- other controllers, add more per abc
│   └───trainedNetworks
│       │-  nnTrainer.py        <--- neural lyapunov trainer
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

## Instructions:
To run the code...