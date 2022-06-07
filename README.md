# ECE 228 Project Repo

This branch is a subset of the ToofControlPackage, used for direct reference with UCSD ECE228 Course Project: Neural Control Lyapunov Functions.

This branch will be killed on 06/22/22 at 11:59pm.

Work here is mostly derivative, borrowing heavily from [neural_clbf](https://github.com/MIT-REALM/neural_clbf) and its associated paper.

## File structure:
This project is not set up as a package yet, so all files assume they are being run from the top level.
```
ToofControlPackage
│-  README.md
│-  main_train.py       <--- trains a neural CLF for a system
│-  main_test.py        <--- tests a controller on a system
│
└───systems
│   │-  sys_Parents.py            <--- abstract parent classes
│   │-  sys_SingleIntegrator.py   <--- system used in report
│   │-  sys_HW1P3_MAE281B.py      <--- system used in report
│   │-  ...                       <--- other systems, add per abc
│   
└───controllers
│   │-  ctr_Parents.py            <--- abstract parent classes
│   │-  ctr_CLF_L2.py             <--- controller used in report
│   │-  ctr_nCLF.py               <--- neural CLF used in report
│   │-  ...                       <--- other controllers, add per abc
│   └───trainedNetworks
│       │-  nnTrainer.py          <--- neural lyapunov trainer
│       │-  ~~~~~~.pth            <--- saved pytorch networks
│
└───simulators
│   │-  sim_Parents.py      <--- abstract parent classes
│   │-  sim_SolIVP.py       <--- simulator used in report
│   │-  ...                 <--- other simulators, add per abc
│
└───visualizers
│   │-  vis_Parents.py      <--- abstract parent classes
│   │-  vis_PlotTime.py     <--- plot each state and control over time
│   │-  ...                 <--- other visualizers, add per abc
│
└───utilities
    │-  ...                 <--- miscellaneous helper functions
```

## Instructions:
To run the code...