## ZebrafishTracking
This repo is for the paper "Online 3D Reconstruction of Zebrafish Behavioral Trajectories within A Holistic Perspective" on IEEE BIBM22.

**Abstract** Recording activities of zebrafish is a fundamental task in biological research that aims to accurately track individuals and recover their real-world movement trajectories from multiple viewpoint videos. In this paper, we propose a novel online tracking solution based on a holistic perspective that leverages the correlation of appearance and location across views. It first reconstructs the 3D coordinates of targets frame by frame and then tracks them directly in 3D space instead of a 2D image plane. However, it is not trivial to implement such a solution which requires the association of targets across views and neighboring frames under occlusion and parallax
distortion. To cope with that, we propose the view-invariant feature representation and the Kalman filter-based 3D state estimation, and combine the advantages of both to generate robust 3D trajectories. Extensive experiments on public datasets verify the efficiency and effectiveness of the approach.

<div align="center">
<img src="./figs/illustration.png" width ="800" height ="300" alt="">
</div>

## Introduction
Coming soon


## View-invariant Feature Representation
<div align="center">
<!-- <img src="./consistency.gif" width ="150" height ="150" alt=""> -->
<img src="./figs/consistency_o7.gif" width ="150" height ="150" alt="">
<img src="./figs/coherence.gif" width ="150" height ="150" alt="">
</div>

## License
Coming soon

##  Pretained Models
- [x] [ResNet Backbone](https://drive.google.com/file/d/1joZMPoQjrmwq0DgPy7p0v-bvtJQ3CWwM/view?usp=sharing)
- [x] [RCDN Backbone](https://drive.google.com/file/d/1idm9ZWKeSzVZw1AHVfSfzMfaDRfLjq4P/view?usp=sharing)

## Results
 SEQ | Tracker       | MOTA $\uparrow$         | MOTP $\uparrow$         | IDF1 $\uparrow$         | FP $\downarrow$     | FN $\downarrow$     | FP+FN $\downarrow$  | Rcll $\uparrow$         | Prcn $\uparrow$         | MT $\uparrow$ | ML $\downarrow$ | Frag $\downarrow$     | IDS $\downarrow$    | MTBFm $\uparrow$       
:---:|:-------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-------------------:|:-------------------:|:-------------------:|:-----------------------:|:-----------------------:|:-------------:|:---------------:|:---------------------:|:-------------------:|:----------------------:
  Tst1  | Oracle        |     100.0  |     100.0  |     100.0  |     0  |     0  |     0  |     100.0  |     100.0  |  1            | 0               |     0    | 0                   |     900   
  Tst1  | Naive         | 79.4                    | 34.6                    | 88.9                    | 28                  | 157                 | 185                 |  82.6                   |  96.4                   |  1            |  0              | 30                    |  0                  | 12.2                   
  Tst1  | MVI           | 89.6                    | 33.4                    | 94.5                    | 2                   | 92                  | 94                  |  89.8                   |  99.8                   |  1            |  0              | 18                    |  0                  | 21.8                   
  Tst1  | Ours~(ResNet) |   99.3            |   42.0            |   99.7            |   3           |   3           |   6           |   99.7            |   99.7            |  1            |  0              |   3             |  0                  |   128.1          
  Tst1  | Ours~(RCDN)   |   99.3            | 41.9                    |   99.7            |   3           |   3           |   6           |   99.7            |   99.7            |  1            |  0              |   3             |  0                  |   128.1          
  Tst2  | Oracle        | 81.6                    |     100.0  | 89.9                    |     0  | 328                 | 328                 |     100.0  |  81.6                   |  2            |  0              |     25   |     0  | 27.4                   
  Tst2  | Naive         | 78.4                    | 34.6                    | 88.9                    |   45          | 344                 | 389                 |  80.9                   |   97.0            |  1            |  0              | 44                    |   0           | 16.2                   
  Tst2  | MVI           | 56.8                    | 32.4                    | 48.5                    |  251                | 519                 | 770                 |  71.2                   |  83.6                   |  1            |  0              | 61                    |  8                  | 10.1                   
  Tst2  | Ours~(ResNet) |   92.4            |   41.2            |   96.2            |  68                 |   68          |   136         |   96.2            |  96.2                   |   2     |  0              |   24            |   0           |   34.6           
  Tst2  | Ours~(RCDN)   | 91.7                    |   41.2            | 95.7                    | 72                  | 76                  | 148                 |  95.8                   |  96.0                   |   2     |  0              | 30                    | 2                   | 27.4                   
  Tst5  | Oracle        | 67.8                    |     100.0  |     80.8   |     0  | 1427                | 1427                |     100.0  |  67.8                   |  1            |  0              |     50   |     0  |     28.1  
  Tst5  | Naive         | 40.3                    | 32.8                    | 54.9                    | 577                 | 2104                | 2681                |  53.2                   |  80.6                   |  1            |  0              | 200                   |   6           | 5.9                    
  Tst5  | MVI           | 49.7                    | 33.4                    | 47.6                    | 753                 | 1495                | 2248                |  66.8                   |  80.0                   |  1            |  0              | 123                   |  15                 | 11.5                   
  Tst5  | Ours~(ResNet) |   82.6            |   40.4            |   63.2            |   375         |   398         |   793         |   91.2            |   91.6            |   5     |  0              |   112           | 12                  |   17.7           
  Tst5  | Ours~(RCDN)   | 78.5                    | 40.3                    | 57.4                    | 518                 | 437                 | 955                 |  90.3                   |  88.7                   |   5     |  0              | 130                   | 12                  | 15.2                   
  Tst10  | Oracle        | 66.6                    |     100.0  |     79.9   |     0  | 2982                | 2982                |     100.0  |  66.6                   |  1            |  0              |     119  |     0  |     23.1  
  Tst10  | Naive         | 48.0                    | 34.6                    | 58.6                    | 720                 | 3947                | 4667                |  56.1                   |  87.5                   |  0            |  0              | 246                   |   12          | 9.8                    
  Tst10  | MVI           | 59.3                    | 30.4                    | 61.2                    | 1160                | 2479                | 3639                |  72.5                   |  84.9                   |  2            |  0              | 343                   |  20                 | 9.3                    
  Tst10  | Ours~(ResNet) | 77.9                    |   40.7            | 58.2                    | 1043                |  926                |  1969               |  89.7                   |  88.6                   |   10    |  0              | 263                   | 18                  | 14.9                   
  Tst10  | Ours~(RCDN)   |   81.0            | 40.5                    |   62.8            |   897         |   805         |   1702        |   91.1            |   90.1            |   10    |  0              |   244           |   12          |   16.3    

## Citing 
Thanks to the authors of ["3DZeF"](https://vap.aau.dk/3d-zef/).

```
Coming soon
```
