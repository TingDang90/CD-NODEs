# CD-NODEs

This is the official code for '**Constrained Dynamical Neural ODE For Time Series Modelling: A Case Study on Continuous Emotion Prediction**'accepted by ICASSP 2023. 

!Keep updating!



## Getting Started

```
python3 setup1.py install
```



## Run experiments

For the CD-NODE<sub>&alpha;&#772; without rate constraint (initial point for ODE solver is given as the groundtruth): 

```
python3 CD-NODEs-initial.py --nepochs 60 --lr 0.01 --lr_decay 0.95 -- PATIENCE 5 --latent_dim 64 --nhidden 64 --seq 100 --filwin 100
```

For the CD-NODE<sub>&alpha;&#772; without rate constraint (initial point for ODE solver is predicted by a single MLP layer) : 

```
python3 CD-NODEs.py --nepochs 60 --lr 0.01 --lr_decay 0.95 -- PATIENCE 5 --latent_dim 64 --nhidden 64 --seq 100 --filwin 100
```

For the CD-NODE<sub>&alpha; with rate constraint: 

```
python3 CD-NODEs-rate.py --nepochs 60 --lr 0.01 --lr_decay 0.95 -- PATIENCE 5 --latent_dim 64 --nhidden 64 --seq 100 --filwin 100
```



## Reference 

```
@inproceedings{TBA,
  title={Constrained Dynamical Neural ODE For Time Series Modelling: A Case Study on Continuous Emotion Prediction.},
  author={Dang, Ting, et al.},
  booktitle={ICASSP},
  year={2023}
}
```



### Credits

Our code relies to a great extent on the  [torchdiffeq](https://github.com/rtqichen/torchdiffeq) by Chen, Ricky T. Q.