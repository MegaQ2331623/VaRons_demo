# Implementation of VaRons and ConVaRons
This is a reproducing code for the paper "**Handling Varied Objectives by Online Decision Making**". We provide a **demo on the online model selection task**. Full code will be released if accepted and after proceedings.

## Experiment Environment
* Python 3.9
* Numpy 1.24.3
* cvxpy 1.3.2
* lightgbm 4.0.0
## Quick start
You can run an example with **existing results** on the **online model selection** task with default scenario **10D** (K=10 and drift) by

    python main.py

## Other Parameters

Retraining is available by adding **-re bool**, the number of repeats by **-r int**, the total time horizon by **-t int**, the number of actions by **-K int**, variation environment by **-env  'random' or 'drift'**. For example, the following means **retraining** with **100** independent trials with scenario 9R (**K=9** and variation environment='drift') with total time horizon **T=2000**.

    python main.py -re True -r 100 -t 2000 -K 9 -env 'drift'

Please refer **main.py** to see or run the following code to see more options.

    python main.py -h

Meanwhile, you are welcome to retrain or add new models for selection, pretrained models are saved in "./results/"
