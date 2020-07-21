# PiTree

PiTree is a conversion tool to automatically and faithfully convert complex adaptive bitrate algorithms into lightweight decision trees. This repository is the official release of the following paper:

*Zili Meng, Jing Chen, Yaning Guo, Chen Sun, Hongxin Hu, Mingwei Xu. PiTree: Practical Implementations of ABR Algorithms Using Decision Trees. In Proceedings of ACM Multimedia 2019.*

For more information, please refer to https://transys.io/pitree.

## Prerequisites

Tested with Python 3.7.4:

```console
pip install -r requirements.txt
unzip traces.zip
unzip models.zip
mkdir results
mkdir tree
```

## Converting Decision Trees

### Pre-built ABR Algorithms: RobustMPC, Pensieve, and HotDASH

```console
python learn_dt.py -a pensieve -t fcc -i 500 -n 100 -q lin
```

Parameter | Candidates | Explanation
:-: | :-: | :-:
-a | {robustmpc, pensieve, hotdash} | The ABR algorithm to convert.
-i | Integer (default=500) | Number of iterations during training.
-n | Integer (default=100) | Number of leaf nodes.
-q | {lin, log, hd} | QoE metrics.
-t | {fcc, norway, oboe} | Trained traces.
-v | {0,1} | Visualized the output decision tree.
-w | Integer (default=1) | Degree of parallelism of `teacher.predict()`.

The converted decision tree could be found at `tree/`, in the pickle format.

### Add Your Own ABR Algorithms

If you want to test your own ABR algorithms with PiTree, you could

- Expose the predict function of your methods in the format of $a=f(s)$.
- Put your model into `models/` (if any).
- Add your methods into the interfaces defined in `learn_dt.py`.

(We will refactor the codes soon in a more user-friendly way and will update the repo soon.)

## Simulation with Pensieve Simulator

```console
python main.py -a pensieve -t fcc -q lin -d path/to/your/tree.pk -l
```

Parameter | Candidates | Explanation
:-: | :-: | :-:
-a | {robustmpc, pensieve, hotdash} | The ABR algorithm to convert.
-d | {0,1} | Predict with the decision tree (`1`) or the original model (`0`).
-l | {0,1} | Log the states and bitrates.
-q | {lin, log, hd} | QoE metrics.
-t | {fcc, norway, oboe} | Trained traces.

## Put the Decision Tree into HTML and Deploy with Apache

Currently, you may want to refer to [this link](https://github.com/transys-project/metis/tree/master/case-deploy) for details. We will refactor this part soon.

## Start a Server with Tornado

```console
python server_tornado.py
```

## Citation

```text
@inproceedings{meng2019pitree,
 author = {Meng, Zili and Chen, Jing and Guo, Yaning and Sun, Chen and Hu, Hongxin and Xu, Mingwei},
 title = {PiTree: Practical Implementation of ABR Algorithms Using Decision Trees},
 year = {2019},
 url = {https://doi.org/10.1145/3343031.3350866},
 booktitle = {Proceedings of the 27th ACM International Conference on Multimedia},
 pages = {2431–2439},
 series = {MM ’19}
}
```

## Contact

For any questions, please post an issue or send an email to [zilim@ieee.org](mailto:zilim@ieee.org).
