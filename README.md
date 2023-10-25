# MSGAT
Code for our CIKM 2023 Paper **"Bi-channel Multiple Sparse Graph Attention Networks for Session-based Recommendation"**


## Overview
![image](https://github.com/QEpiphany/MSGAT/assets/133072736/c1006fe2-32bc-412a-a95a-0deb745cd1ab)


## Cite
Please cite the following paper if you find our code helpful.

@inproceedings{10.1145/3583780.3614791,
author = {Qiao, Shutong and Zhou, Wei and Wen, Junhao and Zhang, Hongyu and Gao, Min},
title = {Bi-Channel Multiple Sparse Graph Attention Networks for Session-Based Recommendation},
year = {2023},
isbn = {9798400701245},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3614791},
doi = {10.1145/3583780.3614791},
abstract = {Session-based Recommendation (SBR) has recently received significant attention due to its ability to provide personalized recommendations based on the interaction sequences of anonymous session users. The challenges facing SBR consist mainly of how to utilize information other than the current session and how to reduce the negative impact of irrelevant information in the session data on the prediction. To address these challenges, we propose a novel graph attention network-based model called Multiple Sparse Graph Attention Networks (MSGAT). MSGAT leverages two parallel channels to model intra-session and inter-session information. In the intra-session channel, we utilize a gated graph neural network to perform initial encoding, followed by a self-attention mechanism to generate the target representation. The global representation is then noise-reduced based on the target representation. Additionally, the target representation is used as a medium to connect the two channels. In the inter-session channel, the noise-reduced relation representation is generated using the global attention mechanism of target perception. Moreover, MSGAT fully considers session similarity from the intent perspective by integrating valid information from both channels. Finally, the intent neighbor collaboration module effectively combines relevant information to enhance the current session representation. Extensive experiments on five datasets demonstrate that simultaneous modeling of intra-session and inter-session data can effectively enhance the performance of the SBR model.},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
pages = {2075–2084},
numpages = {10},
keywords = {graph neural networks, session-based recommendation, multiple sparse graph attention networks},
location = {Birmingham, United Kingdom},
series = {CIKM '23}
}
