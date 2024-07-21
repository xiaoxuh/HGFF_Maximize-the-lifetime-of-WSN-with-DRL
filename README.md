# HGFF_Maximize-the-lifetime-of-WSN-with-DRL

Code for Paper *HGFF: A DRL Approach for Lifetime Maximization in Wireless Sensor Networks*

![image](https://github.com/xiaoxuh/HGFF_Maximize-the-lifetime-of-WSN-with-DRL/blob/main/framework.jpg)

### HGFF

We propose a method called HGFF to maximize the lifetime of wireless sensor networks based on deep reinforcement learning, using heterogeneous graph neural networks and multi-head attention mechanisms to learn the state representation of wireless sensor networks (WSN) and automatically generate the sink's movement path. By intelligently planning the sink's movement path, the energy consumption of nodes in the wireless sensor network is balanced, and the life of the wireless sensor network is effectively extended. The method we proposed overcomes the shortcomings of existing heuristic methods and can generate high-quality sink movement paths in a short time, greatly improving the life of WSN.




### Installing

Python3 package needed:

```
Pytorch-1.2.1
```


### Getting Started

```
python main.py
```
We have designed two kinds of environments to simulate WSNs: stationary and dynamic. The difference is whether the deployment of sensor nodes changes or not during the lifetime. You can specify them by setting '--dyn_map=True/False'.

All the map settings of WSN are in the 'Data' folder. The map can be determined through the following two parameters:
+ map_type
+ map_index

 an example:

```
python main.py --map_type=2 --map_index=0 --dyn_map=True
```


## Citation

```
@misc{han2024hgffdeepreinforcementlearning,
      title={HGFF: A Deep Reinforcement Learning Framework for Lifetime Maximization in Wireless Sensor Networks}, 
      author={Xiaoxu Han and Xin Mu and Jinghui Zhong},
      year={2024},
      eprint={2407.07747},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2407.07747}, 
}
```
