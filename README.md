# xTED: Cross-Domain Policy Adaptation via Diffusion-Based Trajectory Editing
Reusing pre-collected data from different domains is an attractive solution in decision-making tasks where the accessible data is insufficient in the target domain but relatively abundant in other related domains. Existing cross-domain policy transfer methods mostly aim at learning domain correspondences or corrections to facilitate policy learning, which requires learning domain/task-specific model components, representations, or policies that are inflexible or not fully reusable to accommodate arbitrary domains and tasks. These issues make us wonder: can we directly bridge the domain gap at the data (trajectory) level, instead of devising complicated, domain-specific policy transfer models? In this study, we propose a Cross-Domain Trajectory EDiting (xTED) framework with a new transformer-based diffusion model (Decision Diffusion Transformer, DDiT) that captures the trajectory distribution from the target dataset as a prior. The proposed diffusion transformer backbone captures the intricate dependencies among state, action, and reward sequences, as well as the transition dynamics within the target data trajectories. With the above pre-trained diffusion prior, source data trajectories with domain gaps can be transformed into edited trajectories that closely resemble the target data distribution through the diffusion-based editing process, which implicitly corrects the underlying domain gaps, enhancing the state realism and dynamics reliability in source trajectory data, while enabling flexible choices of downstream policy learning methods. Despite its simplicity, DDiT demonstrates superior performance against other baselines in extensive simulation and real-robot experiments.

## Citation
```
@artical{
    niu2024xted,
    title={xTED: Cross-Domain Policy Adaptation via Diffusion-Based Trajectory Editing},
    author={Niu, Haoyi and Chen, Qimao and Liu, Tenglong and Li, Jianxiong and Zhou, Guyue and Zhang, Yi and Hu, Jianming and Zhan, Xianyuan},
    booktitle={arxiv},
    year={2024}
}
```
