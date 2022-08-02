# ResNet-LDDMM: neural networks as learnable diffeomorphic image registers

I implement "ResNet-LDDMM: Advancing the LDDMM Framework Using Deep Residual Networks" found: https://arxiv.org/abs/2102.07951.

This paper introduces neural nets as discrete ODE solvers, and specifically employs one that uses the forward Euler scheme by learning kinetic energy minimising trajectories. In fact, the learnt register is that of LDDMM methods. The resulting learnt network is able to produce a diffeomorphic shape register showing the shapes morphism given a starting and ending frame.

Not yet implemented: Sinkhorn's algorithm, the L2 norm is used in-place
