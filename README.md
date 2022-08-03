### Image registration for shapes that match through diffeomorphisms, inspired by LDDMMs and implemented with Neural Networks as ODEs

An example:

![](dense_resnet_correct_LDDMM_flow_paper_specs_1_width_500.gif)

This is a simple (and slightly incomplete) implementation of https://arxiv.org/abs/2102.07951.

This paper introduces neural nets as discrete ODE solvers, and specifically employs one that uses the forward Euler scheme by learning kinetic energy minimising trajectories. In fact, the learnt register is that of LDDMM methods. The resulting learnt network is able to produce a diffeomorphic shape register showing the shapes morphism given a starting and ending frame. My implementation is incomplete as it uses L2 as the data term instead of Earth Mover's or Chamfer's distance. This is due to still implementing Sinkhorn's algorithm.

A terse summary of the results from this paper are below:

