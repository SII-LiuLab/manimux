---
hide:
  - toc
---

<div class="guide-intro" markdown>
<div class="guide-kicker">Documentation / Real-robot research</div>

# ManiMux

<p class="guide-lead">An extensible real-world manipulation harness.</p>

**Any embodiment. Any policy. Any inference strategy.**

ManiMux **standardizes** deployment and experiment workflows across embodiments,
policies, and inference strategies. Built around **RoboGUI**, it brings together
**live digital twin visualization** and a runtime designed for **100–200 Hz command execution**.

<div class="guide-links">
<a href="usage/getting-started.md">Get started <span aria-hidden="true">↗</span></a>
<a href="development/README.md">Development protocols <span aria-hidden="true">↗</span></a>
<a href="https://github.com/SII-LiuLab/manimux">GitHub <span aria-hidden="true">↗</span></a>
</div>
</div>

<div class="guide-overview">
<figure class="guide-demo">
<video class="guide-video" controls playsinline preload="none" poster="assets/media/robogui.webp" aria-label="RoboGUI demonstration with a YAM robot">
<source src="assets/media/demo.mp4" type="video/mp4">
Your browser does not support inline video.
</video>
<figcaption>RoboGUI · YAM deployment. Demo footage; some controls have since changed.</figcaption>
</figure>
<nav class="guide-start" aria-label="Getting started paths">
<div class="guide-kicker">Start with your setup</div>
<a href="usage/getting-started.md"><span class="guide-step">01</span><span><strong>Try RoboGUI</strong><small>Explore the demo without hardware.</small></span><span aria-hidden="true">→</span></a>
<a href="usage/station.md"><span class="guide-step">02</span><span><strong>Connect your station</strong><small>Bind robots, cameras and services.</small></span><span aria-hidden="true">→</span></a>
<a href="usage/deployments.md"><span class="guide-step">03</span><span><strong>Run a policy</strong><small>Choose a recipe and start inference.</small></span><span aria-hidden="true">→</span></a>
<a href="usage/research.md"><span class="guide-step">04</span><span><strong>Manage experiments</strong><small>Free exploration or a study template.</small></span><span aria-hidden="true">→</span></a>
</nav>
</div>

## One workflow, interchangeable components

Model frameworks own inference. ManiMux connects observations, action conversion,
chunk scheduling and command execution. RoboGUI lets you operate the experiment;
recording and replay let you inspect what happened. Evaluation is optional.

```mermaid
flowchart LR
    O["Observe<br/>Cameras + robot state"] --> P["Predict<br/>Policy framework"]
    P --> T["Adapt + schedule<br/>Adapter → Timeline"]
    T --> E["Execute<br/>Executor → Robot"]
    E --> O
    E -.-> G["Operate + review<br/>RoboGUI · records · replay"]
    style P fill:#fff,color:#233d36,stroke:#377b68
    style T fill:#fff,color:#233d36,stroke:#377b68
    style E fill:#fff,color:#233d36,stroke:#377b68
    style O fill:#fff,color:#233d36,stroke:#377b68
    style G fill:#fff,color:#233d36,stroke:#8d9894,stroke-dasharray:4 3
```

[Architecture and ownership →](development/architecture.md)

<div class="guide-topics" markdown>
<div markdown>

### Use ManiMux

Choose an experiment config, start the services, and work from RoboGUI.

- [PiPER and ALOHA RoboGUI previews](usage/getting-started.md#hardware-free-start)
- [Configuration: what belongs where](usage/configuration.md)
- [Free rollouts and study templates](usage/research.md)
- [Saved records and offline replay](usage/replay.md)
- [Optional evaluation](usage/evaluation.md)

</div>
<div markdown>

### Extend with your agent

Give your agent the [development skill](../.agents/skills/manimux-development/SKILL.md).
It routes each integration to its existing interface and protocol.

- [Robots, grippers, cameras and digital twins](development/components.md)
- [Models, frameworks and action adapters](development/policies.md)
- [Inference strategies, executors and RoboGUI](development/runtime-config.md)
- [Integration map and validation](development/README.md)

</div>
</div>

Model training and demonstration collection remain in their own tools.
Choose your own research questions, experiment plans and evaluation criteria.

## Citation

**[ManiMux: An Extensible Real-World Manipulation Harness](../README.md#citation)**

Find ready-to-copy BibTeX in the [citation guide](../README.md#citation), including
XPolicyLab, StarVLA and PRM-as-a-Judge. Cite the components used in your research.
