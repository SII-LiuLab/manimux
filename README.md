<div align="center">

<p><a href="https://sii-liulab.github.io/manimux/"><img src="assets/manimux-logo.png" alt="ManiMux" width="300"></a></p>

### Real robot experiments. One inference harness.

An open-source harness for real-world manipulation inference and experiment management.

**[Website](https://sii-liulab.github.io/manimux/)** · **[Documentation](https://sii-liulab.github.io/manimux/guideline.html)** · **[Quick start](#quick-start)** · **[Demo](https://sii-liulab.github.io/manimux/#recordings)**

[![Python: 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-5C996B?style=flat&labelColor=45534B&logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-5C996B?style=flat&labelColor=45534B)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/Contributions-welcome-5C996B?style=flat&labelColor=45534B)](#contributing)
[![GitHub stars](https://img.shields.io/github/stars/SII-LiuLab/manimux?style=flat&label=Stars&color=5C996B&labelColor=45534B&logo=github&logoColor=white)](https://github.com/SII-LiuLab/manimux/stargazers)

</div>

[![Dream-1 demo · lossless autoplay · 1440p, 30 fps](assets/dream1-task-1440p30-lossless.webp)](https://sii-liulab.github.io/manimux/assets/dream1-task-1440p30.mp4)

**Dream-1 on YAM, through RoboGUI.** Watch a real-robot run with live camera views,
robot state and predicted action chunks in one workspace.

[Watch the original video · 1440p, 30 fps](https://sii-liulab.github.io/manimux/assets/dream1-task-1440p30.mp4) · [Run Dream-1 on your station](manimux/configs/examples/dream1-yam/README.md)

<a id="what-you-can-do"></a>

## Why ManiMux

- **One deployment workflow.** Connect supported policies and robot stations through
  shared configuration, then inspect and control runs in RoboGUI.
- **Inference as an experiment variable.** Compare serial, asynchronous and
  model-supported sampling strategies while recording their configurations and execution traces.
- **Records you can revisit.** Keep trajectories, camera recordings and task-defined
  human evaluations together for replay, analysis and comparison.

## Architecture

ManiMux sits between policy servers and robot hardware, coordinating observation/action
adaptation, scheduling, execution, and experiment records. Model frameworks own inference;
RoboGUI provides the operator workspace.

![ManiMux architecture: observations and actions flow between policy serving, the runtime and a robot station interacting with a rotating world.](assets/architecture/manimux-architecture.gif)

<details>
<summary>View the static diagram</summary>

![Static ManiMux architecture diagram](assets/architecture/manimux-architecture.png)

</details>

[Architecture and extension protocols](.agents/skills/manimux-development/references/README.md) · [Inference guide](https://sii-liulab.github.io/manimux/guideline.html#algorithms)

<a id="integrate"></a>
<a id="included-integrations"></a>

<a id="integrations"></a>

## Feature Matrix

<table>
  <thead>
    <tr>
      <th>Embodiments</th>
      <th>Policy &amp; Harness &amp; API</th>
      <th>Inference</th>
    </tr>
  </thead>
  <tbody>
    <tr valign="top">
      <td>
        <strong>Arm</strong>
        <ul>
          <li><a href="manimux/embodiments/arm/yam/README.md">YAM</a> ✅</li>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#configuration">Tianji</a> ✅</li>
          <li><a href="manimux/embodiments/arm/franka/README.md">Franka FR3</a> ✅</li>
          <li><a href="manimux/embodiments/arm/arx_x5/README.md">ARX X5</a></li>
          <li><a href="manimux/embodiments/arm/piper/README.md">PiPER</a></li>
        </ul>
        <strong>Gripper</strong>
        <ul>
          <li><a href="manimux/embodiments/end_effector/robotiq/README.md">Robotiq</a> ✅</li>
          <li><a href="manimux/embodiments/end_effector/taccap/README.md">TacCap</a> ✅</li>
        </ul>
        <strong>Sensor</strong>
        <ul>
          <li><a href="manimux/embodiments/sensor/realsense/README.md">RealSense</a> ✅</li>
          <li><a href="manimux/embodiments/sensor/orbbec/README.md">Orbbec</a> ✅</li>
          <li><a href="manimux/embodiments/sensor/taccap">TacCap Camera</a> ✅</li>
        </ul>
      </td>
      <td>
        <strong>Policy</strong>
        <ul>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#support">XPolicyLab</a> ✅
            <ul>
              <li><a href="https://github.com/Cuzyoung/XPolicyLab/tree/ziyang/policy/Pi_05">Pi05</a></li>
              <li><a href="https://github.com/Cuzyoung/XPolicyLab/tree/ziyang/policy/LingBot_VLA2">LingBot-VLA2</a></li>
              <li><a href="https://github.com/Cuzyoung/XPolicyLab/tree/ziyang/policy/Xiaomi_Robotics_1">Xiaomi Robotics 1</a></li>
              <li><a href="https://github.com/Cuzyoung/XPolicyLab/tree/ziyang/policy/OpenWAM">OpenWAM</a></li>
              <li><a href="https://github.com/Cuzyoung/XPolicyLab/tree/ziyang/policy/FastWAM">FastWAM</a></li>
              <li><a href="https://github.com/Cuzyoung/XPolicyLab/tree/ziyang/policy/GR00T_N17">GR00T</a></li>
              <li><a href="https://github.com/Cuzyoung/XPolicyLab/tree/ziyang/policy/DP">Diffusion Policy</a></li>
              <li><a href="https://github.com/Cuzyoung/XPolicyLab/tree/ziyang#readme">More models →</a></li>
            </ul>
          </li>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#support">StarVLA</a> ✅</li>
        </ul>
        <strong>Harness</strong>
        <ul>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#agent-integrations">RPent</a> ✅</li>
        </ul>
        <strong>API</strong>
        <ul>
          <li><a href="manimux/configs/examples/dream1-yam/README.md">Dream-1</a> ✅</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#serial">Serial</a> ✅</li>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#async">Async chunking</a> ✅</li>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#rtc">RTC</a> ✅</li>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#train-time-rtc">Train-time RTC</a> ✅</li>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#act-temporal-ensemble">ACT temporal ensemble</a> ✅</li>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#paint">PAINT</a> ✅</li>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#aac">AAC</a> ✅</li>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#autohorizon">AutoHorizon</a> ✅</li>
          <li><a href="https://sii-liulab.github.io/manimux/guideline.html#bid-backward">BID (backward-only)</a> ✅</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

✅ Integrated. Models are listed under their serving framework; station support varies.
Unmarked hardware entries are experimental. See the [support catalog](https://sii-liulab.github.io/manimux/guideline.html#support)
and [supported combinations](https://sii-liulab.github.io/manimux/guideline.html#optional-algorithm-composition) for compatibility and validation scope.

## Quick start

Try RoboGUI without hardware. Requires **Python 3.11 or 3.12** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SII-LiuLab/manimux.git
cd manimux
uv sync --dev
uv run manimux-robogui --robot yam --demo --host 127.0.0.1 --port 8086
```

Open **http://127.0.0.1:8086**. The preview uses synthetic data and does not connect
to a robot. Model environments and checkpoints are needed only for real deployment.

<a id="dream-1-hosted-inference"></a>

**Next, choose your workflow:**

| I want to… | Start here |
| --- | --- |
| Deploy a model on a real robot | [Station setup](https://sii-liulab.github.io/manimux/guideline.html#configuration) → [Deployment recipes](https://sii-liulab.github.io/manimux/guideline.html#support) |
| Run Dream-1 through its cloud API | [Dream-1 on YAM](manimux/configs/examples/dream1-yam/README.md) |
| Compare methods and evaluate runs | [Experiment workflow](https://sii-liulab.github.io/manimux/guideline.html#experiments) · [Records & replay](https://sii-liulab.github.io/manimux/guideline.html#records) |

<a id="contributing"></a>

## Contributing & community

Contributions to robot and policy integrations, inference strategies, RoboGUI,
and documentation are welcome. Start with the [contribution guide](.agents/skills/manimux-development/references/README.md#contributing).
For agent-assisted development, use the [development skill](.agents/skills/manimux-development/SKILL.md).

**Latest update · 2026-10-18:** The [ManiMux website](https://sii-liulab.github.io/manimux/) and dedicated [user guide](https://sii-liulab.github.io/manimux/guideline.html) are live.

<a id="developer-community"></a>

<details>
<summary>Join the developer WeChat group</summary>

Join the **ManiMux developer WeChat group** to discuss integrations, experiments,
and contributions. Open the invitation below and scan it in WeChat.

<p align="center">
  <img src="assets/community/wechat-developers-20261021.jpg" alt="ManiMux developer WeChat group invitation QR code" width="300">
</p>

This invitation is valid until **October 28, 2026**. If it has expired, please open
[a GitHub issue](https://github.com/SII-LiuLab/manimux/issues) to request an updated invitation.

</details>

## Citation

If ManiMux supports your research, please cite this repository. The paper link is forthcoming.

```bibtex
@misc{manimux2026,
  title = {{ManiMux}: An Open-Source Harness for Real-World Manipulation Inference and Experiment Management},
  year = {2026},
  howpublished = {GitHub repository},
  url = {https://github.com/SII-LiuLab/manimux}
}
```

Please also cite the components, models and methods used in your work:
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), [StarVLA](https://github.com/starVLA/starVLA) and [PRM-as-a-Judge](https://github.com/YuyangLiu2003/PRM-as-a-Judge).

## License

ManiMux is licensed under [MIT](LICENSE). Third-party frameworks, SDKs and assets
retain their own licenses; see [third-party notices](licenses/THIRD_PARTY_NOTICES.md).
