# Dream-1 hosted inference on YAM

[Back to ManiMux](../../../../README.md#quick-start) · [Deployment guide](https://sii-liulab.github.io/manimux/guideline.html#configuration)

Run Dream-1 beta on a configured dual-arm YAM station with three cameras. No local
model server, checkpoint download or inference GPU is required. The
[standalone example](serial30-example.yaml)
uses a simple **30 Hz serial baseline**, executing all 48 returned actions before
requesting the next chunk. It does not use Smooth, blending or latency cropping;
API latency therefore produces a pause between chunks. This is a starting example,
not a claim of validated task performance.

First complete the [station setup](../../../../.agents/skills/manimux-station-setup/SKILL.md),
including the YAM SDK, camera bindings and your private
`manimux/configs/local/station.yaml`. From the repository root, use the Python
environment with the station SDKs installed:

```bash
python -m pip install -e '.[dream,realsense]'
# Bash: enter your key without echoing it or saving it in shell history.
read -rsp "Dream API key: " DREAM_API_KEY; echo
export DREAM_API_KEY
python -m manimux inspect \
  --config manimux/configs/examples/dream1-yam/serial30-example.yaml \
  --local manimux/configs/local/station.yaml
```

Start the camera service and RoboGUI in separate terminals using that environment:

```bash
python -m manimux.servers.camera.server \
  --config manimux/configs/examples/dream1-yam/serial30-example.yaml \
  --local manimux/configs/local/station.yaml
```

```bash
python -m manimux.robogui.dashboard --robot yam --host 127.0.0.1 --port 8086
```

In the terminal where you exported the key, start the runtime:

```bash
python -m manimux run \
  --config manimux/configs/examples/dream1-yam/serial30-example.yaml \
  --local manimux/configs/local/station.yaml \
  --output-dir data/dream1-example
```

Open **http://127.0.0.1:8086/**, edit the task prompt, then use **Prepare Run →
Activate Warmup → Start rollout**. Prepare moves the robot to its selected initial
pose; warmup previews predictions without dispatching them. Finish with **Finish &
Home**. Images, robot state and task text are sent to Dream Labs for inference;
provider capacity and network latency affect availability. Keep API keys in the
environment, never in YAML or commits.

