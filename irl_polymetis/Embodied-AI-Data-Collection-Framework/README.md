# Lite LeRobot Infra

A lightweight robot data collection framework that keeps the useful structure of
`lerobot-record` while staying small and easy to extend:

- `Robot` is an interface implemented by hardware backends.
- `Teleoperator` is an interface implemented by control devices.
- `DatasetRecorder` writes episodes with `LeRobotDataset`.
- `FrankaPolymetisRobot` adapts `irl_polymetis` for FR3/Franka control.
- `XboxTeleoperator` ports the main control logic from the existing Xbox script.

## Install

### Linux

Use Linux as a two-environment setup:

- Linux workstation / recorder client: runs `lite_lerobot_infra`, `lerobot`, Xbox input, cameras, and the Python `polymetis` client.
- Linux NUC / robot server: runs the real-time Polymetis Franka server.

Do not try to collapse both sides into one environment unless you are intentionally resolving the current
`lerobot` and `irl_polymetis` Python/Torch version split yourself.

#### Linux workstation / recorder client

Fresh Ubuntu bootstrap:

```bash
sudo apt update
sudo apt install -y git curl wget build-essential cmake pkg-config ffmpeg libgl1 libglib2.0-0 libusb-1.0-0
cd "$HOME"
wget -O Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/bin/activate"
conda config --set auto_activate_base false
```

Clone the exact repo versions used for this project:

```bash
mkdir -p "$HOME/robotics"
cd "$HOME/robotics"

git clone https://github.com/huggingface/lerobot.git
git -C lerobot checkout df0763a2bc8153ae69ab360af39673a328004b33

git clone --recurse-submodules https://github.com/intuitive-robots/irl_polymetis.git
git -C irl_polymetis checkout 93bf7d5e056012707cd0be26282bf5e36e668db9
git -C irl_polymetis submodule update --init --recursive

git clone https://github.com/LucioSunj/Embodied-AI-Data-Collection-Framework.git
git -C Embodied-AI-Data-Collection-Framework checkout b5b832215e65785b7e4a984b5f3e7985cdbec453
```

Create the recorder environment and install dependencies:

```bash
source "$HOME/miniforge3/bin/activate"
mamba create -n lerobot-recorder python=3.12 -y
conda activate lerobot-recorder

python -m pip install --upgrade pip setuptools wheel
cd "$HOME/robotics/lerobot"
pip install -e ".[dataset]"
pip install hydra-core omegaconf grpcio protobuf
cd "$HOME/robotics/irl_polymetis/polymetis"
pip install -e .
cd "$HOME/robotics/Embodied-AI-Data-Collection-Framework/lite_lerobot_infra"
pip install -e ".[franka,realsense]"
python -c "import lerobot, polymetis, lite_lerobot_infra; print('linux recorder client ready')"
```

#### Linux NUC / Franka Polymetis server

Fresh Ubuntu bootstrap:

```bash
sudo apt update
sudo apt install -y git curl wget build-essential bc ca-certificates gnupg2 libssl-dev lsb-release libelf-dev bison flex cmake pkg-config
cd "$HOME"
wget -O Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/bin/activate"
conda config --set auto_activate_base false
```

Clone the server repo:

```bash
mkdir -p "$HOME/robotics"
cd "$HOME/robotics"

git clone --recurse-submodules https://github.com/intuitive-robots/irl_polymetis.git
git -C irl_polymetis checkout 93bf7d5e056012707cd0be26282bf5e36e668db9
git -C irl_polymetis submodule update --init --recursive
```

Install a PREEMPT_RT kernel on the NUC before building the Franka server. These commands follow the
`irl_polymetis/docs/source/prereq.md` Ubuntu 20.04 path and end with a reboot:

```bash
sudo apt update
sudo apt install -y build-essential bc curl ca-certificates gnupg2 libssl-dev lsb-release libelf-dev bison flex
mkdir -p "$HOME/preempt_rt"
cd "$HOME/preempt_rt"
curl -SLO https://mirrors.edge.kernel.org/pub/linux/kernel/v5.x/linux-5.11.tar.xz
curl -SLO https://mirrors.edge.kernel.org/pub/linux/kernel/projects/rt/5.11/older/patch-5.11-rt7.patch.xz
xz -d linux-5.11.tar.xz
xz -d patch-5.11-rt7.patch.xz
tar xf linux-5.11.tar
cd linux-5.11
patch -p1 < ../patch-5.11-rt7.patch
make oldconfig
sudo make -j"$(nproc)" deb-pkg
sudo dpkg -i ../linux-headers-5.11.0-rt7_*.deb ../linux-image-5.11.0-rt7_*.deb
sudo reboot
```

When `make oldconfig` asks for the preemption model, choose `Fully Preemptible Kernel`.

After reboot, create the CPU-side Polymetis environment and build the Franka server:

```bash
source "$HOME/miniforge3/bin/activate"
mamba env create -n robo -f "$HOME/robotics/irl_polymetis/polymetis/environment_cpu.yml"
conda activate robo

cd "$HOME/robotics/irl_polymetis"
./scripts/build_libfranka.sh
mkdir -p polymetis/build
cd polymetis/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=ON
make -j"$(nproc)"
cd ../..
pip install -e ./polymetis
python -c "import polymetis; print('linux polymetis server ready')"
```

Start the robot and gripper servers:

```bash
source "$HOME/miniforge3/bin/activate"
conda activate robo
cd "$HOME/robotics/irl_polymetis"

ROBOT_ID=101
GRIPPER_ID=201

./scripts/start_robot.sh "$ROBOT_ID" -c robo
./scripts/start_gripper.sh "$GRIPPER_ID" -c robo
```

## Run

Use the example script:

```bash
python examples/record_franka_xbox.py \
  --repo-id local/franka_xbox_demo \
  --root ./outputs/franka_xbox_demo \
  --task "Pick up the carrot and place it in the tray" \
  --camera-serial 352122270841 \
  --camera-serial 348122070707
```

Episode control defaults:

- `Start`: begin recording an episode
- `B`: stop and save the current episode
- `Y`: discard the current episode buffer
- `Back + Start`: exit
- `Back`: request gripper recovery

## Extension Contract

To add a new robot:

1. Implement `Robot`.
2. Define stable `observation_features` and `action_features`.
3. Make `send_action()` consume exactly the same action schema that will be stored in the dataset.

To add a new teleop device:

1. Implement `Teleoperator`.
2. Return the same `action_features` as the target robot.
3. Emit episode control events from `get_episode_events()`.
