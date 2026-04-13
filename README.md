# Lite LeRobot Infra

A lightweight robot data collection framework that keeps the useful structure of
`lerobot-record` while staying small and easy to extend:

- `Robot` is an interface implemented by hardware backends.
- `Teleoperator` is an interface implemented by control devices.
- `DatasetRecorder` writes episodes with `LeRobotDataset`.
- `FrankaPolymetisRobot` adapts `irl_polymetis` for FR3/Franka control.
- `XboxTeleoperator` ports the main control logic from the existing Xbox script.

## Install

This package expects local editable installs of:

1. `lerobot`
2. `irl_polymetis/polymetis` as `polymetis`

Example:

```bash
cd lerobot && pip install -e .[dataset]
cd ../irl_polymetis/polymetis && pip install -e .
cd ../../lite_lerobot_infra && pip install -e .[franka,realsense]
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
