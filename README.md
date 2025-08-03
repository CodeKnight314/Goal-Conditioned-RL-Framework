<table align="center">
  <tr>
    <td align="center">
      <img src="resources/video/PandaReach-v3-DDPG.gif" width="250" style="border:1px solid black;"/>
      <br>
      <sub><b>PandaReach-v3</b></sub>
    </td>
    <td align="center">
      <img src="resources/video/PandaPush-v3-DDPG.gif" width="250" style="border:1px solid black;"/>
      <br>
      <sub><b>PandaPush-v3</b></sub>
    </td>
    <td align="center">
      <img src="resources/video/PandaSlide-v3-DDPG.gif" width="250" style="border:1px solid black;"/>
      <br>
      <sub><b>PandaSlide-v3</b></sub>
    </td>
    <td align="center">
      <img src="resources/video/PandaReach-v3-DDPG.gif" width="250" style="border:1px solid black;"/>
      <br>
      <sub><b>PandaReach-v3</b></sub>
    </td>
  </tr>
</table>

# Panda Manipulation RL

## Overview

This project implements a reinforcement learning environment for controlling a Franka Robot arm through end effector control for manipulation tasks using TD3 (Twin Delayed Deep Deterministic Policy Gradient) and SAC (Soft Actor-Critic) algorithms with optional Hindsight Experience Replay (HER).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/CodeKnight314/panda-manipulation-RL.git
    ```
2.  Ensure Mujoco is installed. For Linux, Mujoco can be installed via the following:

    ```bash
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
    mkdir -p /root/.mujoco
    tar -xzf mujoco210-linux-x86_64.tar.gz -C /root/.mujoco/
    ```

3.  Ensure all necessary supporting packagaes are downloaded:

    ```bash
    apt install -y build-essential libosmesa6-dev libglew-dev libgl1-mesa-glx libglfw3 patchelf
    apt install -y libegl1 libosmesa6 libglfw3
    ```

4.  Install necessary python packagaes via `pip`:

    ```bash
    cd panda-manipulation-RL/
    pip install -r requirements.txt
    ```

5.  Install gymnasium v1.4:
    ```bash
    pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git
    pip show gymnasium-robotics
    ```
