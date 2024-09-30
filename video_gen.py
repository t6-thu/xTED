# import torch
# import dmc_envs
# import robel
import gym
import numpy as np
from gym import wrappers
import matplotlib.pyplot as plt 
from matplotlib import animation 
# from utilis.config import ARGConfig
# from utilis.default_config import default_config
# from model.algorithm import SAC, OPT_Q, TD3, OPT_TD3
# from utilis.Replaybuffer import ReplayMemory
import datetime
import itertools
from copy import copy
import ipdb
import sys
import os
import re
import ipdb
import time
from utilities.utils import update_target_env_gravity, update_target_env_density, update_target_env_friction


def display_frames_as_mp4(frames, i, save_path, width=64, height=48):
    # ipdb.set_trace()
    fig, ax = plt.subplots(figsize=(width, height), dpi=10)
    patch = ax.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1)
    anim.save('%s/%d.gif' % (save_path, i), writer='ffmpeg', fps=30)
    plt.close()

def display_frame_as_pdf(frame, save_path, env_name, unreal_dynamics, variety="None", width=8, height=6):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    ax.imshow(frame)
    ax.axis('off')
    if variety == "None":
        render_env_name = "{}_{}".format(env_name, unreal_dynamics)
    else:
        render_env_name = "{} {} {}".format(env_name, unreal_dynamics, variety)
    
    render_env_name = "HalfCheetah-v2 Long Torso"
    text_x = 0.20 * (1 + ax.get_xlim()[1])
    ax.text(text_x, 60, render_env_name, color='white', fontsize=20, bbox=dict(facecolor=(0, 0, 0, 0.8), edgecolor='black', boxstyle='square,pad=0.4'))
    plt.savefig('%s/Scenarios_%s.pdf' % (save_path, render_env_name), bbox_inches='tight')
    plt.close()
    
    
def parse_xml_name(env_name):
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    else:
        raise RuntimeError("No available environment named \'%s\'" % env_name)

    return xml_name

# generate xml assets path: gym_xml_path
def generate_xml_path():
    import gym, os
    xml_path = os.path.join(gym.__file__[:-11], 'envs/mujoco/assets')

    assert os.path.exists(xml_path)
    print("gym_xml_path: ",xml_path)

    return xml_path


gym_xml_path = generate_xml_path()
# def display_frame_as_pdf(frame, save_path, env_name, width=8, height=6):
#     fig, ax = plt.subplots(figsize=(width, height), dpi=150)
#     ax.imshow(frame)
#     ax.axis('off')
#     plt.savefig('%s/%s.pdf' % (save_path, env_name), bbox_inches='tight')
#     plt.close()

def display_model(env_name, unreal_dynamics, variety):
    # update_target_env_ellipsoid_limb(env_name)
    if unreal_dynamics == "gravity":
        update_target_env_gravity(float(variety), env_name)
    elif unreal_dynamics == "density":
        update_target_env_density(float(variety), env_name)
    elif unreal_dynamics == "friction":
        update_target_env_friction(float(variety), env_name)
    else:
        raise RuntimeError("Got erroneous unreal dynamics %s" % unreal_dynamics)
    env = gym.make(env_name)

    #* video save path
    import ipdb
    # ipdb.set_trace()
    
    scenario_path = os.path.join('Scenarios')
    os.system('mkdir -p %s'%scenario_path)

    state = env.reset()
    # # Change the camera settings
    # env.viewer.cam.distance = env.model.stat.extent * 0.5
    # env.viewer.cam.lookat[0] = 0.5  # x-coordinate
    # env.viewer.cam.lookat[1] = 0.5  # y-coordinate
    # env.viewer.cam.lookat[2] = 0.5  # z-coordinate
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    
    # frames =
    frames = env.render(mode = 'rgb_array', width=640, height=480)
    env.close()
    #! save scenario pdf
    display_frames_as_mp4(frames,save_path=scenario_path, i=0)


if __name__ == "__main__":
    env_name = sys.argv[1]
    unreal_dynamics = sys.argv[2]
    variety_degree = sys.argv[3]
    display_model(env_name, unreal_dynamics, variety_degree)