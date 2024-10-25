import mujoco
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import time

# XML model definition
xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""

# Create model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Create a rendering context
renderer = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Set up the camera
camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)
camera.lookat = np.array([0.0, 0.0, 0.0])
camera.distance = 2.0
camera.azimuth = 90.0
camera.elevation = -30.0

# Initialize the scene
scene = mujoco.MjvScene(model, maxgeom=1000)

# Rendering and capturing frames
frames = []
for _ in range(100):
    mujoco.mj_step(model, data)

    # Update the scene
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), camera, mujoco.mjtCatBit.mjCAT_ALL, scene)

    # Render the scene to an offscreen buffer
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    viewport = mujoco.MjrRect(0, 0, img.shape[1], img.shape[0])
    mujoco.mjr_render(viewport, scene, renderer)

    # Read pixels from the offscreen buffer
    mujoco.mjr_readPixels(img, None, viewport, renderer)
    frames.append(img)

# Display the captured frames as a video
media.show_video(frames, fps=20)
