import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# Global variables for callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model, data):
    # Initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    # Put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if not button_left and not button_middle and not button_right:
        return

    width, height = glfw.get_window_size(window)
    mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS) or \
                (glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

def start_simulation(xml_file):
    global model, data, cam, opt, scene, context

    # Get the full path
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname, xml_file)

    # MuJoCo data structures
    model = mj.MjModel.from_xml_path(abspath)  # MuJoCo model
    data = mj.MjData(model)                # MuJoCo data
    cam = mj.MjvCamera()                   # Abstract camera
    opt = mj.MjvOption()                   # Visualization options

    # Init GLFW, create window, make OpenGL context current, request v-sync
    glfw.init()
    window = glfw.create_window(1200, 900, "Demo", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Initialize visualization data structures
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    # Install GLFW mouse and keyboard callbacks
    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, scroll)

    # Initialize the controller
    init_controller(model, data)

    # Set the controller
    mj.set_mjcb_control(controller)

    simend = 5000  # Simulation time
    print_camera_config = 0  # Set to 1 to print camera config (useful for initializing view)

    while not glfw.window_should_close(window):
        time_prev = data.time

        while (data.time - time_prev < 1.0 / 60.0):
            mj.mj_step(model, data)

        if data.time >= simend:
            break

        # Get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # Print camera configuration (help to initialize the view)
        if print_camera_config == 1:
            print('cam.azimuth =', cam.azimuth, ';', 'cam.elevation =', cam.elevation, ';', 'cam.distance = ', cam.distance)
            print('cam.lookat =np.array([', cam.lookat[0], ',', cam.lookat[1], ',', cam.lookat[2], '])')

        # Update scene and render
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        # Swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(window)

        # Process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    glfw.terminate()

# If running this file directly, start the simulation with the default XML path
if __name__ == "__main__":
    start_simulation('humanoid.xml')
