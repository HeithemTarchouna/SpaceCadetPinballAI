import pydirectinput

# pydirectinput uses DirectInput scancodes which SDL2 games recognize
pydirectinput.PAUSE = 0  # Remove default pause between actions
pydirectinput.FAILSAFE = False  # Disable failsafe that triggers on mouse at screen corner

# Define the actions as functions
def press_key(key):
    pydirectinput.keyDown(key)
    # print(f"Key down: {key}")

def release_key(key):
    pydirectinput.keyUp(key)
    # print(f"Key up: {key}")

def press_left_flipper():
    press_key('z')

def release_left_flipper():
    release_key('z')

def press_right_flipper():
    press_key('/')

def release_right_flipper():
    release_key('/')

def press_left_table_bump():
    press_key('x')

def release_left_table_bump():
    release_key('x')

def press_right_table_bump():
    press_key('.')

def release_right_table_bump():
    release_key('.')

def press_bottom_table_bump():
    press_key('Up')

def release_bottom_table_bump():
    release_key('Up')

def press_enter():
    pydirectinput.press('enter')
    print("Pressed: enter")

def press_f2():
    pydirectinput.press('f2')
    print("Pressed: f2")

def press_plunger():
    press_key('space')

def release_plunger():
    release_key('space')

def no_action():
    pass

# Updated perform_action function to map integer actions to action functions
# Simplified to 5 actions: no_action, left_flipper, right_flipper, both_flippers, plunger
# Plunger uses toggle: action 4 holds it, any other action releases it

_plunger_held = False

def perform_action(action):
    global _plunger_held
    
    if action == 0:
        # No action - release all
        release_key('z')
        release_key('/')
        if _plunger_held:
            release_key('space')  # Release plunger to launch!
            _plunger_held = False
    elif action == 1:
        # Left flipper
        press_key('z')
        release_key('/')
        if _plunger_held:
            release_key('space')
            _plunger_held = False
    elif action == 2:
        # Right flipper
        release_key('z')
        press_key('/')
        if _plunger_held:
            release_key('space')
            _plunger_held = False
    elif action == 3:
        # Both flippers
        press_key('z')
        press_key('/')
        if _plunger_held:
            release_key('space')
            _plunger_held = False
    elif action == 4:
        # Plunger - hold it (release on next non-plunger action)
        press_key('space')
        _plunger_held = True
    # print(f"Executed action: {action}, plunger_held: {_plunger_held}")
