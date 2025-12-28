"""
Reader for Space Cadet Pinball AI shared memory interface.
This reads ball position directly from the game via shared memory.
"""

import mmap
import struct
import ctypes
import time
import logging

# Match the C++ BallState structure
class BallState(ctypes.Structure):
    _fields_ = [
        ("ball_x", ctypes.c_float),
        ("ball_y", ctypes.c_float),
        ("ball_z", ctypes.c_float),
        ("ball_speed", ctypes.c_float),
        ("ball_active", ctypes.c_bool),
        ("_padding1", ctypes.c_char * 3),  # Alignment padding
        ("score", ctypes.c_int32),
        ("balls_remaining", ctypes.c_int32),
        ("frame_count", ctypes.c_int32),
    ]

# Size of the shared memory
SHARED_MEMORY_SIZE = ctypes.sizeof(BallState)
SHARED_MEMORY_NAME = "SpaceCadetPinballAI"


class PinballGameReader:
    """Reads game state directly from the modified Space Cadet Pinball game."""
    
    def __init__(self):
        self.shared_mem = None
        self.last_frame_count = -1
        self.connected = False
        
    def connect(self):
        """Connect to the game's shared memory."""
        try:
            # Open existing file mapping
            self.shared_mem = mmap.mmap(
                -1,  # File handle (-1 = use system paging file)
                SHARED_MEMORY_SIZE,
                tagname=SHARED_MEMORY_NAME,
                access=mmap.ACCESS_READ
            )
            self.connected = True
            logging.info(f"Connected to SpaceCadetPinball shared memory")
            return True
        except Exception as e:
            logging.warning(f"Could not connect to game: {e}")
            logging.warning("Make sure the modified SpaceCadetPinball game is running!")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from shared memory."""
        if self.shared_mem:
            self.shared_mem.close()
            self.shared_mem = None
        self.connected = False
    
    def read_state(self):
        """
        Read current game state.
        Returns dict with ball position, score, etc.
        """
        if not self.connected or not self.shared_mem:
            if not self.connect():
                return None
        
        try:
            self.shared_mem.seek(0)
            data = self.shared_mem.read(SHARED_MEMORY_SIZE)
            
            # Unpack the structure
            state = BallState.from_buffer_copy(data)
            
            # Check if data is fresh
            is_new_frame = state.frame_count != self.last_frame_count
            self.last_frame_count = state.frame_count
            
            return {
                'ball_x': state.ball_x,
                'ball_y': state.ball_y,
                'ball_z': state.ball_z,
                'ball_speed': state.ball_speed,
                'ball_active': state.ball_active,
                'score': state.score,
                'balls_remaining': state.balls_remaining,
                'frame_count': state.frame_count,
                'is_new_frame': is_new_frame,
            }
            
        except Exception as e:
            logging.error(f"Error reading game state: {e}")
            self.connected = False
            return None
    
    def get_ball_position(self):
        """
        Get ball position in a format compatible with the AI training.
        Returns (x, y, width, height) or None.
        """
        state = self.read_state()
        if state is None or not state['ball_active']:
            return None
        
        # The game uses its own coordinate system
        # You'll need to calibrate these based on the game's actual dimensions
        # Typically the playfield is about 500x800 units in game coordinates
        GAME_WIDTH = 500.0
        GAME_HEIGHT = 800.0
        
        # Normalize to 0-1 range, then scale to frame dimensions
        FRAME_WIDTH = 320
        FRAME_HEIGHT = 240
        
        # Map game coordinates to frame coordinates
        x = int((state['ball_x'] / GAME_WIDTH) * FRAME_WIDTH)
        y = int((state['ball_y'] / GAME_HEIGHT) * FRAME_HEIGHT)
        
        # Clamp to valid range
        x = max(0, min(x, FRAME_WIDTH - 1))
        y = max(0, min(y, FRAME_HEIGHT - 1))
        
        return (x, y, FRAME_WIDTH, FRAME_HEIGHT)
    
    def get_score(self):
        """Get current score."""
        state = self.read_state()
        return state['score'] if state else 0
    
    def get_balls_remaining(self):
        """Get number of balls remaining."""
        state = self.read_state()
        return state['balls_remaining'] if state else 0


def test_reader():
    """Test the game reader."""
    logging.basicConfig(level=logging.INFO)
    
    reader = PinballGameReader()
    
    print("=" * 50)
    print("Space Cadet Pinball AI Game Reader Test")
    print("=" * 50)
    print("\nMake sure the MODIFIED SpaceCadetPinball game is running!")
    print("The modified game exposes ball position via shared memory.")
    print("\nPress Ctrl+C to stop.\n")
    
    try:
        while True:
            state = reader.read_state()
            
            if state:
                print(f"Ball: ({state['ball_x']:.1f}, {state['ball_y']:.1f}, z={state['ball_z']:.1f}) "
                      f"Speed: {state['ball_speed']:.1f} Active: {state['ball_active']} "
                      f"Score: {state['score']} Balls: {state['balls_remaining']} "
                      f"Frame: {state['frame_count']}")
            else:
                print("Game not connected...")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopped.")
        reader.disconnect()


if __name__ == "__main__":
    test_reader()
