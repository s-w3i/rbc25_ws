#!/usr/bin/env python3
import os
import time
import random
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import pygame
from PIL import Image

# Import the OpenAI client for TTS (ensure it is installed and configured)
from openai import OpenAI

# Initialize the OpenAI client (credentials must be properly set)
client = OpenAI()


class RobotFace:
    def __init__(self):
        # Initialize pygame and its mixer for audio playback.
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((1024, 600))
        pygame.display.set_caption("Robot Face")

        # Load assets and initialize state.
        self.load_assets()
        self.setup_state()
        self.setup_letter_mapping()

        # Variables for mouth animation during speech.
        self.last_mouth_update_time = time.time()
        self.speech_frame_index = 1  # For cycling through mouth frames.

    def setup_state(self):
        self.is_speaking = False
        self.is_blinking = False
        self.last_blink_time = time.time()
        self.current_mouth = self.mouth_images['6']  # Neutral mouth image.
        self.current_eyes = self.open_eyes_scaled

    def setup_letter_mapping(self):
        self.letter_to_mouth = {
            'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7',
            'H': '8', 'I': '9', 'J': '10', 'K': '11', 'L': '12', 'M': '13',
            'N': '14', 'O': '1', 'P': '2', 'Q': '3', 'R': '4', 'S': '5', 'T': '6',
            'U': '7', 'V': '8', 'W': '9', 'X': '10', 'Y': '11', 'Z': '12',
            'Ä': '13', 'Ö': '14', 'Ü': '1', 'ß': '2'
        }

    def load_assets(self):
        # Adjust these directories as needed.
        eye_dir = "/home/robot11/Robot_Face/Face"
        open_eyes = Image.open(os.path.join(eye_dir, "1.png")).convert('RGBA')
        closed_eyes = Image.open(os.path.join(eye_dir, "2.png")).convert('RGBA')

        self.open_eyes_scaled = open_eyes.resize(
            (int(open_eyes.width * 1.3), int(open_eyes.height * 1.3)), Image.LANCZOS
        )
        self.closed_eyes_scaled = closed_eyes.resize(
            (int(closed_eyes.width * 1.3), int(closed_eyes.height * 1.3)), Image.LANCZOS
        )

        # Load mouth images from 1.png to 14.png.
        mouth_dir = "/home/robot11/Robot_Face/Mouth"
        self.mouth_images = {}
        for i in range(1, 15):
            img = Image.open(os.path.join(mouth_dir, f"{i}.png")).convert('RGBA')
            # Use the original dimensions.
            self.mouth_images[str(i)] = img.resize((img.width, img.height), Image.LANCZOS)

    def update_blink_state(self):
        now = time.time()
        if now - self.last_blink_time > random.uniform(3, 6):
            self.is_blinking = True
            self.last_blink_time = now
            self.current_eyes = self.closed_eyes_scaled
        elif self.is_blinking and now - self.last_blink_time > 0.2:
            self.is_blinking = False
            self.current_eyes = self.open_eyes_scaled

    def update_speech_animation(self):
        # Cycle through mouth frames while the robot is speaking.
        if self.is_speaking:
            current_time = time.time()
            if current_time - self.last_mouth_update_time > 0.1:
                self.speech_frame_index = (self.speech_frame_index % 14) + 1
                self.current_mouth = self.mouth_images[str(self.speech_frame_index)]
                self.last_mouth_update_time = current_time
        else:
            self.current_mouth = self.mouth_images['6']

    def draw(self):
        # Convert PIL images to pygame surfaces.
        eyes_surface = pygame.image.fromstring(
            self.current_eyes.tobytes(), self.current_eyes.size, self.current_eyes.mode
        ).convert_alpha()

        mouth_surface = pygame.image.fromstring(
            self.current_mouth.tobytes(), self.current_mouth.size, self.current_mouth.mode
        ).convert_alpha()

        self.screen.fill((255, 255, 255))
        self.screen.blit(eyes_surface, (512 - eyes_surface.get_width() // 2, 20))
        self.screen.blit(mouth_surface, (512 - mouth_surface.get_width() // 2, 300))
        pygame.display.flip()

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update_blink_state()
            self.update_speech_animation()
            self.draw()
            clock.tick(30)  # 30 FPS for smooth animation.

        pygame.quit()


class TTSManager:
    def __init__(self, node):
        # Use the node for logging and accessing the RobotFace instance.
        self.node = node
        # Fixed directory for speech files.
        self.speech_dir = Path("/tmp/robot_speech")
        self.speech_dir.mkdir(parents=True, exist_ok=True)
        self.speech_file_path = self.speech_dir / "speech.wav"

    def speak(self, text):
        self.node.get_logger().info(f"Generating speech from text: {text}")
        # Generate the speech file using OpenAI TTS and store it in the fixed directory.
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="echo",
            input=text,
            # instructions="Speak in a cheerful and positive tone.",
        ) as response:
            response.stream_to_file(self.speech_file_path)

        # Play the generated audio file.
        sound = pygame.mixer.Sound(str(self.speech_file_path))
        self.node.robot_face.is_speaking = True
        channel = sound.play()

        while channel.get_busy():
            time.sleep(0.01)

        self.node.robot_face.is_speaking = False


class TalkFaceNode(Node):
    def __init__(self):
        super().__init__('talk_face_node')
        # Instantiate the RobotFace.
        self.robot_face = RobotFace()
        # Instantiate the TTSManager.
        self.tts_manager = TTSManager(self)
        # Create a subscription to the /talk topic.
        self.subscription = self.create_subscription(
            String,
            '/talk',
            self.talk_callback,
            10  # QoS profile depth
        )
        self.get_logger().info("Subscribed to /talk topic. Awaiting messages...")

    def talk_callback(self, msg):
        self.get_logger().info(f"Received message: '{msg.data}'")
        self.tts_manager.speak(msg.data)
        self.get_logger().info("Finished speaking.")


def main(args=None):
    rclpy.init(args=args)
    node = TalkFaceNode()

    # Run ROS spinning in a background thread.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # IMPORTANT: Run the Pygame face animation (robot_face.run) on the main thread.
    node.robot_face.run()

    # When the Pygame loop finishes (i.e. window is closed), clean up.
    node.destroy_node()
    rclpy.shutdown()
    spin_thread.join()


if __name__ == '__main__':
    main()
