#!/usr/bin/env python3
import os
import time
import random
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node

import pygame
from PIL import Image

# Import your custom service definition:
# Make sure you have your package and service name correct.
# e.g.: from my_robot_msgs.srv import SpeakText
from robot_interfaces.srv import SpeakText

# Import the OpenAI client for TTS (hypothetical usage)
from openai import OpenAI

###############################################################################
# RobotFace: Handles Pygame window (eyes, mouth images, blinking, mouth frames)
###############################################################################
class RobotFace:
    def __init__(self):
        # Initialize pygame for both display and audio.
        pygame.init()
        pygame.mixer.init()
        info = pygame.display.Info()
        self.width = info.current_w
        self.height = info.current_h

        self.screen = pygame.display.set_mode((self.width, self.height), 
                                              pygame.NOFRAME)
        pygame.display.set_caption("Robot Face")

        # Load graphical assets
        self.load_assets()
        self.setup_state()
        self.setup_letter_mapping()

        # For mouth animation timing
        self.last_mouth_update_time = time.time()
        self.speech_frame_index = 1

    def setup_state(self):
        self.is_speaking = False
        self.is_blinking = False
        self.last_blink_time = time.time()
        self.current_mouth = self.mouth_images['6']  # neutral mouth
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
        # Adjust these paths to your actual images
        eye_dir = "/home/usern/rbc25_ws/robot_face/Face"
        mouth_dir = "/home/usern/rbc25_ws/robot_face/Mouth"

        open_eyes = Image.open(os.path.join(eye_dir, "1.png")).convert('RGBA')
        closed_eyes = Image.open(os.path.join(eye_dir, "2.png")).convert('RGBA')

        self.open_eyes_scaled = open_eyes.resize(
            (int(open_eyes.width * 1.3), int(open_eyes.height * 1.3)), Image.LANCZOS
        )
        self.closed_eyes_scaled = closed_eyes.resize(
            (int(closed_eyes.width * 1.3), int(closed_eyes.height * 1.3)), Image.LANCZOS
        )

        self.mouth_images = {}
        for i in range(1, 15):
            img_path = os.path.join(mouth_dir, f"{i}.png")
            img = Image.open(img_path).convert('RGBA')
            self.mouth_images[str(i)] = img

    def update_blink_state(self):
        now = time.time()
        if not self.is_blinking:
            # Possibly start a blink
            if now - self.last_blink_time > random.uniform(3, 6):
                self.is_blinking = True
                self.last_blink_time = now
                self.current_eyes = self.closed_eyes_scaled
        else:
            # If already blinking, stop after 0.2s
            if now - self.last_blink_time > 0.2:
                self.is_blinking = False
                self.current_eyes = self.open_eyes_scaled

    def update_speech_animation(self):
        if self.is_speaking:
            current_time = time.time()
            if current_time - self.last_mouth_update_time > 0.1:
                self.speech_frame_index = (self.speech_frame_index % 14) + 1
                self.current_mouth = self.mouth_images[str(self.speech_frame_index)]
                self.last_mouth_update_time = current_time
        else:
            # If not speaking, default to mouth image '6'
            self.current_mouth = self.mouth_images['6']

    def draw(self):
        # Convert PIL images to pygame Surfaces
        eyes_surface = pygame.image.fromstring(
            self.current_eyes.tobytes(), 
            self.current_eyes.size, 
            self.current_eyes.mode
        ).convert_alpha()

        mouth_surface = pygame.image.fromstring(
            self.current_mouth.tobytes(), 
            self.current_mouth.size, 
            self.current_mouth.mode
        ).convert_alpha()

        self.screen.fill((255, 255, 255))
        self.screen.blit(eyes_surface, (512 - eyes_surface.get_width() // 2, 20))
        self.screen.blit(mouth_surface, (512 - mouth_surface.get_width() // 2, 300))
        pygame.display.flip()

    def run(self):
        """Main animation loop on the main thread."""
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update_blink_state()
            self.update_speech_animation()
            self.draw()
            clock.tick(30)  # ~30 FPS

        pygame.quit()


###############################################################################
# TTSManager: Uses an OpenAI TTS endpoint to generate .wav, then plays it
###############################################################################
class TTSManager:
    def __init__(self, node):
        self.node = node
        self.speech_dir = Path("/tmp/robot_speech")
        self.speech_dir.mkdir(parents=True, exist_ok=True)

        # We'll store the .wav file here every time
        self.speech_file_path = self.speech_dir / "speech.wav"

        # Hypothetical OpenAI client
        self.client = OpenAI()

    def speak(self, text: str) -> bool:
        """
        Generate TTS from text, save to speech.wav, play it, and animate mouth.
        Return True if success, False otherwise.
        """
        try:
            if not text.strip():
                self.node.get_logger().warn("No text provided to TTS.")
                return False

            self.node.get_logger().info(f"Generating TTS from text: {text}")
            # Example usage of a hypothetical streaming TTS from OpenAI
            with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="echo",
                input=text
            ) as response:
                response.stream_to_file(self.speech_file_path)

            # Playback
            sound = pygame.mixer.Sound(str(self.speech_file_path))
            self.node.robot_face.is_speaking = True
            channel = sound.play()

            # Wait until audio finishes
            while channel.get_busy():
                time.sleep(0.01)

            self.node.robot_face.is_speaking = False
            self.node.get_logger().info("TTS finished playing.")
            return True

        except Exception as e:
            self.node.get_logger().error(f"Error during TTS: {str(e)}")
            self.node.robot_face.is_speaking = False
            return False


###############################################################################
# TalkFaceNode: Offers a ROS2 service /speak_text instead of the /talk topic
###############################################################################
class TalkFaceNode(Node):
    def __init__(self):
        super().__init__('talk_face_node')
        self.get_logger().info("TalkFaceNode started.")

        # Instantiate the Pygame-based face
        self.robot_face = RobotFace()
        # Instantiate TTS manager
        self.tts_manager = TTSManager(self)

        # Create the SpeakText service server
        self.srv = self.create_service(SpeakText, 'speak_text', self.speak_text_callback)
        self.get_logger().info("Service '/speak_text' ready for requests.")

    def speak_text_callback(self, request, response):
        """
        Callback for the 'speak_text' service.
        request.text => the text to speak
        """
        self.get_logger().info(f"Service /speak_text called with text: '{request.text}'")

        success = self.tts_manager.speak(request.text)
        response.success = success
        if success:
            response.message = "Successfully spoke the text."
        else:
            response.message = "Failed to speak the text."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TalkFaceNode()

    # Run ROS in the background
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Run the Pygame face animation on the main thread
    node.robot_face.run()

    # Cleanup once the Pygame window is closed
    node.destroy_node()
    rclpy.shutdown()
    spin_thread.join()


if __name__ == '__main__':
    main()
