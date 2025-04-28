import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper  # Import the local Whisper transcription library
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AudioFileHandler(FileSystemEventHandler):
    def __init__(self, node, ready_delay=0.5):
        """
        ready_delay: seconds to wait after an event to verify file is stable
        """
        self.node = node
        self.ready_delay = ready_delay
        super().__init__()

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".wav"):
            time.sleep(self.ready_delay)
            self.node.transcribe_if_final(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".wav"):
            time.sleep(self.ready_delay)
            self.node.transcribe_if_final(event.src_path)

class WhisperNode(Node):
    def __init__(self, directory, model_name="base"):
        super().__init__('whisper_node')
        self.directory = directory
        self.last_processed_file = None
        
        # Publisher to broadcast transcripts
        self.transcript_pub = self.create_publisher(String, 'speech_recognition_transcript', 10)
        
        # Load the local Whisper model
        self.get_logger().info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        self.get_logger().info("Model loaded successfully.")

        # Start watchdog observer to monitor file system events
        self.observer = Observer()
        event_handler = AudioFileHandler(self, ready_delay=1.0)
        self.observer.schedule(event_handler, self.directory, recursive=False)
        self.observer.start()
        self.get_logger().info(f"Monitoring directory with watchdog: {self.directory}")

    def transcribe_if_final(self, file_path):
        # Avoid duplicate processing
        if file_path == self.last_processed_file:
            return
        
        # Check if file size is stable over a short period
        try:
            initial_size = os.path.getsize(file_path)
            time.sleep(0.2)
            if os.path.getsize(file_path) != initial_size:
                return
        except Exception as e:
            self.get_logger().error(f"Error checking file size: {e}")
            return
        
        transcript_text = self.transcribe_audio(file_path)
        self.get_logger().info(f"Transcript: {transcript_text}")

        # Publish the transcript message
        msg = String()
        msg.data = transcript_text
        self.transcript_pub.publish(msg)

        self.last_processed_file = file_path

    def transcribe_audio(self, file_path):
        self.get_logger().info(f"Transcribing {file_path} ...")
        try:
            # Use the local Whisper model to transcribe the audio file.
            # The language parameter is set to English.
            result = self.model.transcribe(file_path, language="en")
            transcript = result.get("text", "")
            return transcript.strip()
        except Exception as e:
            self.get_logger().error(f"Transcription failed: {e}")
            return "Transcription failed."

    def destroy_node(self):
        self.observer.stop()
        self.observer.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    # Replace with the directory where your WAV files are saved
    directory_to_watch = "/home/usern/rbc25_ws/audio"
    
    # Optionally change the model name if needed ("base", "small", "medium", etc.)
    node = WhisperNode(directory=directory_to_watch, model_name="base.en")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Whisper node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
