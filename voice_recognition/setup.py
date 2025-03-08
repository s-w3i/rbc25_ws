from setuptools import setup
import os
from glob import glob

package_name = 'voice_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='YOUR_NAME',
    maintainer_email='YOUR_EMAIL@domain.com',
    description='ROS 2 node for voice recognition using OpenAI Whisper.',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_recognition_node = voice_recognition.voice_recognition_node:main'
        ],
    },
)
