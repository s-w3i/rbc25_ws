from setuptools import find_packages, setup
import glob

package_name = 'open_task'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob.glob('launch/*.py')),
    ],
    install_requires=[
        'rclpy',
        'std_msgs',
        'sensor_msgs',
        'cv_bridge',
        'python-dotenv',
        'Pillow',
        'requests',
        'openai',
        'opencv-python',
    ],
    zip_safe=True,
    maintainer='usern',
    maintainer_email='w3i.0425@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'assistant_node = open_task.assistant_node:main',
            'classifier_node = open_task.voice_classifier:main',
            'last_task = open_task.last_task:main',
        ],
    },
)
