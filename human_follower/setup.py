from setuptools import setup
import glob

package_name = 'human_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob.glob('launch/*.py')),  
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Human following robot using YOLO and DeepSORT',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detector = human_follower.yolo_detector:main',
            'deep_sort_tracker = human_follower.deep_sort_tracker:main',
            'nvblox_human_detection = human_follower.nvblox_detection:main',
            'goal_publisher = human_follower.goal_publisher:main',
            'task_state = human_follower.task_state:main',
        ],
    },
)