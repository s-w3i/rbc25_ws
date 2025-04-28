from setuptools import find_packages, setup
import glob

package_name = 'receptionist'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install all launch files using glob
        ('share/' + package_name + '/launch',
            glob.glob('launch/*.py')),  
    ],
    install_requires=[
        'setuptools',
        'openai',
    ],
    zip_safe=True,
    maintainer='usern',
    maintainer_email='w3i.0425@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_state = receptionist.task_state:main',
            'name_and_drink_node = receptionist.ask_name_and_drink:main',
            'guest_description_node = receptionist.describe_guest:main',
            'guest_description_sentence_node = receptionist.describe_generation:main',
            'find_host_node = receptionist.compare_and_find_host_service:main',
            'move_to_target = receptionist.compute_goal_pose:main',
        ],
    },
)
