from setuptools import find_packages, setup

package_name = 'robot_face'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'openai'],
    zip_safe=True,
    maintainer='robot11',
    maintainer_email='w3i.0425@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talk_face_node = robot_face.talk_face_node:main',
            'name_and_drink_node = robot_face.ask_name_and_drink:main',
        ],
    },
)
