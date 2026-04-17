from setuptools import find_packages, setup

package_name = 'pi_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/websocket_bridge.launch.py']),
        ('share/' + package_name, ['README.md']),
    ],
    install_requires=['setuptools', 'numpy', 'msgpack', 'websockets>=11.0', 'typing_extensions'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='172489412+Bradley-ChenYiYu@users.noreply.github.com',
    description='ROS2 websocket bridge for OpenPI policy serving.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'websocket_bridge = pi_bridge.websocket_bridge_node:main',
            'random_test_publisher = pi_bridge.random_test_publisher:main',
        ],
    },
)
