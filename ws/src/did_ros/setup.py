from setuptools import setup

package_name = 'did_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DID Hack team',
    maintainer_email='91024625+Mput13@users.noreply.github.com',
    description='Узлы ROS 2 стенда DID Hack: судья (/did/*) и показ скрытой правды.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'judge = did_ros.judge_node:main',
        ],
    },
)
