from glob import glob

from setuptools import setup

package_name = 'did_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DID Hack team',
    maintainer_email='91024625+Mput13@users.noreply.github.com',
    description='Launch-файлы DID Hack: симуляция, судья, агент.',
    license='Apache-2.0',
)
