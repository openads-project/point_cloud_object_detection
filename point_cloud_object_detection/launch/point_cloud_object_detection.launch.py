#!/usr/bin/env python3

import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():

    remappable_topics = [
        DeclareLaunchArgument('point_cloud_topic',
                              default_value='~/point_cloud'),
        DeclareLaunchArgument('object_list_topic',
                              default_value='~/object_list'),
        DeclareLaunchArgument('class_point_cloud_car_topic',
                              default_value='~/class_point_cloud/car'),
        DeclareLaunchArgument('class_point_cloud_pedestrian_topic',
                              default_value='~/class_point_cloud/pedestrian'),
        DeclareLaunchArgument('class_point_cloud_truck_topic',
                              default_value='~/class_point_cloud/truck'),
        DeclareLaunchArgument('class_point_cloud_trailer_topic',
                              default_value='~/class_point_cloud/trailer'),
        DeclareLaunchArgument('class_point_cloud_bus_topic',
                              default_value='~/class_point_cloud/bus'),
        DeclareLaunchArgument('class_point_cloud_two_wheeler_topic',
                              default_value='~/class_point_cloud/two_wheeler'),
        DeclareLaunchArgument('unclassified_point_cloud_topic',
                              default_value='~/unclassified_point_cloud'),
    ]

    args = [
        DeclareLaunchArgument('name',
                              default_value='point_cloud_object_detection',
                              description='node name'),
        DeclareLaunchArgument('namespace',
                              default_value='',
                              description='node namespace'),
        DeclareLaunchArgument(
            'params',
            default_value=os.path.join(
                get_package_share_directory('point_cloud_object_detection'),
                'config', 'params.yml'),
            description='path to parameter file'),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='ROS logging level (debug, info, warn, error, fatal)'),
        DeclareLaunchArgument('use_sim_time',
                              default_value='false',
                              description='use simulation clock'),
        *remappable_topics,
    ]

    node = Node(
        package='point_cloud_object_detection',
        executable='point_cloud_object_detection',
        namespace=LaunchConfiguration('namespace'),
        name=LaunchConfiguration('name'),
        parameters=[LaunchConfiguration('params')],
        arguments=[
            '--ros-args', '--log-level',
            LaunchConfiguration('log_level')
        ],
        remappings=[(la.default_value[0].text, LaunchConfiguration(la.name))
                    for la in remappable_topics],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        *args,
        SetParameter('use_sim_time', LaunchConfiguration('use_sim_time')),
        node,
    ])
