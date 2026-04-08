#!/usr/bin/env python3

import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from tracetools_launch.action import Trace


def generate_launch_description():

    remappable_topics = [
        DeclareLaunchArgument('point_cloud_topic',
                              default_value='~/point_cloud',
                              description='input point cloud topic remap'),
        DeclareLaunchArgument('object_list_topic',
                              default_value='~/object_list',
                              description='output object list topic remap'),
        DeclareLaunchArgument(
            'no_detection_zone_topic',
            default_value='~/no_detection_zone',
            description='no-detection zone polygon topic remap'),
        DeclareLaunchArgument(
            'no_detection_zone_points_topic',
            default_value='~/no_detection_zone_points',
            description='no-detection zone points topic remap'),
        DeclareLaunchArgument('detection_area_topic',
                              default_value='~/detection_area',
                              description='detection area polygon topic remap'),
        DeclareLaunchArgument('model_bounds_topic',
                              default_value='~/model_bounds',
                              description='model bounds polygon topic remap'),
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
            'manifest_path',
            default_value=os.path.join(
                get_package_share_directory('point_cloud_object_detection'),
                'model_manifests',
                'model_manifest.yml'),
            description='path to model_manifest.yml (absolute or package-relative)'),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='ROS logging level (debug, info, warn, error, fatal)'),
        DeclareLaunchArgument('use_sim_time',
                              default_value='false',
                              description='use simulation clock'),
        DeclareLaunchArgument('trace',
                              default_value='false',
                              description='Enable tracing'),
        *remappable_topics,
    ]

    nodes = [
        Node(
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
    ]

    if Trace is not None:
        nodes.append(
            Trace(
                session_name='trace',
                dual_session=True,
                condition=IfCondition(LaunchConfiguration('trace')),
            ))

    return LaunchDescription([
        *args,
        SetEnvironmentVariable('POINT_CLOUD_OBJECT_DETECTION_MODEL_MANIFEST_PATH',
                               LaunchConfiguration('manifest_path')),
        SetParameter('use_sim_time', LaunchConfiguration('use_sim_time')),
        *nodes,
    ])
