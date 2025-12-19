#!/usr/bin/env python3

import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnExecutionComplete
from launch.substitutions import EqualsSubstitution, LaunchConfiguration, NotEqualsSubstitution, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, SetParameter
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    remappable_topics = [
        DeclareLaunchArgument('point_cloud_topic',
                              default_value='~/point_cloud'),
        DeclareLaunchArgument('object_list_topic',
                              default_value='~/object_list'),
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
        DeclareLaunchArgument('combined_params',
                              default_value=PathJoinSubstitution([
                                  '/tmp', 'point_cloud_object_detection',
                                  'combined_params.yml'
                              ])),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='ROS logging level (debug, info, warn, error, fatal)'),
        DeclareLaunchArgument('use_sim_time',
                              default_value='false',
                              description='use simulation clock'),
        DeclareLaunchArgument(
            'composable_node_container',
            default_value='',
            description='composable node container to load to (empty for new container)'),
        *remappable_topics,
    ]

    composable_node = ComposableNode(
        package='point_cloud_object_detection',
        plugin='point_cloud_object_detection::PointCloudObjectDetection',
        namespace=LaunchConfiguration('namespace'),
        name=LaunchConfiguration('name'),
        parameters=[LaunchConfiguration('combined_params')],
        # arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        extra_arguments=[{
            'use_intra_process_comms': True
        }],
        remappings=[(la.default_value[0].text, LaunchConfiguration(la.name))
                    for la in remappable_topics],
    )

    # new component container
    composable_node_container = ComposableNodeContainer(
        condition=IfCondition(
            EqualsSubstitution(LaunchConfiguration('composable_node_container'),
                               '')),
        package='rclcpp_components',
        executable='component_container',
        namespace=LaunchConfiguration('namespace'),
        name='point_cloud_object_detection_container',
        composable_node_descriptions=[
            composable_node,
        ],
        output='screen',
        emulate_tty=True,
    )

    # external component container
    load_composable_node_to_external_container = LoadComposableNodes(
        condition=IfCondition(
            NotEqualsSubstitution(
                LaunchConfiguration('composable_node_container'), '')),
        target_container=LaunchConfiguration('composable_node_container'),
        composable_node_descriptions=[
            composable_node,
        ])

    # combine params before launching node
    node_group_action = GroupAction(actions=[
        composable_node_container, load_composable_node_to_external_container
    ])
    join_params_executor = ExecuteProcess(cmd=[[
        'python3 ',
        PathJoinSubstitution([
            get_package_share_directory('point_cloud_object_detection'),
            'scripts', 'update_params.py '
        ]),
        LaunchConfiguration('params'), ' ',
        LaunchConfiguration('combined_params'), ' ',
        LaunchConfiguration('name'), ' ',
        LaunchConfiguration('namespace')
    ]],
                                          shell=True)
    join_params_event_handler = RegisterEventHandler(
        OnExecutionComplete(target_action=join_params_executor,
                            on_completion=[node_group_action]))

    return LaunchDescription([
        *args,
        SetParameter('use_sim_time', LaunchConfiguration('use_sim_time')),
        join_params_event_handler,
        join_params_executor,
    ])
