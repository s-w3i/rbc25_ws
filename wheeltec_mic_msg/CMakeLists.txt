cmake_minimum_required(VERSION 3.8)
project(wheeltec_mic_msg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PcmMsg.msg"
  "msg/Position.msg"
  "srv/GetDeviceType.srv"
  "srv/GetOfflineResult.srv"
  "srv/SetAwakeWord.srv"
  "srv/SetMajorMic.srv"
  "srv/SwitchMic.srv"
   DEPENDENCIES builtin_interfaces
   #DEPENDENCIES std_msgs
 )
 #ament_index_register_resource("ros1_bridge_foreign_mapping")
#  install(FILES mapping_rules.yaml
#   DESTINATION share/${PROJECT_NAME}
# )

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # uncomment the line when a copyright and license is not present in all source files
#   #set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # uncomment the line when this package is not in a git repo
#   #set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

ament_export_dependencies(rosidl_default_runtime)

ament_package()
