cmake_minimum_required(VERSION 3.8)
project(robot_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies.
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox.msg"
  "msg/Detections.msg"
  "msg/Tracks.msg"
  "srv/SpeakText.srv"
  "srv/SegmentHumans.srv"
  "srv/DescribeGuest.srv"
  "srv/DescribeGuestSentence.srv"
  "srv/BagCentroid.srv"
  "srv/CompareAndFindHost.srv"
  "srv/CompareAndFindHostDir.srv"
  "srv/ComputeGoalPose.srv"
  DEPENDENCIES sensor_msgs geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # The following line skips the linter which checks for copyrights.
  # Uncomment it when a copyright/license is added.
  set(ament_cmake_copyright_FOUND TRUE)
  # The following line skips cpplint (only works in a git repo).
  # Uncomment it when your package is in a git repo and when proper copyright/license is in place.
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
