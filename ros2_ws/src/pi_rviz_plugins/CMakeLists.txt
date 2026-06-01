cmake_minimum_required(VERSION 3.8)
project(pi_rviz_plugins)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rviz_common REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Qt5Widgets REQUIRED)

set(CMAKE_AUTOMOC ON)

qt5_wrap_cpp(MOC_FILES
  include/pi_rviz_plugins/PromptPanel.hpp
  include/pi_rviz_plugins/AvailablePromptsPanel.hpp
)

# Explicitly set as SHARED library because pluginlib requires a .so file for dynamic loading at runtime.
# Without SHARED, it may default to a static library, causing RViz to fail to load the plugin.
add_library(prompt_panel SHARED
  src/PromptPanel.cpp
  src/AvailablePromptsPanel.cpp
  ${MOC_FILES}
)
target_include_directories(prompt_panel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(prompt_panel
  rviz_common
  pluginlib
  rclcpp
  std_msgs
)

install(TARGETS prompt_panel
  EXPORT export_pi_rviz_plugins
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
install(FILES pi_rviz_plugins.xml DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_targets(export_pi_rviz_plugins)
pluginlib_export_plugin_description_file(rviz_common pi_rviz_plugins.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
