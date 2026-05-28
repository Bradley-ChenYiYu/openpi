#ifndef PI_RVIZ_PLUGINS__PROMPT_PANEL_HPP_
#define PI_RVIZ_PLUGINS__PROMPT_PANEL_HPP_

#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter_client.hpp>
#include <std_msgs/msg/string.hpp>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>

namespace pi_rviz_plugins {

class PromptPanel : public rviz_common::Panel {
    Q_OBJECT
public:
    explicit PromptPanel(QWidget *parent = nullptr);
    ~PromptPanel() override;

protected:
    // UI Elements
    QLineEdit* prompt_input_;
    QPushButton* update_button_;
    QPushButton* stop_button_;
    QLabel* status_label_;

private slots:
    void onUpdateClicked();
    void onStopClicked();

private:
    // The node name we want to update parameters for
    const std::string target_node_name_ = "/pi_websocket_bridge";
    const std::string target_param_name_ = "prompt";
    const std::string target_enable_action_flow_param_name_ = "enable_request_action_flow";
};

} // namespace pi_rviz_plugins

#endif // PI_RVIZ_PLUGINS__PROMPT_PANEL_HPP_
