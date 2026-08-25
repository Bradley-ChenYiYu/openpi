#include <pi_rviz_plugins/PromptPanel.hpp>
#include <rviz_common/display_context.hpp>
#include <QVBoxLayout>
#include <QFont>
#include <algorithm>

namespace pi_rviz_plugins {

PromptPanel::PromptPanel(QWidget* parent) : Panel(parent) {
    // Create a vertical layout
    const auto layout = new QVBoxLayout(this);

    // Create UI elements
    prompt_input_ = new QLineEdit(this);
    prompt_input_->setPlaceholderText("Enter new prompt...");
    
    update_button_ = new QPushButton("Update Prompt", this);
    stop_button_ = new QPushButton("Stop", this);
    
    status_label_ = new QLabel("[Ready]", this);
    status_label_->setAlignment(Qt::AlignCenter);
    status_label_->setStyleSheet("color: gray;");

    // Add elements to layout
    layout->addWidget(new QLabel("Prompt Update:", this));
    layout->addWidget(prompt_input_);
    layout->addWidget(update_button_);
    layout->addWidget(stop_button_);
    layout->addWidget(status_label_);

    // Connect button click to the update slot
    QObject::connect(update_button_, &QPushButton::clicked, this, &PromptPanel::onUpdateClicked);
    QObject::connect(stop_button_, &QPushButton::clicked, this, &PromptPanel::onStopClicked);
}

PromptPanel::~PromptPanel() = default;

void PromptPanel::onUpdateClicked() {
    std::string new_prompt = prompt_input_->text().toStdString();
    if (new_prompt.empty()) {
        status_label_->setText("[Error: Empty Prompt]");
        status_label_->setStyleSheet("color: red;");
        return;
    }

    // Access the ROS node from RViz context
    auto node_abstraction = getDisplayContext()->getRosNodeAbstraction().lock();
    if (!node_abstraction) {
        status_label_->setText("[Error: No ROS Node]");
        status_label_->setStyleSheet("color: red;");
        return;
    }

    rclcpp::Node::SharedPtr node = node_abstraction->get_raw_node();

    // Use an AsyncParametersClient to update the parameter of the target node
    auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(node, target_node_name_);
    
    // We use a lambda to handle the result asynchronously to avoid blocking the UI thread
    param_client->set_parameters(
        {
            rclcpp::Parameter(target_param_name_, std::string(new_prompt)),
            rclcpp::Parameter(target_enable_action_flow_param_name_, true)
        },
        [this, param_client](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> future) {
            auto result = future.get();
            const bool all_successful = !result.empty() &&
                                        std::all_of(result.begin(), result.end(), [](const auto& item) {
                                            return item.successful;
                                        });

            if (all_successful) {
                status_label_->setText("[Success: Updated]");
                status_label_->setStyleSheet("color: green;");
            } else {
                status_label_->setText("[Error: Update Failed]");
                status_label_->setStyleSheet("color: red;");
            }
        });
}

void PromptPanel::onStopClicked() {
    // Access the ROS node from RViz context
    auto node_abstraction = getDisplayContext()->getRosNodeAbstraction().lock();
    if (!node_abstraction) {
        status_label_->setText("[Error: No ROS Node]");
        status_label_->setStyleSheet("color: red;");
        return;
    }

    rclcpp::Node::SharedPtr node = node_abstraction->get_raw_node();

    // Use an AsyncParametersClient to disable action flow on the target node
    auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(node, target_node_name_);
    param_client->set_parameters(
        {rclcpp::Parameter(target_enable_action_flow_param_name_, false)},
        [this, param_client](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> future) {
            auto result = future.get();
            if (!result.empty() && result[0].successful) {
                status_label_->setText("[Stopped]");
                status_label_->setStyleSheet("color: orange;");
            } else {
                status_label_->setText("[Error: Stop Failed]");
                status_label_->setStyleSheet("color: red;");
            }
        });
}

} // namespace pi_rviz_plugins

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(pi_rviz_plugins::PromptPanel, rviz_common::Panel)
