#ifndef PI_RVIZ_PLUGINS__AVAILABLE_PROMPTS_PANEL_HPP_
#define PI_RVIZ_PLUGINS__AVAILABLE_PROMPTS_PANEL_HPP_

#include <rviz_common/panel.hpp>
#include <QLabel>
#include <QVBoxLayout>

namespace pi_rviz_plugins {

class AvailablePromptsPanel : public rviz_common::Panel {
    Q_OBJECT
public:
    explicit AvailablePromptsPanel(QWidget* parent = nullptr);
    ~AvailablePromptsPanel() override;
};

} // namespace pi_rviz_plugins

#endif // PI_RVIZ_PLUGINS__AVAILABLE_PROMPTS_PANEL_HPP_