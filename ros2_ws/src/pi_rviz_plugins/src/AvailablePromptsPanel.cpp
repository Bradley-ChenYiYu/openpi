#include <pi_rviz_plugins/AvailablePromptsPanel.hpp>

#include <QFont>
#include <QTextBrowser>
#include <QSizePolicy>
#include <pluginlib/class_list_macros.hpp>

namespace pi_rviz_plugins {

AvailablePromptsPanel::AvailablePromptsPanel(QWidget* parent) : Panel(parent) {
    const auto layout = new QVBoxLayout(this);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(0);

    auto title_label = new QLabel("Available Prompts", this);
    QFont title_font = title_label->font();
    title_font.setBold(true);
    title_label->setFont(title_font);
    title_label->setContentsMargins(0, 0, 0, 0);
    title_label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);

    auto prompts_label = new QTextBrowser(this);
    prompts_label->setFrameShape(QFrame::NoFrame);
    prompts_label->setContentsMargins(0, 0, 0, 0);
    prompts_label->setMinimumHeight(240);
    prompts_label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    prompts_label->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    prompts_label->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    prompts_label->setOpenExternalLinks(false);
    prompts_label->setReadOnly(true);
    prompts_label->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::TextSelectableByKeyboard);
    prompts_label->document()->setDocumentMargin(0);
    prompts_label->setStyleSheet(
        "QTextBrowser { border: none; background: transparent; padding: 0px; }"
        "QTextBrowser ul { margin: 0; padding-left: 18px; }"
        "QTextBrowser li { margin-bottom: 6px; line-height: 1.4; }");
    prompts_label->setHtml(
        "<ul style='margin: 0; padding-left: 18px;'>"
        "<li>Go to the table with brown cups.</li>"
        "<li>From the table with brown cups, go to the table with a black cloth.</li>"
        "<li>Go to the table with red cups.</li>"
        "<li>From the table with red cups, go to the table with a black cloth.</li>"
        "<li>Go to the table with a white jug.</li>"
        "<li>From the table with a white jug, go to the table with a black cloth.</li>"
        "<li>Go to the table with a sandwich.</li>"
        "<li>From the table with a sandwich, go to the table with a black cloth.</li>"
        "</ul>");

    layout->addWidget(title_label);
    layout->addWidget(prompts_label);
}

AvailablePromptsPanel::~AvailablePromptsPanel() = default;

} // namespace pi_rviz_plugins

PLUGINLIB_EXPORT_CLASS(pi_rviz_plugins::AvailablePromptsPanel, rviz_common::Panel)