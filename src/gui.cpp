#include "gui.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include <SDL.h>

namespace {
    namespace fs = std::filesystem;

    // Control reference shown by the viewport's "Controls" hover panel: each row is
    // (action, keys, sequence_only). Sequence-only rows are dimmed when no sequence
    // is loaded. Keys use ASCII labels because the bundled font bakes only Latin
    // glyphs.
    struct ControlRow {
        const char* action;
        const char* keys;
        bool sequence_only;
    };

    const std::array<ControlRow, 12> CONTROLS = {{
        {"Orbit camera", "Right-drag", false},
        {"Pan camera", "Shift + Right-drag", false},
        {"Move camera", "W / A / S / D", false},
        {"Down / Up", "Q / E", false},
        {"Zoom", "Scroll", false},
        {"Play / Pause", "Space", true},
        {"Prev / Next frame", "Left / Right", true},
        {"First / Last frame", "Home / End", true},
        {"Reset camera", "R", false},
        {"Transparent hands", "H", false},
        {"Fullscreen", "F11", false},
        {"Quit", "Esc", false},
    }};

    /// Path to the first .ttf in the bundled fonts/ directory next to the binary.
    std::string find_bundled_font() {
        std::string base = "fonts";
        char* base_path = SDL_GetBasePath();
        if (base_path != nullptr) {
            base = std::string(base_path) + "fonts";
            SDL_free(base_path);
        }

        std::error_code error;
        if (!fs::is_directory(base, error)) {
            return {};
        }
        std::vector<std::string> ttfs;
        for (const fs::directory_entry& entry : fs::directory_iterator(base, error)) {
            std::string name = entry.path().filename().string();
            std::string lowered = name;
            std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            if (lowered.size() >= 4 && lowered.substr(lowered.size() - 4) == ".ttf") {
                ttfs.push_back(entry.path().string());
            }
        }
        if (ttfs.empty()) {
            return {};
        }
        std::sort(ttfs.begin(), ttfs.end());
        return ttfs.front();
    }

    /// Draw the "Controls" button + help panel in the viewport's top-right corner.
    bool draw_controls_overlay(const ImVec2& top_left, int width, bool has_sequence, bool show_panel) {
        const char* label = "Controls";
        const float button_width = ImGui::CalcTextSize(label).x + 16.0f;
        ImGui::SetCursorScreenPos(ImVec2(top_left.x + width - button_width - 8.0f, top_left.y + 8.0f));
        if (ImGui::Button(label)) {
            show_panel = !show_panel;
        }
        if (!show_panel) {
            return show_panel;
        }

        const float panel_width = 360.0f;
        const float panel_x = top_left.x + width - panel_width - 8.0f;
        const float panel_y = top_left.y + 8.0f + ImGui::GetFrameHeight() + 10.0f;
        ImGui::SetCursorScreenPos(ImVec2(panel_x, panel_y));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.10f, 0.88f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 10.0f));
        const ImGuiChildFlags child_flags = ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Borders;
        if (ImGui::BeginChild("controls_panel", ImVec2(panel_width, 0.0f), child_flags)) {
            if (ImGui::BeginTable("controls_help", 2)) {
                ImGui::TableSetupColumn("action", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("keys", ImGuiTableColumnFlags_WidthFixed);
                for (const ControlRow& row : CONTROLS) {
                    const bool dimmed = row.sequence_only && !has_sequence;
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    if (dimmed) {
                        ImGui::TextDisabled("%s", row.action);
                    } else {
                        ImGui::TextUnformatted(row.action);
                    }
                    ImGui::TableNextColumn();
                    // Right-align the key text so the shortcuts line up down the panel.
                    const float spare = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(row.keys).x;
                    if (spare > 0.0f) {
                        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + spare);
                    }
                    if (dimmed) {
                        ImGui::TextDisabled("%s", row.keys);
                    } else {
                        ImGui::TextUnformatted(row.keys);
                    }
                }
                ImGui::EndTable();
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
        return show_panel;
    }
} // namespace

std::pair<ImFont*, ImFont*> load_fonts(ImGuiIO& io) {
    const std::string font_path = find_bundled_font();
    if (font_path.empty()) {
        std::printf("No .ttf found in fonts/ — using imgui's default font.\n");
        ImFont* default_font = io.Fonts->AddFontDefault();
        return {default_font, default_font};
    }
    ImFont* ui_font = io.Fonts->AddFontFromFileTTF(font_path.c_str(), UI_FONT_SIZE);
    ImFont* fps_font = io.Fonts->AddFontFromFileTTF(font_path.c_str(), FPS_FONT_SIZE);
    return {ui_font, fps_font};
}

MenuResult draw_menu_bar(bool translucent, bool show_marker, bool free_camera) {
    bool load_requested = false;
    bool folder_requested = false;

    // Extra padding makes the bar taller; pop it right after begin so dropdown
    // contents don't grow too.
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 8.0f));
    const bool opened = ImGui::BeginMainMenuBar();
    ImGui::PopStyleVar();
    if (opened) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Select File")) {
                load_requested = true;
            }
            if (ImGui::MenuItem("Open Sequence Folder")) {
                folder_requested = true;
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Settings")) {
            ImGui::Checkbox("Transparent Hands", &translucent);
            ImGui::Checkbox("Free Camera", &free_camera);
            // The marker shows the orbit camera's look-at point, which the free
            // camera lacks, so grey it out while free camera is on.
            ImGui::BeginDisabled(free_camera);
            ImGui::Checkbox("Camera Marker", &show_marker);
            ImGui::EndDisabled();
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    return {translucent, show_marker, free_camera, load_requested, folder_requested};
}

ViewportResult draw_viewport_window(
    const Framebuffer& framebuffer, ImGuiID dock_id, float fps, ImFont* fps_font, const std::string& status, bool has_sequence, bool show_controls
) {
    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_Once);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport");
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const int width = std::max(1, static_cast<int>(avail.x));
    const int height = std::max(1, static_cast<int>(avail.y));
    const bool hovered = ImGui::IsWindowHovered();

    const ImVec2 image_pos = ImGui::GetCursorScreenPos();
    // Flip V (uv0 top = 1, uv1 bottom = 0) because GL textures are bottom-up.
    ImGui::Image(framebuffer.texture(), ImVec2(static_cast<float>(width), static_cast<float>(height)), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));

    // FPS overlay on top of the image, in the dedicated crisp font.
    const ImU32 fps_color = ImGui::ColorConvertFloat4ToU32(ImVec4(0.86f, 0.86f, 0.3f, 1.0f));
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    char fps_text[32];
    std::snprintf(fps_text, sizeof(fps_text), "%.0f fps", fps);
    draw_list->AddText(fps_font, FPS_FONT_SIZE, ImVec2(image_pos.x + 8.0f, image_pos.y + 6.0f), fps_color, fps_text);

    // Sequence frame counter (or loading note) just below the FPS readout.
    if (!status.empty()) {
        const ImU32 status_color = ImGui::ColorConvertFloat4ToU32(ImVec4(0.85f, 0.85f, 0.9f, 1.0f));
        draw_list->AddText(fps_font, FPS_FONT_SIZE, ImVec2(image_pos.x + 8.0f, image_pos.y + 6.0f + FPS_FONT_SIZE), status_color, status.c_str());
    }

    show_controls = draw_controls_overlay(image_pos, width, has_sequence, show_controls);

    ImGui::End();
    ImGui::PopStyleVar();
    return {width, height, hovered, show_controls};
}

PlaybackResult draw_playback_bar(int win_width, int win_height, int current_frame, int frame_count, bool playing) {
    ImGui::SetNextWindowPos(ImVec2(0.0f, win_height - PLAYBACK_BAR_HEIGHT));
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(win_width), PLAYBACK_BAR_HEIGHT));
    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::Begin("Playback", nullptr, flags);

    const int last_frame = frame_count - 1;
    if (ImGui::Button(playing ? "Pause" : "Play", ImVec2(70.0f, 0.0f))) {
        playing = !playing;
    }

    // Stretch the scrubber to fill the gap between the button and the label.
    ImGui::SameLine();
    char label[48];
    std::snprintf(label, sizeof(label), "frame %d / %d", current_frame + 1, frame_count);
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float slider_width = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(label).x - spacing;
    ImGui::SetNextItemWidth(std::max(1.0f, slider_width));
    // Empty format: the standalone label shows the 1-based frame instead.
    ImGui::SliderInt("##frame", &current_frame, 0, last_frame, "");

    ImGui::SameLine();
    ImGui::TextUnformatted(label);

    ImGui::End();
    ImGui::PopStyleVar();
    return {current_frame, playing};
}
