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
        bool sequence_only; // dimmed when no sequence is loaded
        bool paused_only;   // dimmed while a sequence is playing
    };

    const std::array<ControlRow, 12> CONTROLS = {{
        {"Orbit camera", "Right-drag", false, false},
        {"Pan camera", "Shift + Right-drag", false, false},
        {"Move camera", "W / A / S / D", false, false},
        {"Down / Up", "Q / E", false, false},
        {"Zoom", "Scroll", false, false},
        {"Play / Pause", "Space", true, false},
        {"Prev / Next frame", "Left / Right", true, true},
        {"First / Last frame", "Home / End", true, true},
        {"Reset camera", "R", false, false},
        {"Transparent hands", "H", false, false},
        {"Fullscreen", "F11", false, false},
        {"Quit", "Esc", false, false},
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
    bool draw_controls_overlay(const ImVec2& top_left, int width, bool has_sequence, bool playing, bool show_panel) {
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
                    const bool dimmed = (row.sequence_only && !has_sequence) || (row.paused_only && playing);
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

    // Result of drawing the transport bar: the (possibly user-changed) frame and
    // play state, plus whether the scrubber is being dragged this frame.
    struct TransportState {
        int current_frame;
        bool playing;
        bool scrubbing;
    };

    // Playback transport (play/pause button + scrubber + frame label) drawn as an
    // overlay along the bottom of whichever pane currently carries it. ``strip_top``
    // is the screen Y where the overlay strip begins; a translucent backing is
    // painted behind it so the controls stay legible over the scene. Updates and
    // returns the transport state, including whether the scrubber is being dragged.
    TransportState draw_transport_bar(float left, float strip_top, float width, int current_frame, int frame_count, bool playing) {
        // Translucent backing so the controls read over any scene content beneath.
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        const ImU32 backing = ImGui::ColorConvertFloat4ToU32(ImVec4(0.08f, 0.08f, 0.10f, 0.65f));
        draw_list->AddRectFilled(ImVec2(left, strip_top), ImVec2(left + width, strip_top + PLAYBACK_BAR_HEIGHT), backing);

        const float row_height = ImGui::GetFrameHeight();
        ImGui::SetCursorScreenPos(ImVec2(left + 8.0f, strip_top + (PLAYBACK_BAR_HEIGHT - row_height) * 0.5f));

        if (ImGui::Button(playing ? "Pause" : "Play", ImVec2(70.0f, 0.0f))) {
            playing = !playing;
        }

        // Stretch the scrubber to fill the gap between the button and the label.
        ImGui::SameLine();
        char label[48];
        std::snprintf(label, sizeof(label), "frame %d / %d", current_frame + 1, frame_count);
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float slider_width = width - 8.0f - 70.0f - spacing - ImGui::CalcTextSize(label).x - spacing - 8.0f;
        ImGui::SetNextItemWidth(std::max(1.0f, slider_width));
        // Empty format: the standalone label shows the 1-based frame instead.
        ImGui::SliderInt("##frame", &current_frame, 0, frame_count - 1, "");
        // True while the user holds and drags the scrubber, so playback can be
        // suspended and the hands don't jitter between the dragged and next frame.
        const bool scrubbing = ImGui::IsItemActive();

        ImGui::SameLine();
        ImGui::TextUnformatted(label);
        return {current_frame, playing, scrubbing};
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
    const Framebuffer& framebuffer,
    ImGuiID dock_id,
    float fps,
    ImFont* fps_font,
    const std::string& status,
    bool show_controls,
    bool has_sequence,
    int current_frame,
    int frame_count,
    bool playing,
    bool show_transport
) {
    // FirstUseEver (not Once): only seed the default dock node when the window has
    // no saved .ini entry, so a layout the user rearranged is restored on launch.
    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Scene");
    const bool focused = ImGui::IsWindowFocused();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    // The transport is drawn as an overlay (see below), so the scene fills the
    // whole window and does not resize when the transport moves between panes.
    const bool draw_transport = has_sequence && show_transport;
    const int width = std::max(1, static_cast<int>(avail.x));
    const int height = std::max(1, static_cast<int>(avail.y));

    const ImVec2 image_pos = ImGui::GetCursorScreenPos();
    // Flip V (uv0 top = 1, uv1 bottom = 0) because GL textures are bottom-up.
    ImGui::Image(framebuffer.texture(), ImVec2(static_cast<float>(width), static_cast<float>(height)), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
    // Hover is the scene image only, so the camera does not react to drags on the
    // transport bar below.
    const bool hovered = ImGui::IsItemHovered();

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

    show_controls = draw_controls_overlay(image_pos, width, has_sequence, playing, show_controls);

    // Transport overlay pinned to the bottom edge of the scene image.
    bool scrubbing = false;
    if (draw_transport) {
        const float strip_top = image_pos.y + static_cast<float>(height) - PLAYBACK_BAR_HEIGHT;
        const TransportState transport = draw_transport_bar(image_pos.x, strip_top, static_cast<float>(width), current_frame, frame_count, playing);
        current_frame = transport.current_frame;
        playing = transport.playing;
        scrubbing = transport.scrubbing;
    }

    ImGui::End();
    ImGui::PopStyleVar();
    return {width, height, hovered, focused, show_controls, current_frame, playing, scrubbing};
}

ImageViewResult draw_image_window(
    unsigned int texture,
    int texture_width,
    int texture_height,
    ImGuiID dock_id,
    bool has_sequence,
    int current_frame,
    int frame_count,
    bool playing,
    bool show_transport
) {
    // FirstUseEver (not Once): only seed the default dock node when the window has
    // no saved .ini entry, so a layout the user rearranged is restored on launch.
    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Image");
    const bool focused = ImGui::IsWindowFocused();
    const ImVec2 avail = ImGui::GetContentRegionAvail();

    // The transport is drawn as an overlay, so the image uses the full region and
    // does not resize when the transport moves between panes.
    const bool draw_transport = has_sequence && show_transport;
    const float region_width = std::max(1.0f, avail.x);
    const float region_height = std::max(1.0f, avail.y);

    const ImVec2 region_pos = ImGui::GetCursorScreenPos();
    bool hovered = false;
    if (texture != 0 && texture_width > 0 && texture_height > 0) {
        // Fit the image inside the region while preserving its aspect ratio, and
        // centre it so it does not stretch as the pane is resized.
        const float tex_aspect = static_cast<float>(texture_width) / static_cast<float>(texture_height);
        const float region_aspect = region_width / region_height;
        float draw_width = region_width;
        float draw_height = region_height;
        if (tex_aspect > region_aspect) {
            draw_height = region_width / tex_aspect;
        } else {
            draw_width = region_height * tex_aspect;
        }
        const float offset_x = (region_width - draw_width) * 0.5f;
        const float offset_y = (region_height - draw_height) * 0.5f;
        ImGui::SetCursorScreenPos(ImVec2(region_pos.x + offset_x, region_pos.y + offset_y));
        // No V flip: stb_image rows run top-to-bottom, matching imgui's UV space.
        ImGui::Image(static_cast<ImTextureID>(texture), ImVec2(draw_width, draw_height), ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));
        hovered = ImGui::IsItemHovered();
    } else {
        const char* note = has_sequence ? "No keypoint image for this frame." : "Load a sequence folder to see modeled frames.";
        const ImVec2 text_size = ImGui::CalcTextSize(note);
        ImGui::SetCursorScreenPos(ImVec2(region_pos.x + (region_width - text_size.x) * 0.5f, region_pos.y + (region_height - text_size.y) * 0.5f));
        ImGui::TextDisabled("%s", note);
    }

    // Transport overlay pinned to the bottom edge of the image region.
    bool scrubbing = false;
    if (draw_transport) {
        const float strip_top = region_pos.y + region_height - PLAYBACK_BAR_HEIGHT;
        const TransportState transport = draw_transport_bar(region_pos.x, strip_top, region_width, current_frame, frame_count, playing);
        current_frame = transport.current_frame;
        playing = transport.playing;
        scrubbing = transport.scrubbing;
    }

    ImGui::End();
    ImGui::PopStyleVar();
    return {hovered, focused, current_frame, playing, scrubbing};
}
