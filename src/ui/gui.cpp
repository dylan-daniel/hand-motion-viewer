#include "ui/gui.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>
#include <imgui_freetype.h>
#include <stb_image.h>

namespace {
    namespace fs = std::filesystem;

    /// Decode a PNG and upload it as an RGBA GL texture, returning the texture name
    /// (0 if the file is missing or fails to decode). Linear filtered and clamped.
    unsigned int load_texture(const std::string& path) {
        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
        if (pixels == nullptr) {
            std::printf("Transport icon failed to load %s: %s\n", path.c_str(), stbi_failure_reason());
            return 0;
        }
        unsigned int texture = 0;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        glBindTexture(GL_TEXTURE_2D, 0);
        stbi_image_free(pixels);
        return texture;
    }

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

    /// Path to the first .ttf in the bundled assets/fonts/ directory next to the binary.
    std::string find_bundled_font() {
        std::string base = "assets/fonts";
        const char* base_path = SDL_GetBasePath();
        if (base_path != nullptr) {
            base = std::string(base_path) + "assets/fonts";
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
        ImGui::SetCursorScreenPos(ImVec2(top_left.x + width - button_width, top_left.y + 8.0f));
        if (ImGui::Button(label)) {
            show_panel = !show_panel;
        }
        if (!show_panel) {
            return show_panel;
        }

        const float panel_width = 420.0f;
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

    // Playback transport (play/pause button + scrubber + frame label) drawn as an
    // overlay along the bottom of whichever pane currently carries it. ``strip_top``
    // is the screen Y where the overlay strip begins; a translucent backing is
    // painted behind it so the controls stay legible over the scene. Updates and
    // returns the transport state, including whether the scrubber is being dragged.
    // Bounds for the playback-speed multiplier exposed by the gauge slider.
    constexpr float MIN_PLAYBACK_SPEED = 0.5f;
    constexpr float MAX_PLAYBACK_SPEED = 4.0f;

    TransportState draw_transport_bar(float left, float strip_top, float width, const Transport& transport) {
        int current_frame = transport.current_frame;
        const int frame_count = transport.frame_count;
        bool playing = transport.playing;
        float speed = transport.speed;

        // Translucent backing so the controls read over any scene content beneath.
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        const ImU32 backing = ImGui::ColorConvertFloat4ToU32(ImVec4(0.08f, 0.08f, 0.10f, 0.65f));
        draw_list->AddRectFilled(ImVec2(left, strip_top), ImVec2(left + width, strip_top + PLAYBACK_BAR_HEIGHT), backing);

        const float row_height = ImGui::GetFrameHeight();
        ImGui::SetCursorScreenPos(ImVec2(left + 8.0f, strip_top + (PLAYBACK_BAR_HEIGHT - row_height) * 0.5f));

        // Play/pause as a filled icon button, tinted to the text colour; falls back
        // to a text button if the icon texture is missing.
        float button_width = 70.0f;
        const unsigned int play_pause_icon = playing ? transport.pause_icon : transport.play_icon;
        if (play_pause_icon != 0) {
            const float icon_size = ImGui::GetFontSize();
            if (ImGui::ImageButton(
                    "transport_play_pause",
                    static_cast<ImTextureID>(play_pause_icon),
                    ImVec2(icon_size, icon_size),
                    ImVec2(0.0f, 0.0f),
                    ImVec2(1.0f, 1.0f),
                    ImVec4(0.0f, 0.0f, 0.0f, 0.0f),
                    ImGui::GetStyleColorVec4(ImGuiCol_Text)
                )) {
                playing = !playing;
            }
            button_width = ImGui::GetItemRectSize().x;
        } else {
            if (ImGui::Button(playing ? "Pause" : "Play", ImVec2(70.0f, 0.0f))) {
                playing = !playing;
            }
        }

        // Stretch the scrubber to fill the gap between the play button and the label,
        // leaving room for the frame label and the speed (gauge) button on the right.
        ImGui::SameLine();
        char label[48];
        std::snprintf(label, sizeof(label), "frame %d / %d", current_frame + 1, frame_count);
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        // The gauge button is an icon button the same size as the play button.
        const float speed_button_width = button_width;
        const float slider_width = width - 8.0f - button_width - spacing - ImGui::CalcTextSize(label).x - spacing - speed_button_width - spacing - 8.0f;
        ImGui::SetNextItemWidth(std::max(1.0f, slider_width));
        // Empty format: the standalone label shows the 1-based frame instead.
        ImGui::SliderInt("##frame", &current_frame, 0, frame_count - 1, "");
        // True while the user holds and drags the scrubber, so playback can be
        // suspended and the hands don't jitter between the dragged and next frame.
        const bool scrubbing = ImGui::IsItemActive();

        ImGui::SameLine();
        ImGui::TextUnformatted(label);

        // Speed (gauge) button at the far right: clicking opens a small popup with a
        // vertical slider for the playback-rate multiplier.
        ImGui::SameLine();
        const char* speed_popup = "transport_speed_popup";
        if (transport.speed_icon != 0) {
            const float icon_size = ImGui::GetFontSize();
            if (ImGui::ImageButton(
                    "transport_speed",
                    static_cast<ImTextureID>(transport.speed_icon),
                    ImVec2(icon_size, icon_size),
                    ImVec2(0.0f, 0.0f),
                    ImVec2(1.0f, 1.0f),
                    ImVec4(0.0f, 0.0f, 0.0f, 0.0f),
                    ImGui::GetStyleColorVec4(ImGuiCol_Text)
                )) {
                ImGui::OpenPopup(speed_popup);
            }
        } else {
            char speed_label[16];
            std::snprintf(speed_label, sizeof(speed_label), "%.1fx", speed);
            if (ImGui::Button(speed_label)) {
                ImGui::OpenPopup(speed_popup);
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Playback speed (%.1fx)", speed);
        }
        // Anchor the popup above the button (pivot at its own bottom-left), so the
        // slider rises out of the gauge instead of dropping off the window bottom.
        const ImVec2 button_min = ImGui::GetItemRectMin();
        ImGui::SetNextWindowPos(ImVec2(button_min.x, button_min.y - spacing), ImGuiCond_Always, ImVec2(0.0f, 1.0f));
        // Solid black backing; window padding makes the box wider/taller than the
        // slider and, being symmetric, centres the slider inside it.
        const float slider_box_width = 18.0f; // matches the imgui demo's vertical sliders
        const float slider_box_height = 140.0f;
        const float pad_x = std::max(4.0f, (speed_button_width - slider_box_width) * 0.5f);
        ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(pad_x, pad_x));
        if (ImGui::BeginPopup(speed_popup)) {
            // Bare vertical slider (empty format => no text on the grab); the value
            // shows only as a tooltip while hovering or dragging, like the imgui demo.
            ImGui::VSliderFloat("##speed", ImVec2(slider_box_width, slider_box_height), &speed, MIN_PLAYBACK_SPEED, MAX_PLAYBACK_SPEED, "");
            if (ImGui::IsItemActive() || ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%.1fx", speed);
            }
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();

        return {current_frame, playing, scrubbing, speed};
    }
} // namespace

std::pair<ImFont*, ImFont*> load_fonts(ImGuiIO& io) {
    // Rasterize glyphs with FreeType using light hinting: stems are grid-fit to the
    // pixel grid (crisp UI text) without the heavier auto-hinter distorting the
    // bundled font's letter shapes. Applies to every font baked below.
    io.Fonts->FontLoaderFlags = ImGuiFreeTypeBuilderFlags_LightHinting;

    const std::string font_path = find_bundled_font();
    if (font_path.empty()) {
        std::printf("No .ttf found in fonts/ — using imgui's default font.\n");
        ImFont* default_font = io.Fonts->AddFontDefault();
        return {default_font, default_font};
    }
    ImFont* ui_font = io.Fonts->AddFontFromFileTTF(font_path.c_str(), UI_FONT_SIZE);

    // The FPS/status overlay sits as white text directly over the 3D scene, so bump
    // its stroke weight a touch (RasterizerMultiply > 1 thickens glyphs) to make it
    // read more boldly against busy backgrounds. Hinting at this size barely moves
    // the edges, so weight is the lever that actually changes its appearance.
    ImFontConfig fps_config;
    fps_config.RasterizerMultiply = 1.15f;
    ImFont* fps_font = io.Fonts->AddFontFromFileTTF(font_path.c_str(), FPS_FONT_SIZE, &fps_config);
    return {ui_font, fps_font};
}

TransportIcons::~TransportIcons() {
    if (play != 0) {
        glDeleteTextures(1, &play);
    }
    if (pause != 0) {
        glDeleteTextures(1, &pause);
    }
}

void TransportIcons::load(const std::string& icons_dir) {
    const std::string prefix = icons_dir.empty() || icons_dir.back() == '/' || icons_dir.back() == '\\' ? icons_dir : icons_dir + "/";
    play = load_texture(prefix + "play.png");
    pause = load_texture(prefix + "pause.png");
    speed = load_texture(prefix + "gauge.png");
}

MenuResult draw_menu_bar(MenuState state) {
    bool csv_requested = false;
    bool images_requested = false;
    bool sam3_requested = false;
    bool da3_requested = false;
    bool tracking_requested = false;
    bool baby_hand_idx_requested = false;
    bool k_metric_csv_requested = false;
    bool production_csv_requested = false;

    // Extra padding makes the bar taller; pop it right after begin so dropdown
    // contents don't grow too.
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 8.0f));
    const bool opened = ImGui::BeginMainMenuBar();
    ImGui::PopStyleVar();
    if (opened) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Trial CSV")) {
                csv_requested = true;
            }
            if (ImGui::MenuItem("Set Images Folder")) {
                images_requested = true;
            }
            if (ImGui::MenuItem("Set SAM3 Folder")) {
                sam3_requested = true;
            }
            if (ImGui::MenuItem("Set DA3 Folder")) {
                da3_requested = true;
            }
            if (ImGui::MenuItem("Set Tracking Folder")) {
                tracking_requested = true;
            }
            if (ImGui::MenuItem("Set Baby-Hand-Idx Folder")) {
                baby_hand_idx_requested = true;
            }
            if (ImGui::MenuItem("Set K-Metric CSV")) {
                k_metric_csv_requested = true;
            }
            if (ImGui::MenuItem("Set Production Log CSV")) {
                production_csv_requested = true;
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Settings")) {
            ImGui::Checkbox("Transparent Hands", &state.hand_translucent);
            ImGui::Checkbox("Free Camera", &state.free_camera);
            // The marker shows the orbit camera's look-at point, which the free
            // camera lacks, so grey it out while free camera is on.
            ImGui::BeginDisabled(state.free_camera);
            ImGui::Checkbox("Camera Marker", &state.show_camera_marker);
            ImGui::EndDisabled();
            ImGui::EndMenu();
        }

        // Right-aligned path of the open trial CSV, always visible (even fullscreen)
        // without crowding any pane. Drawn via the draw list and clipped to the space
        // after the menus so a long path can't overlap File/Settings — when it
        // doesn't fit, the left of the path is clipped, keeping the tail (the actual
        // file) at the right edge.
        if (!state.open_path.empty()) {
            const ImVec2 window_pos = ImGui::GetWindowPos();
            const float window_width = ImGui::GetWindowWidth();
            const float window_height = ImGui::GetWindowHeight();
            const float right_pad = 12.0f;
            const float region_left = window_pos.x + ImGui::GetCursorPosX();
            const float region_right = window_pos.x + window_width - right_pad;
            if (region_right > region_left) {
                const char* path = state.open_path.c_str();
                const float text_width = ImGui::CalcTextSize(path).x;
                const float text_x = region_right - text_width; // right-aligned (may sit left of region, then clips)
                const float text_y = window_pos.y + (window_height - ImGui::GetTextLineHeight()) * 0.5f;
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                draw_list->PushClipRect(ImVec2(region_left, window_pos.y), ImVec2(region_right, window_pos.y + window_height), true);
                draw_list->AddText(ImVec2(text_x, text_y), ImGui::GetColorU32(ImGuiCol_TextDisabled), path);
                draw_list->PopClipRect();
            }
        }

        ImGui::EndMainMenuBar();
    }
    return {
        state,     csv_requested,   images_requested,        sam3_requested,        da3_requested,
        tracking_requested, baby_hand_idx_requested, k_metric_csv_requested, production_csv_requested
    };
}

ViewportResult draw_viewport_window(
    const Framebuffer& framebuffer, ImGuiID dock_id, float fps, ImFont* fps_font, const std::string& status, bool show_controls, const Transport& transport
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
    const bool draw_transport = transport.has_sequence && transport.show_transport;
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

    show_controls = draw_controls_overlay(image_pos, width, transport.has_sequence, transport.playing, show_controls);

    // Transport overlay pinned to the bottom edge of the scene image. Seed the echo
    // from the incoming state so an unchanged frame still reports it back.
    TransportState echo{transport.current_frame, transport.playing, false};
    if (draw_transport) {
        const float strip_top = image_pos.y + static_cast<float>(height) - PLAYBACK_BAR_HEIGHT;
        echo = draw_transport_bar(image_pos.x, strip_top, static_cast<float>(width), transport);
    }

    ImGui::End();
    ImGui::PopStyleVar();
    return {width, height, hovered, focused, show_controls, echo};
}

ImageViewResult draw_image_window(const char* title, unsigned int texture, int texture_width, int texture_height, ImGuiID dock_id, const Transport& transport) {
    // FirstUseEver (not Once): only seed the default dock node when the window has
    // no saved .ini entry, so a layout the user rearranged is restored on launch.
    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin(title);
    const bool focused = ImGui::IsWindowFocused();
    const ImVec2 avail = ImGui::GetContentRegionAvail();

    // The transport is drawn as an overlay, so the image uses the full region and
    // does not resize when the transport moves between panes.
    const bool draw_transport = transport.has_sequence && transport.show_transport;
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
        const char* note = transport.has_sequence ? "No keypoint image for this frame." : "Load a mesh sequence folder to see modeled frames.";
        const ImVec2 text_size = ImGui::CalcTextSize(note);
        ImGui::SetCursorScreenPos(ImVec2(region_pos.x + (region_width - text_size.x) * 0.5f, region_pos.y + (region_height - text_size.y) * 0.5f));
        ImGui::TextDisabled("%s", note);
    }

    // Transport overlay pinned to the bottom edge of the image region. Seed the
    // echo from the incoming state so an unchanged frame still reports it back.
    TransportState echo{transport.current_frame, transport.playing, false};
    if (draw_transport) {
        const float strip_top = region_pos.y + region_height - PLAYBACK_BAR_HEIGHT;
        echo = draw_transport_bar(region_pos.x, strip_top, region_width, transport);
    }

    ImGui::End();
    ImGui::PopStyleVar();
    return {hovered, focused, echo};
}

HandPaneState draw_hand_pane(HandPaneState state, ImGuiID dock_id) {
    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin("Hands");
    ImGui::Checkbox("Hide duplicates", &state.hide_duplicates);
    ImGui::Checkbox("Hide adult hands", &state.hide_adults);
    ImGui::End();
    return state;
}

CameraPaneState draw_camera_pane(CameraPaneState state, ImGuiID dock_id) {
    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin("Camera");
    ImGui::TextUnformatted("Recording-camera intrinsics (pixels)");
    state.intrinsics_committed = false;
    ImGui::DragFloat("fx", &state.fx, 1.0f, 1.0f, 20000.0f);
    state.intrinsics_committed |= ImGui::IsItemDeactivatedAfterEdit();
    ImGui::DragFloat("fy", &state.fy, 1.0f, 1.0f, 20000.0f);
    state.intrinsics_committed |= ImGui::IsItemDeactivatedAfterEdit();
    ImGui::DragFloat("cx", &state.cx, 1.0f, 0.0f, 8000.0f);
    state.intrinsics_committed |= ImGui::IsItemDeactivatedAfterEdit();
    ImGui::DragFloat("cy", &state.cy, 1.0f, 0.0f, 8000.0f);
    state.intrinsics_committed |= ImGui::IsItemDeactivatedAfterEdit();
    ImGui::Separator();
    ImGui::DragFloat("k_metric", &state.k_metric, 0.001f, 0.01f, 5.0f);
    state.intrinsics_committed |= ImGui::IsItemDeactivatedAfterEdit();
    if (!state.calibrated) {
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.2f, 1.0f), "k_metric not verified for this focal length");
        ImGui::TextWrapped("fx/fy/cx/cy were auto-detected from the trial's CSV, but k_metric is not portable across "
                            "focal lengths -- the value shown is whatever was set before opening this trial. Hand "
                            "depth/size will be wrong until a real calibration for this focal length is known.");
    }
    ImGui::Separator();
    ImGui::Checkbox("Prefer DA3 depth for hands", &state.prefer_da3_hand_depth);
    ImGui::TextWrapped(
        state.prefer_da3_hand_depth ? "Samples the wrist depth from the DA3 depth map (needs SAM3/DA3 folders set), "
                                       "falling back to k_metric * cam_t.z per hand/frame when no robust sample exists."
                                     : "Always uses k_metric * cam_t.z (HaMeR's own depth) -- ignores DA3 for hand "
                                       "placement even if SAM3/DA3 folders are set. The tracked object is unaffected "
                                       "either way; it always uses DA3."
    );
    ImGui::Separator();
    ImGui::Checkbox("Temporal smoothing (sigma=3 frames)", &state.smoothing_enabled);
    ImGui::TextWrapped(
        state.smoothing_enabled
            ? "Hand wrist depth and the tracked object's pose are each Gaussian-smoothed across the whole trial "
              "(matching render_scene_video.py's default). Loading/reloading the trial takes noticeably longer."
            : "Each frame keeps its own raw resolved depth/pose -- no whole-trial smoothing pass, faster to (re)load."
    );
    ImGui::End();
    return state;
}

ObjectPaneState draw_object_pane(ObjectPaneState state, ImGuiID dock_id) {
    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin("Object");
    static constexpr const char* SHAPE_NAMES[3] = {"None", "Cube", "Sphere"};
    if (ImGui::BeginCombo("Shape", SHAPE_NAMES[std::clamp(state.shape, 0, 2)])) {
        for (int shape = 0; shape < 3; ++shape) {
            const bool selected = state.shape == shape;
            if (ImGui::Selectable(SHAPE_NAMES[shape], selected)) {
                state.shape = shape;
            }
            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    char label_buffer[64];
    std::snprintf(label_buffer, sizeof(label_buffer), "%s", state.object_label.c_str());
    if (ImGui::InputText("SAM3 label", label_buffer, sizeof(label_buffer))) {
        state.object_label = label_buffer;
    }
    ImGui::DragFloat("Size (m)", &state.size_m, 0.001f, 0.001f, 1.0f, "%.3f");
    ImGui::TextDisabled("Sphere: diameter. Cube: edge length.");
    if (state.auto_detected) {
        ImGui::TextDisabled("Auto-detected from the production log for this trial.");
    } else {
        ImGui::TextColored(
            ImVec4(1.0f, 0.65f, 0.2f, 1.0f), "Not found in the production log -- set manually, or set File > Set Production Log CSV."
        );
    }
    ImGui::End();
    return state;
}
