#include "ui/explorer.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include <SDL3/SDL_opengl.h>
#include <stb_image.h>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

namespace {
    namespace fs = std::filesystem;

    /// Decode a PNG and upload it as an RGBA GL texture, returning the texture name
    /// (0 if the file is missing or fails to decode). Linear filtered and clamped so
    /// it scales cleanly to the small size it is drawn at in the tree.
    unsigned int load_texture(const std::string& path) {
        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
        if (pixels == nullptr) {
            std::printf("Explorer icon failed to load %s: %s\n", path.c_str(), stbi_failure_reason());
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

    /// Display name for a folder path: its final component, or the whole path when
    /// that is empty (e.g. a drive root like ``C:\``).
    std::string folder_name(const fs::path& path) {
        const std::string name = path.filename().string();
        return name.empty() ? path.string() : name;
    }

    /// Natural-order ("human") comparison of two names. Each name is read as
    /// an alternating sequence of non-digit and digit runs; digit runs compare by
    /// numeric value rather than lexically, so e.g. ``T1_BabyView`` splits into
    /// (``T``, ``1``, ``_BabyView``) and sorts before ``T10_BabyView`` instead of
    /// after it (plain string order puts ``T10`` before ``T2``). Non-digit runs
    /// compare case-insensitively so ``a`` and ``A`` interleave naturally.
    bool natural_less(const std::string& left, const std::string& right) {
        const auto is_digit = [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)) != 0; };
        const auto lower = [](char ch) { return static_cast<char>(std::tolower(static_cast<unsigned char>(ch))); };

        std::size_t left_index = 0;
        std::size_t right_index = 0;
        while (left_index < left.size() && right_index < right.size()) {
            if (is_digit(left[left_index]) && is_digit(right[right_index])) {
                // Compare two numeric runs by value. Skip leading zeros, then the
                // run with more significant digits is the larger number; on a tie,
                // compare digit by digit.
                std::size_t left_end = left_index;
                std::size_t right_end = right_index;
                while (left_end < left.size() && is_digit(left[left_end])) {
                    ++left_end;
                }
                while (right_end < right.size() && is_digit(right[right_end])) {
                    ++right_end;
                }
                std::size_t left_start = left_index;
                std::size_t right_start = right_index;
                while (left_start < left_end - 1 && left[left_start] == '0') {
                    ++left_start;
                }
                while (right_start < right_end - 1 && right[right_start] == '0') {
                    ++right_start;
                }
                const std::size_t left_digits = left_end - left_start;
                const std::size_t right_digits = right_end - right_start;
                if (left_digits != right_digits) {
                    return left_digits < right_digits;
                }
                for (std::size_t offset = 0; offset < left_digits; ++offset) {
                    if (left[left_start + offset] != right[right_start + offset]) {
                        return left[left_start + offset] < right[right_start + offset];
                    }
                }
                left_index = left_end;
                right_index = right_end;
            } else {
                const char left_char = lower(left[left_index]);
                const char right_char = lower(right[right_index]);
                if (left_char != right_char) {
                    return left_char < right_char;
                }
                ++left_index;
                ++right_index;
            }
        }
        // One name is a prefix of the other (or they tie): the shorter sorts first.
        return (left.size() - left_index) < (right.size() - right_index);
    }

    /// Sort a node's kept children: folders before files (so containers group
    /// together above the exports they lead to), natural name order within each
    /// group.
    void sort_children(FileExplorer::Node& node) {
        std::sort(node.children.begin(), node.children.end(), [](const FileExplorer::Node& left, const FileExplorer::Node& right) {
            if (left.is_file != right.is_file) {
                return !left.is_file;
            }
            return natural_less(left.name, right.name);
        });
    }

    /// True once a kept child should stay in the pruned tree: a file child is
    /// always a real ``.hexport`` export (kept unconditionally); a folder child
    /// is kept only if it retained anything below it (otherwise it is a dead
    /// branch, dropped).
    bool keep_child(const FileExplorer::Node& child) { return child.is_file || !child.children.empty(); }

#ifdef _WIN32
    /// Convert a UTF-8 path to UTF-16 for the wide Win32 directory APIs.
    std::wstring widen(const std::string& utf8) {
        if (utf8.empty()) {
            return std::wstring();
        }
        const int length = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
        std::wstring result(static_cast<std::size_t>(length), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), result.data(), length);
        return result;
    }

    /// Convert a UTF-16 file name (as returned by FindFirstFile) back to UTF-8.
    std::string narrow(const wchar_t* wide, int wide_length) {
        if (wide_length <= 0) {
            return std::string();
        }
        const int length = WideCharToMultiByte(CP_UTF8, 0, wide, wide_length, nullptr, 0, nullptr, nullptr);
        std::string result(static_cast<std::size_t>(length), '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide, wide_length, result.data(), length, nullptr, nullptr);
        return result;
    }

    /// True if ``name`` ends with a case-insensitive ``.hexport`` extension. Tests
    /// the suffix in place — no path/extension allocation, which matters across
    /// the hundreds of thousands of files a data folder can hold.
    bool wname_is_hexport(const wchar_t* name) {
        static const wchar_t ext[] = L".hexport";
        const std::size_t ext_length = 8; // wcslen(L".hexport")
        const std::size_t name_length = std::wcslen(name);
        if (name_length < ext_length) {
            return false;
        }
        for (std::size_t offset = 0; offset < ext_length; ++offset) {
            if (std::towlower(name[name_length - ext_length + offset]) != ext[offset]) {
                return false;
            }
        }
        return true;
    }

    /// Recursively fill ``node`` (whose ``path``/``name`` are already set) with its
    /// pruned subtree: kept subfolders plus any ``.hexport`` files found directly
    /// inside it, both as ``children`` (sorted, folders first).
    ///
    /// Uses FindFirstFileExW with FIND_FIRST_EX_LARGE_FETCH rather than
    /// std::filesystem: the directory entry's type and name come straight from the
    /// WIN32_FIND_DATA, so there is no per-entry attribute syscall and no temporary
    /// path allocation. On a large data folder this is dramatically faster than the
    /// std::filesystem walk, which issues a stat per entry (see bench/). Runs on the
    /// worker thread; touches no shared state.
    void scan_children(FileExplorer::Node& node) {
        const std::wstring pattern = widen(node.path) + L"\\*";
        WIN32_FIND_DATAW find_data;
        HANDLE handle = FindFirstFileExW(pattern.c_str(), FindExInfoBasic, &find_data, FindExSearchNameMatch, nullptr, FIND_FIRST_EX_LARGE_FETCH);
        if (handle != INVALID_HANDLE_VALUE) {
            do {
                const wchar_t* name = find_data.cFileName;
                if (name[0] == L'.' && (name[1] == L'\0' || (name[1] == L'.' && name[2] == L'\0'))) {
                    continue; // skip "." and ".."
                }
                if ((find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
                    FileExplorer::Node child;
                    child.name = narrow(name, static_cast<int>(std::wcslen(name)));
                    child.path = node.path + "\\" + child.name;
                    scan_children(child);
                    if (keep_child(child)) {
                        node.children.push_back(std::move(child));
                    }
                } else if (wname_is_hexport(name)) {
                    FileExplorer::Node child;
                    child.is_file = true;
                    child.name = narrow(name, static_cast<int>(std::wcslen(name)));
                    child.path = node.path + "\\" + child.name;
                    node.children.push_back(std::move(child));
                }
            } while (FindNextFileW(handle, &find_data) != 0);
            FindClose(handle);
        }
        sort_children(node);
    }
#else
    /// True if ``name`` ends with a case-insensitive ``.hexport`` extension, tested
    /// in place without building a path/extension string.
    bool name_is_hexport(const std::string& name) {
        static const std::string ext = ".hexport";
        if (name.size() < ext.size()) {
            return false;
        }
        for (std::size_t offset = 0; offset < ext.size(); ++offset) {
            const char ch = static_cast<char>(std::tolower(static_cast<unsigned char>(name[name.size() - ext.size() + offset])));
            if (ch != ext[offset]) {
                return false;
            }
        }
        return true;
    }

    /// Portable std::filesystem fallback for non-Windows builds. Same contract as the
    /// Win32 ``scan_children`` above: a single pass classifies each entry as a kept
    /// subdirectory or an ``.hexport`` file (both become children of this node).
    void scan_children(FileExplorer::Node& node) {
        std::error_code error;
        fs::directory_iterator iterator(node.path, fs::directory_options::skip_permission_denied, error);
        if (!error) {
            for (const fs::directory_entry& entry : iterator) {
                std::error_code type_error;
                if (entry.is_directory(type_error)) {
                    FileExplorer::Node child;
                    child.path = entry.path().string();
                    child.name = folder_name(entry.path());
                    scan_children(child);
                    if (keep_child(child)) {
                        node.children.push_back(std::move(child));
                    }
                } else if (entry.is_regular_file(type_error) && name_is_hexport(entry.path().filename().string())) {
                    FileExplorer::Node child;
                    child.is_file = true;
                    child.path = entry.path().string();
                    child.name = entry.path().filename().string();
                    node.children.push_back(std::move(child));
                }
            }
        }
        sort_children(node);
    }
#endif

    /// Paint a node's icon and name over the tree-node row just submitted. The row
    /// itself is drawn with an empty label (so it stays full-width clickable via
    /// SpanFullWidth and keeps its expand arrow); here we blit the folder/file icon
    /// where the label would start, then the name after it. Drawing both ourselves
    /// is what lets the icon sit between the arrow and the text — imgui's tree node
    /// has no built-in slot for it. A missing icon texture (0) just draws the name.
    /// ``node_left`` is the node's own (indented) left edge, captured before the row
    /// was drawn. It is passed in rather than read from GetItemRectMin because
    /// SpanFullWidth stretches the item rect to the full row width, which would put
    /// the icon over the arrow and ignore nesting indent.
    void draw_node_label(const ExplorerIcons& icons, const std::string& name, bool is_file, bool expanded, float node_left) {
        const float icon_size = ImGui::GetFontSize(); // square icon, matched to text height
        const ImVec2 item_min = ImGui::GetItemRectMin();
        const ImVec2 item_max = ImGui::GetItemRectMax();
        const float label_x = node_left + ImGui::GetTreeNodeToLabelSpacing();
        const float center_y = (item_min.y + item_max.y) * 0.5f;
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        float text_x = label_x;
        const unsigned int texture = is_file ? icons.file : (expanded ? icons.folder_open : icons.folder_closed);
        if (texture != 0) {
            const ImVec2 icon_min(label_x, center_y - icon_size * 0.5f);
            const ImVec2 icon_max(label_x + icon_size, center_y + icon_size * 0.5f);
            draw_list->AddImage(static_cast<ImTextureID>(texture), icon_min, icon_max);
            text_x += icon_size + ImGui::GetStyle().ItemInnerSpacing.x;
        }
        draw_list->AddText(ImVec2(text_x, center_y - icon_size * 0.5f), ImGui::GetColorU32(ImGuiCol_Text), name.c_str());
    }

    /// The state shared, unchanged, by every node in one frame's tree walk: where to
    /// fold user actions, and the running stripe state used to shade alternate rows.
    /// Bundled so draw_node stays a two-argument (context, node) call as more drawing
    /// state is added, rather than growing its parameter list.
    struct TreeDraw {
        FileExplorer& explorer;
        ExplorerResult& result;
        float row_left = 0.0f;  // screen x where every row's stripe begins (full width, ignores indent)
        float row_width = 0.0f; // stripe width: the whole content region
        int row_index = 0;      // running count of rows drawn this frame, for alternating shading
    };

    /// Paint the alternating row background behind the row about to be drawn, then
    /// advance the row counter. Mimics a table's RowBg striping without a table: the
    /// stripe spans the full content width (so nesting indent doesn't ragged the left
    /// edge) and reuses the theme's TableRowBgAlt colour so it matches real tables.
    void stripe_row(TreeDraw& draw) {
        if (draw.row_index % 2 == 1) {
            const float row_top = ImGui::GetCursorScreenPos().y;
            // One row's pitch: the text line height plus the item spacing below it, so
            // each stripe covers exactly its row and the gap to the next (no overlap).
            const float row_pitch = ImGui::GetTextLineHeightWithSpacing();
            const ImU32 color = ImGui::GetColorU32(ImGuiCol_TableRowBgAlt);
            ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(draw.row_left, row_top), ImVec2(draw.row_left + draw.row_width, row_top + row_pitch), color);
        }
        ++draw.row_index;
    }

    /// Recursively draw one node and record any click on a ``.hexport`` leaf into
    /// ``draw.result``. The tree is already fully scanned, so this only walks
    /// memory. Only file leaves are openable; folders are pure containers that
    /// just expand/collapse via their arrow.
    void draw_node(TreeDraw& draw, FileExplorer::Node& node) {
        // A file node is always a leaf by construction (scan_children never gives
        // it children); a folder node with no children would have been pruned, so
        // in practice every leaf reaching this point is a file.
        const bool leaf = node.children.empty();

        // SpanFullWidth makes the node's hover/selection fill the row so it lines up
        // with the full-width alternating stripe painted behind it.
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanFullWidth;
        if (leaf) {
            flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            stripe_row(draw);
            // Empty label, path as the id: the name is painted by draw_node_label so
            // the icon can sit between the arrow and the text, and the path keeps ids
            // unique even when two files share a name in different folders.
            const float node_left = ImGui::GetCursorScreenPos().x;
            ImGui::TreeNodeEx(node.path.c_str(), flags, "%s", "");
            draw_node_label(draw.explorer.icons(), node.name, node.is_file, false, node_left);
            if (node.is_file && ImGui::IsItemClicked()) {
                draw.result.open_file = node.path;
            }
            return;
        }

        // Seed the initial open state exactly once: the root opens by default, and any
        // folder left open last session is restored. The user is free to collapse it
        // afterward (Once means this seed never fights later manual toggles).
        if (node.default_open || draw.explorer.should_open(node.path)) {
            ImGui::SetNextItemOpen(true, ImGuiCond_Once);
            node.default_open = false;
        }

        stripe_row(draw);
        const float node_left = ImGui::GetCursorScreenPos().x;
        const bool open = ImGui::TreeNodeEx(node.path.c_str(), flags, "%s", "");
        draw_node_label(draw.explorer.icons(), node.name, node.is_file, open, node_left);
        if (open) {
            draw.explorer.mark_expanded(node.path); // record for persisting the open state
            for (FileExplorer::Node& child : node.children) {
                draw_node(draw, child);
            }
            ImGui::TreePop();
        }
    }
} // namespace

ExplorerIcons::~ExplorerIcons() {
    if (folder_closed != 0) {
        glDeleteTextures(1, &folder_closed);
    }
    if (folder_open != 0) {
        glDeleteTextures(1, &folder_open);
    }
    if (file != 0) {
        glDeleteTextures(1, &file);
    }
    if (change_root != 0) {
        glDeleteTextures(1, &change_root);
    }
    if (refresh != 0) {
        glDeleteTextures(1, &refresh);
    }
}

void ExplorerIcons::load(const std::string& assets_dir) {
    const std::string prefix = assets_dir.empty() || assets_dir.back() == '/' || assets_dir.back() == '\\' ? assets_dir : assets_dir + "/";
    folder_closed = load_texture(prefix + "folder-blue.png");
    folder_open = load_texture(prefix + "folder-blue-open.png");
    file = load_texture(prefix + "file-hexport.png");
    change_root = load_texture(prefix + "folder-lucide.png");
    refresh = load_texture(prefix + "folder-sync.png");
}

FileExplorer::Node FileExplorer::scan_root(const std::string& root) {
    Node node;
    node.path = root;
    node.name = folder_name(fs::path(root));
    node.default_open = true; // root opens so its kept subfolders are visible
    // Fill the pruned, sorted children. Unlike a subtree the root is always kept
    // (even with no exports below) so the pane has something to show.
    scan_children(node);
    return node;
}

void FileExplorer::start_scan() {
    bool expected = false;
    if (!scanning_.compare_exchange_strong(expected, true)) {
        return; // a scan is already running; ignore overlapping requests
    }
    {
        std::lock_guard<std::mutex> guard(mutex_);
        has_ready_ = false;
    }
    const std::string root = root_path_;
    worker_.submit([this, root] {
        Node tree = scan_root(root);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            ready_root_ = std::move(tree);
            has_ready_ = true;
        }
        scanning_.store(false);
    });
}

void FileExplorer::set_root(const std::string& root) {
    root_.reset();
    root_path_.clear();

    std::error_code error;
    if (root.empty() || !fs::is_directory(root, error)) {
        return;
    }
    root_path_ = root;
    start_scan();
}

void FileExplorer::refresh() {
    if (root_path_.empty()) {
        return;
    }
    start_scan();
}

void FileExplorer::poll() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (has_ready_) {
        root_ = std::move(ready_root_);
        ready_root_.reset();
        has_ready_ = false;
    }
}

void FileExplorer::set_saved_open(std::vector<std::string> paths) {
    saved_open_ = std::unordered_set<std::string>(std::make_move_iterator(paths.begin()), std::make_move_iterator(paths.end()));
}

ExplorerResult draw_explorer_window(FileExplorer& explorer, ImGuiID dock_id) {
    ExplorerResult result;

    // Fold in a finished background scan (if any) before drawing this frame's tree.
    explorer.poll();

    ImGui::SetNextWindowDockID(dock_id, ImGuiCond_FirstUseEver);
    ImGui::Begin("Explorer");
    result.focused = ImGui::IsWindowFocused();
    result.hovered = ImGui::IsWindowHovered();

    if (!explorer.has_root()) {
        // Empty state: a muted folder icon, a heading, one line of guidance, and the
        // primary action — centred as a block a little above the pane's middle.
        const ImGuiStyle& style = ImGui::GetStyle();
        const ImVec2 origin = ImGui::GetCursorPos();
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const ExplorerIcons& icons = explorer.icons();

        const char* heading = "No data folder selected";
        const char* hint = "Choose a folder to scan for .hexport files.";
        const char* button_label = "Choose Data Folder";
        const float icon_size = 48.0f;
        const bool has_icon = icons.change_root != 0;
        const float gap = ImGui::GetTextLineHeight() * 0.6f;
        const float line_height = ImGui::GetTextLineHeight();
        const float button_width = ImGui::CalcTextSize(button_label).x + style.FramePadding.x * 2.0f;

        // Total block height, to place it slightly above centre.
        float block_height = line_height + style.ItemSpacing.y + line_height + gap + ImGui::GetFrameHeight();
        if (has_icon) {
            block_height += icon_size + gap;
        }
        ImGui::SetCursorPosY(origin.y + std::max(0.0f, (avail.y - block_height) * 0.4f));

        // Centre each line on its own width. Not clamped: when a line is wider than the
        // pane it stays centred and simply overflows both edges, rather than pinning left.
        const auto center_x = [&](float width) { ImGui::SetCursorPosX(origin.x + (avail.x - width) * 0.5f); };

        if (has_icon) {
            center_x(icon_size);
            const ImVec2 icon_pos = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddImage(
                static_cast<ImTextureID>(icons.change_root),
                icon_pos,
                ImVec2(icon_pos.x + icon_size, icon_pos.y + icon_size),
                ImVec2(0.0f, 0.0f),
                ImVec2(1.0f, 1.0f),
                ImGui::GetColorU32(ImGuiCol_TextDisabled)
            );
            ImGui::Dummy(ImVec2(icon_size, icon_size));
            ImGui::Dummy(ImVec2(0.0f, gap - style.ItemSpacing.y));
        }

        center_x(ImGui::CalcTextSize(heading).x);
        ImGui::TextUnformatted(heading);
        center_x(ImGui::CalcTextSize(hint).x);
        ImGui::TextDisabled("%s", hint);
        ImGui::Dummy(ImVec2(0.0f, gap - style.ItemSpacing.y));
        center_x(button_width);
        if (ImGui::Button(button_label)) {
            result.choose_root_requested = true;
        }
    } else {
        // Header row: the data-folder path on the left, then two right-aligned icon
        // buttons — change-root, then refresh (rescan). The path is clipped to stop
        // short of the buttons so a long path never renders under or past them.
        const ImGuiStyle& style = ImGui::GetStyle();
        const float icon = ImGui::GetFontSize();
        const ExplorerIcons& icons = explorer.icons();
        const float button_width = icon + style.FramePadding.x * 2.0f;
        const float buttons_width = button_width * 2.0f + style.ItemSpacing.x; // change + refresh

        const ImVec2 row_start = ImGui::GetCursorPos();
        const float region_width = ImGui::GetContentRegionAvail().x;

        // Tint icon buttons to the text colour so they track the theme.
        const ImVec4 tint = ImGui::GetStyleColorVec4(ImGuiCol_Text);
        const ImVec4 no_bg(0.0f, 0.0f, 0.0f, 0.0f);
        const ImVec2 uv0(0.0f, 0.0f);
        const ImVec2 uv1(1.0f, 1.0f);

        // Path on the left, vertically centred to the button row and clipped.
        ImGui::AlignTextToFramePadding();
        const float text_width = std::max(0.0f, region_width - buttons_width - style.ItemSpacing.x);
        const ImVec2 text_pos = ImGui::GetCursorScreenPos();
        ImGui::PushClipRect(text_pos, ImVec2(text_pos.x + text_width, text_pos.y + ImGui::GetFrameHeight()), true);
        ImGui::TextDisabled("%s", explorer.root_path().c_str());
        ImGui::PopClipRect();

        // Change-root button. Falls back to a text button if its texture is missing.
        ImGui::SetCursorPos(ImVec2(row_start.x + region_width - buttons_width, row_start.y));
        bool change_clicked = false;
        if (icons.change_root != 0) {
            change_clicked = ImGui::ImageButton("change_root", static_cast<ImTextureID>(icons.change_root), ImVec2(icon, icon), uv0, uv1, no_bg, tint);
        } else {
            change_clicked = ImGui::Button("Change");
        }
        if (change_clicked) {
            result.choose_root_requested = true;
        }
        ImGui::SetItemTooltip("Change data folder");

        // Refresh (rescan) button to the right of it, disabled while a scan runs.
        ImGui::SameLine();
        const bool scanning = explorer.scanning();
        ImGui::BeginDisabled(scanning);
        bool refresh_clicked = false;
        if (icons.refresh != 0) {
            refresh_clicked = ImGui::ImageButton("refresh_root", static_cast<ImTextureID>(icons.refresh), ImVec2(icon, icon), uv0, uv1, no_bg, tint);
        } else {
            refresh_clicked = ImGui::Button("Refresh");
        }
        ImGui::EndDisabled();
        if (refresh_clicked) {
            explorer.refresh();
        }
        ImGui::SetItemTooltip("Rescan this folder");

        ImGui::Separator();

        // Walk the folder tree. The stripe origin/width are captured here, at the
        // root indent level, so every row's alternating background spans the full
        // content width regardless of how deep the row is nested.
        if (FileExplorer::Node* root = explorer.root_node()) {
            explorer.clear_live_expanded(); // rebuilt from the open folders walked below
            TreeDraw draw{explorer, result, ImGui::GetCursorScreenPos().x, ImGui::GetContentRegionAvail().x, 0};
            draw_node(draw, *root);
        } else if (scanning) {
            ImGui::TextDisabled("Scanning...");
        } else {
            ImGui::TextDisabled("No .hexport files found.");
        }
    }

    ImGui::End();
    return result;
}
