#pragma once

// Explorer pane: a folder tree rooted at a chosen data folder, pruned to the
// folders that lead to mesh sequences.
//
// On set_root (and on an explicit refresh) the whole subtree is walked once on a
// background thread, keeping only folders that directly contain ``.hmesh`` files
// or that have a descendant which does — so dead branches (no sequences anywhere
// below) are dropped while the path to every sequence folder is preserved. The
// finished tree is swapped in on the UI thread; drawing each frame only walks the
// in-memory tree. Only folders that directly hold ``.hmesh`` files are openable;
// the rest are pure containers that can merely expand/collapse.

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <imgui.h>

#include "util/worker_queue.h"

/// GL textures for the icons drawn beside each tree node and on the toolbar.
/// Loaded once after the GL context exists (the handles are plain
/// ``GL_TEXTURE_2D`` names) and freed on destruction. A handle left at 0 (its PNG
/// was missing) simply draws no icon, so the explorer still works without the
/// asset. Swapping in a different icon later is a matter of pointing ``load`` at a
/// different file — see ``assets/``.
struct ExplorerIcons {
    ExplorerIcons() = default;
    ~ExplorerIcons();

    ExplorerIcons(const ExplorerIcons&) = delete;
    ExplorerIcons& operator=(const ExplorerIcons&) = delete;

    /// Load the icon PNGs from ``assets_dir`` (the assets/icons/ folder next to the
    /// binary). Decodes synchronously and uploads to GL, so call once on the render
    /// thread after the GL context is current. A file that fails to load leaves its
    /// handle at 0.
    void load(const std::string& assets_dir);

    unsigned int folder_closed = 0; // shown for a collapsed folder
    unsigned int folder_open = 0;   // shown for an expanded folder
    unsigned int change_root = 0;   // the "change data folder" toolbar button (white, tinted at draw time)
    unsigned int refresh = 0;       // the "rescan current folder" toolbar button (white, tinted at draw time)
};

/// A folder tree pruned to sequence-bearing branches. Built in full by a single
/// background scan rather than lazily, so drawing never touches the disk.
class FileExplorer {
public:
    /// One folder in the pruned tree. ``children`` is the kept subfolders (those
    /// with ``.hmesh`` files somewhere below). ``has_meshes`` means this folder
    /// directly contains ``.hmesh`` files, which is what makes it openable.
    struct Node {
        std::string path;
        std::string name;
        std::vector<Node> children;
        bool has_meshes = false;   // directly contains .hmesh files (=> openable)
        bool default_open = false; // seed imgui's initial open state once (the root)
    };

    /// Point the explorer at ``root`` and kick off a background scan of its whole
    /// subtree. An empty or non-directory path clears the tree. Safe to call again
    /// to switch roots.
    void set_root(const std::string& root);

    /// Re-scan the current root on the background worker. No-op if no root is set
    /// or a scan is already running.
    void refresh();

    /// Fold in a finished background scan, if any. Call once per frame on the UI
    /// thread before drawing.
    void poll();

    bool has_root() const { return !root_path_.empty(); }
    const std::string& root_path() const { return root_path_; }
    bool scanning() const { return scanning_.load(); }

    /// The current tree root, or null until the first scan completes.
    Node* root_node() { return root_ ? &*root_ : nullptr; }

    /// The per-node icon textures, loaded once by the caller after GL is ready.
    ExplorerIcons& icons() { return icons_; }
    const ExplorerIcons& icons() const { return icons_; }

    // ── Persisted expansion state ──────────────────
    // The tree's open/closed state lives in imgui's per-window storage, which is not
    // saved across runs. To restore it, the caller seeds the previously-open folder
    // paths with set_saved_open at startup; draw_node uses should_open to re-open
    // them once. Each frame draw_node rebuilds the live open set (clear_live_expanded
    // then mark_expanded per visible open folder), which the caller reads back with
    // expanded_paths to persist on quit.

    /// Seed the folders to re-open on first appearance (from the saved config).
    void set_saved_open(std::vector<std::string> paths);

    /// Whether ``path`` should start expanded (it was open last session).
    bool should_open(const std::string& path) const { return saved_open_.count(path) != 0; }

    /// Reset the live open-folder set; call once before walking the tree each frame.
    void clear_live_expanded() { live_open_.clear(); }

    /// Record ``path`` as currently expanded (called while drawing an open folder).
    void mark_expanded(const std::string& path) { live_open_.insert(path); }

    /// The folders currently expanded, for persisting to the config on quit.
    std::vector<std::string> expanded_paths() const { return {live_open_.begin(), live_open_.end()}; }

private:
    /// Submit a full scan of root_path_ to the worker (if one isn't already running).
    void start_scan();

    /// Walk ``root`` fully, building the pruned tree's root node. Runs on the worker
    /// thread; touches no shared state.
    static Node scan_root(const std::string& root);

    std::string root_path_;    // selected root folder ("" = none); UI thread
    std::optional<Node> root_; // displayed tree; UI thread (null until first scan done)
    ExplorerIcons icons_;      // node/toolbar icon textures, owned for the explorer's lifetime

    std::unordered_set<std::string> saved_open_; // folders to re-open (from last session); seeds open state once
    std::unordered_set<std::string> live_open_;  // folders currently expanded this frame; persisted on quit

    // Background-scan handoff. scanning_ is atomic so the UI can read it without the
    // lock; ready_root_ is guarded and moved into root_ by poll().
    std::atomic<bool> scanning_{false};
    std::mutex mutex_;
    bool has_ready_ = false;
    std::optional<Node> ready_root_;

    WorkerQueue worker_; // declared last so it joins before the state above is destroyed
};

/// What the user did in the Explorer pane this frame.
struct ExplorerResult {
    bool hovered = false;
    bool focused = false;
    std::optional<std::string> open_folder; // a sequence folder was clicked: load it
    bool choose_root_requested = false;     // the "Choose Data Folder" / change button was pressed
};

/// Draw the dockable "Explorer" window. Polls for a finished scan and may start a
/// refresh when its button is pressed. Returns the user's actions for the caller.
ExplorerResult draw_explorer_window(FileExplorer& explorer, ImGuiID dock_id);
