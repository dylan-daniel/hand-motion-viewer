#pragma once

// Window helpers: query per-monitor desktop geometry, turn the saved config into
// a placement that always lands fully on-screen, and toggle relative mouse mode.
// Unlike the Python viewer (which had to dig the SDL library out of pygame),
// these call SDL2 directly.

#include <vector>

#include "config.h"

struct SDL_Window;

/// A window's absolute desktop placement and the monitor it sits on.
struct WindowGeometry {
    int x = 0; // absolute desktop x of the window's top-left
    int y = 0; // absolute desktop y of the window's top-left
    int width = 0;
    int height = 0;
    int display = 0; // index of the monitor the window is on
};

/// A monitor's (x, y, width, height) rectangle in desktop space.
struct DisplayBounds {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

/// Every monitor's bounds in the virtual desktop.
std::vector<DisplayBounds> get_display_bounds();

/// Resolve the saved config into an on-screen window placement (clamped to the
/// target monitor, centred when nothing is saved).
WindowGeometry compute_initial_geometry(const Config& config);

/// Enable or disable relative mouse mode (cursor hidden, locked, pure deltas).
void set_relative_mouse(bool enabled);

/// Read the live window's placement and which monitor it is on (only meaningful
/// while windowed).
WindowGeometry read_current_geometry(SDL_Window* window);

/// Write a window placement back into the config as monitor-relative values.
void store_geometry(Config& config, const WindowGeometry& geometry);
