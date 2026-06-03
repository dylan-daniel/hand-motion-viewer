#include "window.h"

#include <algorithm>

#include <SDL.h>

namespace {
    // Allowance for the OS title bar / borders so the window's top stays reachable
    // even when its saved size nearly fills the monitor.
    constexpr int TITLE_BAR_HEIGHT = 40;
    // Smallest window we will ever restore to, regardless of the saved size.
    constexpr int MIN_WINDOW_WIDTH = 320;
    constexpr int MIN_WINDOW_HEIGHT = 240;

    /// Clamp ``value`` into [low, high], tolerating an inverted range.
    int clamp_range(int value, int low, int high) {
        if (high < low) {
            return low;
        }
        return std::max(low, std::min(value, high));
    }
} // namespace

std::vector<DisplayBounds> get_display_bounds() {
    std::vector<DisplayBounds> bounds;
    const int count = SDL_GetNumVideoDisplays();
    for (int index = 0; index < count; ++index) {
        SDL_Rect rect;
        if (SDL_GetDisplayBounds(index, &rect) == 0) {
            bounds.push_back({rect.x, rect.y, rect.w, rect.h});
        }
    }
    return bounds;
}

WindowGeometry compute_initial_geometry(const Config& config) {
    const std::vector<DisplayBounds> bounds = get_display_bounds();
    if (bounds.empty()) {
        // No monitor info available; trust the saved size as a last resort.
        WindowGeometry geometry;
        geometry.width = std::max(config.window_width, MIN_WINDOW_WIDTH);
        geometry.height = std::max(config.window_height, MIN_WINDOW_HEIGHT);
        geometry.x = config.window_x.value_or(0);
        geometry.y = config.window_y.value_or(0);
        geometry.display = 0;
        return geometry;
    }

    const int display = (config.window_display >= 0 && config.window_display < static_cast<int>(bounds.size())) ? config.window_display : 0;
    const DisplayBounds& monitor = bounds[static_cast<std::size_t>(display)];

    const int width = clamp_range(config.window_width, MIN_WINDOW_WIDTH, monitor.width);
    const int height = clamp_range(config.window_height, MIN_WINDOW_HEIGHT, monitor.height - TITLE_BAR_HEIGHT);

    int pos_x;
    int pos_y;
    if (!config.window_x || !config.window_y) {
        // No saved position: center the window on the chosen monitor.
        pos_x = monitor.x + (monitor.width - width) / 2;
        pos_y = monitor.y + (monitor.height - height) / 2;
    } else {
        pos_x = monitor.x + *config.window_x;
        pos_y = monitor.y + *config.window_y;
    }

    pos_x = clamp_range(pos_x, monitor.x, monitor.x + monitor.width - width);
    pos_y = clamp_range(pos_y, monitor.y + TITLE_BAR_HEIGHT, monitor.y + monitor.height - height);

    return WindowGeometry{pos_x, pos_y, width, height, display};
}

void set_relative_mouse(bool enabled) { SDL_SetRelativeMouseMode(enabled ? SDL_TRUE : SDL_FALSE); }

WindowGeometry read_current_geometry(SDL_Window* window) {
    int pos_x = 0;
    int pos_y = 0;
    int width = 0;
    int height = 0;
    SDL_GetWindowPosition(window, &pos_x, &pos_y);
    SDL_GetWindowSize(window, &width, &height);

    const int center_x = pos_x + width / 2;
    const int center_y = pos_y + height / 2;
    int display = 0;
    int index = 0;
    for (const DisplayBounds& monitor : get_display_bounds()) {
        if (monitor.x <= center_x && center_x < monitor.x + monitor.width && monitor.y <= center_y && center_y < monitor.y + monitor.height) {
            display = index;
            break;
        }
        ++index;
    }

    return WindowGeometry{pos_x, pos_y, width, height, display};
}

void store_geometry(Config& config, const WindowGeometry& geometry) {
    const std::vector<DisplayBounds> bounds = get_display_bounds();
    int monitor_x = 0;
    int monitor_y = 0;
    if (geometry.display >= 0 && geometry.display < static_cast<int>(bounds.size())) {
        monitor_x = bounds[static_cast<std::size_t>(geometry.display)].x;
        monitor_y = bounds[static_cast<std::size_t>(geometry.display)].y;
    }

    config.window_x = geometry.x - monitor_x;
    config.window_y = geometry.y - monitor_y;
    config.window_width = geometry.width;
    config.window_height = geometry.height;
    config.window_display = geometry.display;
}
