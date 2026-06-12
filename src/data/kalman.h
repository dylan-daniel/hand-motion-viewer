#pragma once

// Temporal smoothing of the noisy per-frame hand motion with a constant-velocity
// Kalman filter. The hands are recovered per video frame by a transformer + MANO
// fit, so each frame's vertices carry independent estimation noise; played back
// at the source 30fps this reads as a visible jitter even though the underlying
// motion tracks well. Filtering each coordinate over time removes that jitter
// while preserving the real motion.
//
// Every scalar coordinate of every surface vertex and joint is tracked by its
// own independent 1D constant-velocity filter with state [position, velocity].
// The vertices are filtered independently rather than as a rigid body: the MANO
// noise is small relative to the hand, so applying identical smoothing to every
// coordinate preserves the hand's shape while removing the per-frame wobble.
//
// This header holds the reusable filter core (a single scalar's
// predict/update/smooth) and the whole-sequence forward-backward smoother used
// to precompute smoothed frames.

#include <vector>

#include "data/mesh_sequence.h"

/// Tuning for the constant-velocity model. Only the ratio of the two matters for
/// the steady-state amount of smoothing: a smaller ``process_noise`` (or larger
/// ``measurement_noise``) trusts the motion model more and smooths harder, while
/// a larger ``process_noise`` follows each raw measurement more closely.
struct KalmanParams {
    // Spectral density of the model's acceleration (process) noise.
    float process_noise = 0.01f;
    // Variance of a single raw per-frame position measurement.
    float measurement_noise = 1.0f;
};

/// Forward-backward (Rauch-Tung-Striebel) smoothing of a whole sequence of
/// frames. ``frames[f]`` holds that frame's hands in slot order; the result has
/// the identical shape with the surface vertices and joints replaced by their
/// smoothed positions. Hands are tracked across frames by slot index; a slot's
/// track is broken (and the filter restarted) wherever the hand disappears, its
/// handedness flips, or its vertex count changes. Because the backward pass uses
/// future frames, the smoothed motion has no playback lag.
std::vector<Frame> smooth_sequence(const std::vector<Frame>& frames, const KalmanParams& params = {});
