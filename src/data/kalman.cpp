#include "data/kalman.h"

#include <algorithm>
#include <cstddef>

namespace {
    // ── Flattening a hand to a flat coordinate vector ──
    // A hand's filtered quantities are its surface vertices followed by its joint
    // positions, each as three consecutive scalars. The two helpers below convert
    // between a HandData and that flat layout so the filter can treat a hand as a
    // plain list of independent scalars.

    std::size_t hand_scalar_count(const HandData& hand) { return (hand.verts.size() + hand.joints.size()) * 3; }

    std::vector<float> flatten_hand(const HandData& hand) {
        std::vector<float> flat;
        flat.reserve(hand_scalar_count(hand));
        for (const glm::vec3& vertex : hand.verts) {
            flat.push_back(vertex.x);
            flat.push_back(vertex.y);
            flat.push_back(vertex.z);
        }
        for (const glm::vec3& joint : hand.joints) {
            flat.push_back(joint.x);
            flat.push_back(joint.y);
            flat.push_back(joint.z);
        }
        return flat;
    }

    void apply_flat_to_hand(HandData& hand, const std::vector<float>& flat) {
        std::size_t cursor = 0;
        for (glm::vec3& vertex : hand.verts) {
            vertex.x = flat[cursor++];
            vertex.y = flat[cursor++];
            vertex.z = flat[cursor++];
        }
        for (glm::vec3& joint : hand.joints) {
            joint.x = flat[cursor++];
            joint.y = flat[cursor++];
            joint.z = flat[cursor++];
        }
    }

    // ── 2x2 symmetric covariance helpers (row-major a,b / b,d) ──
    struct Mat2 {
        float a = 0.0f; // (0,0)
        float b = 0.0f; // (0,1) == (1,0) for the symmetric covariances used here
        float c = 0.0f; // (1,0) — kept distinct for the non-symmetric smoother gain
        float d = 0.0f; // (1,1)
    };

    // One scalar's filtered/predicted record at a frame, retained for the backward
    // smoother pass.
    struct ScalarRecord {
        float filt_pos = 0.0f; // filtered mean position
        float filt_vel = 0.0f; // filtered mean velocity
        Mat2 filt_cov;         // filtered covariance
        float pred_pos = 0.0f; // one-step predicted mean position
        float pred_vel = 0.0f; // one-step predicted mean velocity
        Mat2 pred_cov;         // one-step predicted covariance
    };

    /// Forward constant-velocity Kalman filter followed by an RTS backward smooth
    /// of a single scalar's time series, in place. ``dt`` is one frame.
    void smooth_scalar_series(std::vector<float>& series, const KalmanParams& params, std::vector<ScalarRecord>& scratch) {
        const std::size_t length = series.size();
        if (length == 0) {
            return;
        }
        scratch.resize(length);

        // Continuous white-noise-acceleration process covariance for dt = 1:
        // Q = q * [[1/3, 1/2], [1/2, 1]].
        const float process = params.process_noise;
        const float q_a = process / 3.0f;
        const float q_b = process / 2.0f;
        const float q_d = process;
        const float measurement = params.measurement_noise;

        // Seed at the first measurement: known position, unknown velocity.
        ScalarRecord& first = scratch[0];
        first.filt_pos = series[0];
        first.filt_vel = 0.0f;
        first.filt_cov = Mat2{measurement, 0.0f, 0.0f, 1.0f};
        first.pred_pos = first.filt_pos;
        first.pred_vel = first.filt_vel;
        first.pred_cov = first.filt_cov;

        for (std::size_t frame = 1; frame < length; ++frame) {
            const ScalarRecord& prev = scratch[frame - 1];
            ScalarRecord& current = scratch[frame];

            // Predict: x = F x, with F = [[1, 1], [0, 1]] (dt = 1).
            current.pred_pos = prev.filt_pos + prev.filt_vel;
            current.pred_vel = prev.filt_vel;

            // P = F P F^T + Q.
            const Mat2& prior = prev.filt_cov;
            Mat2 predicted;
            predicted.a = prior.a + prior.b + prior.c + prior.d + q_a;
            predicted.b = prior.b + prior.d + q_b;
            predicted.c = prior.c + prior.d + q_b;
            predicted.d = prior.d + q_d;
            current.pred_cov = predicted;

            // Update with the measurement, H = [1, 0].
            const float innovation = series[frame] - current.pred_pos;
            const float innovation_cov = predicted.a + measurement;
            const float gain_pos = predicted.a / innovation_cov;
            const float gain_vel = predicted.c / innovation_cov;

            current.filt_pos = current.pred_pos + gain_pos * innovation;
            current.filt_vel = current.pred_vel + gain_vel * innovation;

            // P = (I - K H) P.
            current.filt_cov.a = predicted.a - gain_pos * predicted.a;
            current.filt_cov.b = predicted.b - gain_pos * predicted.b;
            current.filt_cov.c = predicted.c - gain_vel * predicted.a;
            current.filt_cov.d = predicted.d - gain_vel * predicted.b;
        }

        // Backward RTS smoothing. The last frame's smoothed estimate is its
        // filtered estimate; earlier frames are corrected using future ones.
        float smooth_pos = scratch[length - 1].filt_pos;
        float smooth_vel = scratch[length - 1].filt_vel;
        series[length - 1] = smooth_pos;

        for (std::size_t frame = length - 1; frame-- > 0;) {
            const ScalarRecord& record = scratch[frame];
            const ScalarRecord& next = scratch[frame + 1];

            // Smoother gain C = P_filt F^T P_pred(next)^-1, with F^T = [[1,0],[1,1]].
            // M = P_filt F^T.
            const Mat2& filt = record.filt_cov;
            const float m_a = filt.a + filt.b;
            const float m_b = filt.b;
            const float m_c = filt.c + filt.d;
            const float m_d = filt.d;

            // Inverse of the predicted covariance at the next frame.
            const Mat2& pred = next.pred_cov;
            const float determinant = pred.a * pred.d - pred.b * pred.c;
            if (determinant == 0.0f) {
                series[frame] = record.filt_pos;
                smooth_pos = record.filt_pos;
                smooth_vel = record.filt_vel;
                continue;
            }
            const float inv_scale = 1.0f / determinant;
            const float inv_a = pred.d * inv_scale;
            const float inv_b = -pred.b * inv_scale;
            const float inv_c = -pred.c * inv_scale;
            const float inv_d = pred.a * inv_scale;

            // C = M * P_pred^-1 (full 2x2; the velocity column feeds the next
            // iteration's smoothed velocity, so all four entries are needed).
            const float gain_a = m_a * inv_a + m_b * inv_c;
            const float gain_b = m_a * inv_b + m_b * inv_d;
            const float gain_c = m_c * inv_a + m_d * inv_c;
            const float gain_d = m_c * inv_b + m_d * inv_d;

            // x_smooth = x_filt + C (x_smooth(next) - x_pred(next)).
            const float residual_pos = smooth_pos - next.pred_pos;
            const float residual_vel = smooth_vel - next.pred_vel;

            const float new_pos = record.filt_pos + gain_a * residual_pos + gain_b * residual_vel;
            const float new_vel = record.filt_vel + gain_c * residual_pos + gain_d * residual_vel;

            series[frame] = new_pos;
            smooth_pos = new_pos;
            smooth_vel = new_vel;
        }
    }

    /// True when two hands at the same slot in adjacent frames belong to the same
    /// continuous track (same handedness and vertex/joint counts).
    bool same_track(const HandData& left, const HandData& right) {
        return left.is_right == right.is_right && left.verts.size() == right.verts.size() && left.joints.size() == right.joints.size();
    }
} // namespace

std::vector<Frame> smooth_sequence(const std::vector<Frame>& frames, const KalmanParams& params) {
    std::vector<Frame> result = frames;
    if (frames.empty()) {
        return result;
    }

    std::size_t max_slots = 0;
    for (const Frame& frame : frames) {
        max_slots = std::max(max_slots, frame.size());
    }

    std::vector<ScalarRecord> scratch;
    std::vector<float> series;

    for (std::size_t slot = 0; slot < max_slots; ++slot) {
        // Walk this slot across frames, smoothing each maximal run where the same
        // hand is present (a "track segment").
        std::size_t frame = 0;
        while (frame < result.size()) {
            if (slot >= result[frame].size()) {
                ++frame;
                continue;
            }
            const std::size_t segment_start = frame;
            std::size_t segment_end = frame + 1;
            while (segment_end < result.size() && slot < result[segment_end].size() && same_track(result[segment_start][slot], result[segment_end][slot])) {
                ++segment_end;
            }

            // Flatten each hand in the segment once, smooth every coordinate as an
            // independent series, then write the smoothed coordinates back.
            const std::size_t segment_length = segment_end - segment_start;
            std::vector<std::vector<float>> flattened(segment_length);
            for (std::size_t offset = 0; offset < segment_length; ++offset) {
                flattened[offset] = flatten_hand(result[segment_start + offset][slot]);
            }

            const std::size_t scalar_count = flattened.front().size();
            series.resize(segment_length);
            for (std::size_t scalar = 0; scalar < scalar_count; ++scalar) {
                for (std::size_t offset = 0; offset < segment_length; ++offset) {
                    series[offset] = flattened[offset][scalar];
                }
                smooth_scalar_series(series, params, scratch);
                for (std::size_t offset = 0; offset < segment_length; ++offset) {
                    flattened[offset][scalar] = series[offset];
                }
            }

            for (std::size_t offset = 0; offset < segment_length; ++offset) {
                apply_flat_to_hand(result[segment_start + offset][slot], flattened[offset]);
            }

            frame = segment_end;
        }
    }

    return result;
}
