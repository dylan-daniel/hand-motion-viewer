#include "data/object_fit.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>

#include <glm/gtc/quaternion.hpp>

namespace {
    // ── Small linear algebra helpers ──────────────────────────────────────

    /// Direction of minimum variance of a covariance matrix: power iteration
    /// on (trace(C)*I - C), whose dominant eigenvector is C's smallest-
    /// eigenvalue eigenvector (trace(C) is a safe upper bound on C's largest
    /// eigenvalue since C is positive semi-definite). Used as the best-fit
    /// plane normal in fit_plane_ransac's PCA refinement.
    glm::vec3 smallest_eigenvector(const glm::mat3& covariance) {
        const float trace = covariance[0][0] + covariance[1][1] + covariance[2][2];
        const glm::mat3 shifted = glm::mat3(trace) - covariance;
        glm::vec3 vector(1.0f, 1.0f, 1.0f);
        for (int iteration = 0; iteration < 50; ++iteration) {
            const glm::vec3 next = shifted * vector;
            const float length = glm::length(next);
            if (length < 1e-12f) {
                break;
            }
            vector = next / length;
        }
        return vector;
    }

    /// Solves the 7x7 system Ax = b via Gaussian elimination with partial
    /// pivoting (A, b passed by value — this function mutates its own copy).
    /// Returns false if A is singular to working precision.
    bool solve7(std::array<std::array<float, 7>, 7> a, std::array<float, 7> b, std::array<float, 7>& x) {
        for (int col = 0; col < 7; ++col) {
            int pivot = col;
            float best = std::abs(a[col][col]);
            for (int row = col + 1; row < 7; ++row) {
                if (std::abs(a[row][col]) > best) {
                    best = std::abs(a[row][col]);
                    pivot = row;
                }
            }
            if (best < 1e-12f) {
                return false;
            }
            if (pivot != col) {
                std::swap(a[col], a[pivot]);
                std::swap(b[col], b[pivot]);
            }
            for (int row = col + 1; row < 7; ++row) {
                const float factor = a[row][col] / a[col][col];
                for (int c = col; c < 7; ++c) {
                    a[row][c] -= factor * a[col][c];
                }
                b[row] -= factor * b[col];
            }
        }
        for (int row = 6; row >= 0; --row) {
            float sum = b[row];
            for (int c = row + 1; c < 7; ++c) {
                sum -= a[row][c] * x[c];
            }
            x[row] = sum / a[row][row];
        }
        return true;
    }

    // ── 2D convex hull + rotating-calipers min-area rect ────────────────────

    std::vector<glm::vec2> convex_hull(std::vector<glm::vec2> points) {
        std::sort(points.begin(), points.end(), [](const glm::vec2& a, const glm::vec2& b) { return a.x != b.x ? a.x < b.x : a.y < b.y; });
        points.erase(std::unique(points.begin(), points.end(), [](const glm::vec2& a, const glm::vec2& b) { return a.x == b.x && a.y == b.y; }), points.end());
        const int n = static_cast<int>(points.size());
        if (n < 3) {
            return points;
        }
        auto cross = [](const glm::vec2& o, const glm::vec2& a, const glm::vec2& b) { return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x); };
        std::vector<glm::vec2> hull(2 * static_cast<std::size_t>(n));
        int k = 0;
        for (int i = 0; i < n; ++i) {
            while (k >= 2 && cross(hull[static_cast<std::size_t>(k - 2)], hull[static_cast<std::size_t>(k - 1)], points[static_cast<std::size_t>(i)]) <= 0) {
                --k;
            }
            hull[static_cast<std::size_t>(k++)] = points[static_cast<std::size_t>(i)];
        }
        const int lower = k + 1;
        for (int i = n - 2; i >= 0; --i) {
            while (k >= lower && cross(hull[static_cast<std::size_t>(k - 2)], hull[static_cast<std::size_t>(k - 1)], points[static_cast<std::size_t>(i)]) <= 0) {
                --k;
            }
            hull[static_cast<std::size_t>(k++)] = points[static_cast<std::size_t>(i)];
        }
        hull.resize(static_cast<std::size_t>(k - 1));
        return hull;
    }

    struct MinAreaRect {
        glm::vec2 center{0.0f};
        glm::vec2 axis_a{1.0f, 0.0f}; // unit direction of the ``side_a`` edge
        float side_a = 0.0f;
        float side_b = 0.0f;
    };

    std::optional<MinAreaRect> min_area_rect(const std::vector<glm::vec2>& points) {
        const std::vector<glm::vec2> hull = convex_hull(points);
        const int n = static_cast<int>(hull.size());
        if (n < 3) {
            return std::nullopt;
        }
        MinAreaRect best;
        float best_area = std::numeric_limits<float>::max();
        bool found = false;
        for (int i = 0; i < n; ++i) {
            const glm::vec2 edge = hull[static_cast<std::size_t>((i + 1) % n)] - hull[static_cast<std::size_t>(i)];
            const float length = glm::length(edge);
            if (length < 1e-9f) {
                continue;
            }
            const glm::vec2 dir = edge / length;
            const glm::vec2 perp(-dir.y, dir.x);
            float min_d = std::numeric_limits<float>::max(), max_d = std::numeric_limits<float>::lowest();
            float min_p = std::numeric_limits<float>::max(), max_p = std::numeric_limits<float>::lowest();
            for (const glm::vec2& q : hull) {
                const float d = glm::dot(q, dir);
                const float p = glm::dot(q, perp);
                min_d = std::min(min_d, d);
                max_d = std::max(max_d, d);
                min_p = std::min(min_p, p);
                max_p = std::max(max_p, p);
            }
            const float side_a = max_d - min_d;
            const float side_b = max_p - min_p;
            const float area = side_a * side_b;
            if (area < best_area) {
                best_area = area;
                found = true;
                best.center = ((min_d + max_d) * 0.5f) * dir + ((min_p + max_p) * 0.5f) * perp;
                best.axis_a = dir;
                best.side_a = side_a;
                best.side_b = side_b;
            }
        }
        if (!found) {
            return std::nullopt;
        }
        return best;
    }

    // ── Plane RANSAC ─────────────────────────────────────────────────────

    struct PlaneFit {
        glm::vec3 normal{0.0f, 0.0f, 1.0f};
        glm::vec3 point{0.0f};
        std::vector<char> inliers; // parallel to the input points
    };

    std::optional<PlaneFit> fit_plane_ransac(
        const std::vector<glm::vec3>& points, float inlier_dist, int iterations, const glm::vec3* view_dir, float min_facing
    ) {
        const int n = static_cast<int>(points.size());
        if (n < 30) {
            return std::nullopt;
        }
        std::mt19937 rng(0);
        std::uniform_int_distribution<int> pick(0, n - 1);
        int best_count = 0;
        std::vector<char> best_inliers;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const int ia = pick(rng), ib = pick(rng), ic = pick(rng);
            if (ia == ib || ib == ic || ia == ic) {
                continue;
            }
            const glm::vec3 a = points[static_cast<std::size_t>(ia)], b = points[static_cast<std::size_t>(ib)], c = points[static_cast<std::size_t>(ic)];
            glm::vec3 normal = glm::cross(b - a, c - a);
            const float norm = glm::length(normal);
            if (norm < 1e-12f) {
                continue;
            }
            normal /= norm;
            if (view_dir != nullptr && std::abs(glm::dot(normal, *view_dir)) < min_facing) {
                continue;
            }
            std::vector<char> inliers(static_cast<std::size_t>(n), 0);
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (std::abs(glm::dot(points[static_cast<std::size_t>(i)] - a, normal)) < inlier_dist) {
                    inliers[static_cast<std::size_t>(i)] = 1;
                    ++count;
                }
            }
            if (count > best_count) {
                best_count = count;
                best_inliers = std::move(inliers);
            }
        }
        if (best_count < 30) {
            return std::nullopt;
        }
        glm::vec3 normal(0.0f), centroid(0.0f);
        for (int refine = 0; refine < 2; ++refine) {
            std::vector<glm::vec3> selected;
            for (int i = 0; i < n; ++i) {
                if (best_inliers[static_cast<std::size_t>(i)] != 0) {
                    selected.push_back(points[static_cast<std::size_t>(i)]);
                }
            }
            if (selected.size() < 30) {
                return std::nullopt;
            }
            centroid = glm::vec3(0.0f);
            for (const glm::vec3& p : selected) {
                centroid += p;
            }
            centroid /= static_cast<float>(selected.size());
            glm::mat3 covariance(0.0f);
            for (const glm::vec3& p : selected) {
                const glm::vec3 d = p - centroid;
                covariance += glm::outerProduct(d, d);
            }
            covariance /= static_cast<float>(selected.size());
            normal = smallest_eigenvector(covariance);
            if (glm::length(normal) < 1e-9f) {
                return std::nullopt;
            }
            normal = glm::normalize(normal);
            std::vector<char> inliers(static_cast<std::size_t>(n), 0);
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (std::abs(glm::dot(points[static_cast<std::size_t>(i)] - centroid, normal)) < inlier_dist) {
                    inliers[static_cast<std::size_t>(i)] = 1;
                    ++count;
                }
            }
            if (count < 30) {
                return std::nullopt;
            }
            best_inliers = std::move(inliers);
        }
        if (view_dir != nullptr && std::abs(glm::dot(normal, *view_dir)) < min_facing) {
            return std::nullopt;
        }
        return PlaneFit{normal, centroid, best_inliers};
    }

    // ── Cube plane+silhouette construction ──────────────────────────────

    struct CubeConstruction {
        glm::mat3 rotation{1.0f};
        glm::vec3 center{0.0f};
        float scale = 1.0f;
        float lo = 0.0f;
        float hi = 0.0f;
    };

    std::optional<CubeConstruction> cube_construction(const std::vector<glm::vec3>& points, float edge, const float* scale_hint) {
        if (points.size() < 30) {
            return std::nullopt;
        }
        glm::vec3 centroid(0.0f);
        for (const glm::vec3& p : points) {
            centroid += p;
        }
        centroid /= static_cast<float>(points.size());
        const glm::vec3 centroid_view = glm::normalize(centroid);
        const std::optional<PlaneFit> plane = fit_plane_ransac(points, 0.006f, 120, &centroid_view, 0.35f);
        if (!plane) {
            return std::nullopt;
        }
        glm::vec3 normal = plane->normal;
        const glm::vec3 plane_point = plane->point;
        const glm::vec3 view_dir = glm::normalize(plane_point);
        if (glm::dot(normal, view_dir) > 0.0f) {
            normal = -normal;
        }
        glm::vec3 basis_u = glm::cross(normal, view_dir);
        if (glm::length(basis_u) < 1e-6f) {
            basis_u = glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f));
        }
        basis_u = glm::normalize(basis_u);
        const glm::vec3 basis_v = glm::cross(normal, basis_u);

        std::vector<glm::vec2> uv;
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (plane->inliers[i] == 0) {
                continue;
            }
            const glm::vec3 d = points[i] - plane_point;
            uv.emplace_back(glm::dot(d, basis_u), glm::dot(d, basis_v));
        }
        const std::optional<MinAreaRect> rect = min_area_rect(uv);
        if (!rect) {
            return std::nullopt;
        }
        const glm::vec3 axis1 = rect->axis_a.x * basis_u + rect->axis_a.y * basis_v;
        const glm::vec3 axis2 = glm::cross(normal, axis1);
        const glm::mat3 rotation0(axis1, axis2, normal);
        const float lo = std::min(rect->side_a, rect->side_b);
        const float hi = std::max(rect->side_a, rect->side_b);
        const float scale0 = (lo > 1e-6f && hi / lo < 1.4f) ? std::sqrt(rect->side_a * rect->side_b) / edge : (scale_hint != nullptr ? *scale_hint : 1.0f);
        const glm::vec3 face_center = plane_point + rect->center.x * basis_u + rect->center.y * basis_v;
        const glm::vec3 center0 = face_center - (scale0 * edge * 0.5f) * normal;
        return CubeConstruction{rotation0, center0, scale0, lo, hi};
    }

    glm::mat3 rodrigues(const glm::vec3& rvec) {
        const float theta = glm::length(rvec);
        if (theta < 1e-12f) {
            return glm::mat3(1.0f);
        }
        const glm::vec3 axis = rvec / theta;
        glm::mat3 skew(0.0f);
        skew[1][0] = -axis.z;
        skew[2][0] = axis.y;
        skew[0][1] = axis.z;
        skew[2][1] = -axis.x;
        skew[0][2] = -axis.y;
        skew[1][2] = axis.x;
        return glm::mat3(1.0f) + std::sin(theta) * skew + (1.0f - std::cos(theta)) * (skew * skew);
    }

    float cube_sdf_point(const glm::vec3& point, const glm::mat3& rotation, const glm::vec3& center, float edge) {
        const glm::vec3 local = glm::transpose(rotation) * (point - center);
        const glm::vec3 outward = glm::abs(local) - glm::vec3(edge * 0.5f);
        const float outside = glm::length(glm::max(outward, glm::vec3(0.0f)));
        const float inside = std::min(std::max(outward.x, std::max(outward.y, outward.z)), 0.0f);
        return outside + inside;
    }

    /// The 24 rotational symmetries of a cube: signed permutation matrices
    /// with determinant +1.
    const std::vector<glm::mat3>& cube_symmetries() {
        static const std::vector<glm::mat3> symmetries = [] {
            std::vector<glm::mat3> result;
            constexpr int permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
            for (const auto& perm : permutations) {
                for (int sx = -1; sx <= 1; sx += 2) {
                    for (int sy = -1; sy <= 1; sy += 2) {
                        for (int sz = -1; sz <= 1; sz += 2) {
                            glm::mat3 m(0.0f);
                            const float signs[3] = {static_cast<float>(sx), static_cast<float>(sy), static_cast<float>(sz)};
                            for (int row = 0; row < 3; ++row) {
                                m[perm[row]][row] = signs[row];
                            }
                            if (glm::determinant(m) > 0.5f) {
                                result.push_back(m);
                            }
                        }
                    }
                }
            }
            return result;
        }();
        return symmetries;
    }
} // namespace

std::vector<glm::vec3> backproject_mask(
    const std::vector<float>& depth, int depth_height, int depth_width, const std::vector<std::uint8_t>& mask, int mask_height, int mask_width,
    const CameraIntrinsics& intrinsics
) {
    std::vector<glm::vec3> points;
    const int height = std::min(depth_height, mask_height);
    const int width = std::min(depth_width, mask_width);
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            if (mask[static_cast<std::size_t>(v) * static_cast<std::size_t>(mask_width) + static_cast<std::size_t>(u)] == 0) {
                continue;
            }
            const float z = depth[static_cast<std::size_t>(v) * static_cast<std::size_t>(depth_width) + static_cast<std::size_t>(u)];
            if (z <= 0.0f) {
                continue;
            }
            const float x = (static_cast<float>(u) - intrinsics.cx) * z / intrinsics.fx;
            const float y = (static_cast<float>(v) - intrinsics.cy) * z / intrinsics.fy;
            points.emplace_back(x, y, z);
        }
    }
    return points;
}

std::vector<glm::vec3> filter_depth_outliers(const std::vector<glm::vec3>& points, float margin) {
    if (points.empty()) {
        return points;
    }
    std::vector<float> depths;
    depths.reserve(points.size());
    for (const glm::vec3& p : points) {
        depths.push_back(p.z);
    }
    const std::size_t mid = depths.size() / 2;
    std::nth_element(depths.begin(), depths.begin() + static_cast<std::ptrdiff_t>(mid), depths.end());
    const float median = depths[mid];
    std::vector<glm::vec3> filtered;
    filtered.reserve(points.size());
    for (const glm::vec3& p : points) {
        if (std::abs(p.z - median) <= margin) {
            filtered.push_back(p);
        }
    }
    return filtered;
}

std::optional<glm::vec3> fit_sphere_center(const std::vector<glm::vec3>& points_in, float radius) {
    constexpr int iterations = 8;
    constexpr std::size_t min_points = 20;
    if (points_in.size() < min_points) {
        return std::nullopt;
    }
    const float margin = std::max(0.08f, 3.0f * radius);
    const std::vector<glm::vec3> points = filter_depth_outliers(points_in, margin);
    if (points.size() < min_points) {
        return std::nullopt;
    }

    glm::vec3 centroid(0.0f);
    for (const glm::vec3& p : points) {
        centroid += p;
    }
    centroid /= static_cast<float>(points.size());
    const glm::vec3 view_dir = glm::normalize(centroid);
    glm::vec3 center = centroid + radius * view_dir;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        glm::mat3 JtJ(0.0f);
        glm::vec3 Jtr(0.0f);
        for (const glm::vec3& p : points) {
            const glm::vec3 offset = center - p;
            const float dist = std::max(glm::length(offset), 1e-9f);
            const float residual = dist - radius;
            const glm::vec3 jacobian = offset / dist;
            JtJ += glm::outerProduct(jacobian, jacobian);
            Jtr += jacobian * residual;
        }
        if (std::abs(glm::determinant(JtJ)) < 1e-12f) {
            break;
        }
        const glm::vec3 delta = -(glm::inverse(JtJ) * Jtr);
        center += delta;
        if (glm::length(delta) < 1e-6f) {
            break;
        }
    }
    return center;
}

std::optional<float> estimate_cube_scale(const std::vector<glm::vec3>& points, float edge) {
    constexpr std::size_t min_points = 60;
    if (points.size() < min_points) {
        return std::nullopt;
    }
    const std::optional<CubeConstruction> construction = cube_construction(points, edge, nullptr);
    if (!construction) {
        return std::nullopt;
    }
    if (construction->lo < 1e-6f || construction->hi / construction->lo >= 1.4f) {
        return std::nullopt;
    }
    return std::sqrt(construction->lo * construction->hi) / edge;
}

std::optional<CubeFit> fit_cube_pose(const std::vector<glm::vec3>& points_in, float edge, const CubeFit* init, float scale_prior) {
    constexpr int refine_iterations = 25;
    constexpr std::size_t max_points = 2500;
    constexpr std::size_t min_points = 60;
    constexpr float scale_prior_weight = 0.5f;
    constexpr float eps = 1e-5f;
    if (points_in.size() < min_points) {
        return std::nullopt;
    }

    std::vector<glm::vec3> points = points_in;
    if (points.size() > max_points) {
        std::mt19937 rng(0);
        std::shuffle(points.begin(), points.end(), rng);
        points.resize(max_points);
    }

    glm::mat3 rotation;
    glm::vec3 center;
    float scale;
    const std::optional<CubeConstruction> construction = cube_construction(points, edge, &scale_prior);
    if (construction) {
        rotation = construction->rotation;
        center = construction->center;
        scale = construction->scale;
    } else if (init != nullptr) {
        rotation = init->rotation;
        center = init->center;
        scale = init->scale;
    } else {
        return std::nullopt;
    }
    scale = std::clamp(scale, 0.4f, 1.6f);

    const std::size_t n = points.size();
    const float prior_weight = scale_prior_weight * std::sqrt(static_cast<float>(n)) * edge;

    auto residuals_for = [&](const glm::mat3& r, const glm::vec3& c, float s) {
        std::vector<float> out(n + 1);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = cube_sdf_point(points[i], r, c, edge * s);
        }
        out[n] = prior_weight * (s - scale_prior);
        return out;
    };
    auto sum_sq = [](const std::vector<float>& v) {
        double s = 0.0;
        for (float x : v) {
            s += static_cast<double>(x) * static_cast<double>(x);
        }
        return s;
    };

    std::vector<float> residuals = residuals_for(rotation, center, scale);
    double cost = sum_sq(residuals);
    float damping = 1e-3f;

    for (int iteration = 0; iteration < refine_iterations; ++iteration) {
        std::array<std::vector<float>, 7> jacobian_col;
        for (int p = 0; p < 3; ++p) {
            glm::vec3 rvec(0.0f);
            rvec[p] = eps;
            const std::vector<float> perturbed = residuals_for(rodrigues(rvec) * rotation, center, scale);
            jacobian_col[static_cast<std::size_t>(p)].resize(n + 1);
            for (std::size_t k = 0; k < n + 1; ++k) {
                jacobian_col[static_cast<std::size_t>(p)][k] = (perturbed[k] - residuals[k]) / eps;
            }
        }
        for (int p = 0; p < 3; ++p) {
            glm::vec3 offset(0.0f);
            offset[p] = eps;
            const std::vector<float> perturbed = residuals_for(rotation, center + offset, scale);
            jacobian_col[static_cast<std::size_t>(3 + p)].resize(n + 1);
            for (std::size_t k = 0; k < n + 1; ++k) {
                jacobian_col[static_cast<std::size_t>(3 + p)][k] = (perturbed[k] - residuals[k]) / eps;
            }
        }
        {
            const std::vector<float> perturbed = residuals_for(rotation, center, scale + eps);
            jacobian_col[6].resize(n + 1);
            for (std::size_t k = 0; k < n + 1; ++k) {
                jacobian_col[6][k] = (perturbed[k] - residuals[k]) / eps;
            }
        }

        std::array<std::array<float, 7>, 7> JtJ{};
        std::array<float, 7> Jtr{};
        for (int a = 0; a < 7; ++a) {
            for (int b = 0; b < 7; ++b) {
                double s = 0.0;
                for (std::size_t k = 0; k < n + 1; ++k) {
                    s += static_cast<double>(jacobian_col[static_cast<std::size_t>(a)][k]) * static_cast<double>(jacobian_col[static_cast<std::size_t>(b)][k]);
                }
                JtJ[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] = static_cast<float>(s);
            }
            double s = 0.0;
            for (std::size_t k = 0; k < n + 1; ++k) {
                s += static_cast<double>(jacobian_col[static_cast<std::size_t>(a)][k]) * static_cast<double>(residuals[k]);
            }
            Jtr[static_cast<std::size_t>(a)] = static_cast<float>(s);
        }

        std::array<std::array<float, 7>, 7> a = JtJ;
        for (int i = 0; i < 7; ++i) {
            a[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] += damping * JtJ[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] + 1e-12f;
        }
        std::array<float, 7> b{};
        for (int i = 0; i < 7; ++i) {
            b[static_cast<std::size_t>(i)] = -Jtr[static_cast<std::size_t>(i)];
        }
        std::array<float, 7> delta{};
        if (!solve7(a, b, delta)) {
            break;
        }

        const glm::mat3 new_rotation = rodrigues(glm::vec3(delta[0], delta[1], delta[2])) * rotation;
        const glm::vec3 new_center = center + glm::vec3(delta[3], delta[4], delta[5]);
        const float new_scale = std::clamp(scale + delta[6], 0.4f, 1.6f);
        const std::vector<float> new_residuals = residuals_for(new_rotation, new_center, new_scale);
        const double new_cost = sum_sq(new_residuals);
        float delta_norm = 0.0f;
        for (float d : delta) {
            delta_norm += d * d;
        }
        delta_norm = std::sqrt(delta_norm);

        if (new_cost < cost) {
            rotation = new_rotation;
            center = new_center;
            scale = new_scale;
            residuals = new_residuals;
            cost = new_cost;
            damping = std::max(damping * 0.5f, 1e-6f);
            if (delta_norm < 1e-7f) {
                break;
            }
        } else {
            damping *= 4.0f;
            if (damping > 1e3f) {
                break;
            }
        }
    }

    const float rms = std::sqrt(static_cast<float>(cost / static_cast<double>(n)));
    return CubeFit{rotation, center, scale, rms};
}

std::vector<glm::mat3> canonicalize_cube_rotations(const std::vector<glm::mat3>& rotations) {
    std::vector<glm::mat3> out;
    if (rotations.empty()) {
        return out;
    }
    out.reserve(rotations.size());
    out.push_back(rotations[0]);
    const std::vector<glm::mat3>& symmetries = cube_symmetries();
    for (std::size_t i = 1; i < rotations.size(); ++i) {
        const glm::mat3& prev = out.back();
        const glm::mat3& rotation = rotations[i];
        glm::mat3 best(1.0f);
        float best_trace = -std::numeric_limits<float>::max();
        for (const glm::mat3& sym : symmetries) {
            const glm::mat3 candidate = rotation * sym;
            const glm::mat3 product = glm::transpose(prev) * candidate;
            const float trace = product[0][0] + product[1][1] + product[2][2];
            if (trace > best_trace) {
                best_trace = trace;
                best = candidate;
            }
        }
        out.push_back(best);
    }
    return out;
}

std::vector<glm::mat3> smooth_rotations(const std::vector<int>& frame_numbers, const std::vector<glm::mat3>& rotations, float sigma_frames) {
    if (sigma_frames <= 0.0f || rotations.empty()) {
        return rotations;
    }
    std::vector<glm::quat> quats(rotations.size());
    for (std::size_t i = 0; i < rotations.size(); ++i) {
        quats[i] = glm::quat_cast(rotations[i]);
    }
    for (std::size_t k = 1; k < quats.size(); ++k) {
        if (glm::dot(quats[k], quats[k - 1]) < 0.0f) {
            quats[k] = -quats[k];
        }
    }
    const float cutoff = 3.0f * sigma_frames;
    std::vector<glm::mat3> out(rotations.size());
    for (std::size_t k = 0; k < rotations.size(); ++k) {
        glm::quat blended(0.0f, 0.0f, 0.0f, 0.0f);
        for (std::size_t j = 0; j < rotations.size(); ++j) {
            const float dt = static_cast<float>(frame_numbers[j] - frame_numbers[k]);
            if (std::abs(dt) > cutoff) {
                continue;
            }
            const float weight = std::exp(-0.5f * (dt / sigma_frames) * (dt / sigma_frames));
            blended = glm::quat(blended.w + weight * quats[j].w, blended.x + weight * quats[j].x, blended.y + weight * quats[j].y, blended.z + weight * quats[j].z);
        }
        out[k] = glm::mat3_cast(glm::normalize(blended));
    }
    return out;
}

std::vector<float> smooth_sequence(const std::vector<int>& frame_numbers, const std::vector<float>& values, float sigma_frames) {
    if (sigma_frames <= 0.0f) {
        return values;
    }
    const float cutoff = 3.0f * sigma_frames;
    std::vector<float> out(values.size());
    for (std::size_t k = 0; k < values.size(); ++k) {
        double weight_sum = 0.0, value_sum = 0.0;
        for (std::size_t j = 0; j < values.size(); ++j) {
            const float dt = static_cast<float>(frame_numbers[j] - frame_numbers[k]);
            if (std::abs(dt) > cutoff) {
                continue;
            }
            const double weight = std::exp(-0.5 * static_cast<double>(dt / sigma_frames) * static_cast<double>(dt / sigma_frames));
            weight_sum += weight;
            value_sum += weight * static_cast<double>(values[j]);
        }
        out[k] = weight_sum > 0.0 ? static_cast<float>(value_sum / weight_sum) : values[k];
    }
    return out;
}

int select_hand_mask_instance(const std::vector<std::uint8_t>& masks, int count, int height, int width, const glm::vec2& wrist_uv, float max_snap_px) {
    if (count <= 0 || height <= 0 || width <= 0) {
        return -1;
    }
    const std::size_t plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    const int ui = static_cast<int>(std::lround(wrist_uv.x));
    const int vi = static_cast<int>(std::lround(wrist_uv.y));
    if (vi >= 0 && vi < height && ui >= 0 && ui < width) {
        for (int instance = 0; instance < count; ++instance) {
            const std::size_t offset = static_cast<std::size_t>(instance) * plane + static_cast<std::size_t>(vi) * static_cast<std::size_t>(width) +
                                        static_cast<std::size_t>(ui);
            if (masks[offset] != 0) {
                return instance;
            }
        }
    }
    int best_instance = -1;
    float best_dist = max_snap_px;
    for (int instance = 0; instance < count; ++instance) {
        const std::uint8_t* plane_ptr = masks.data() + static_cast<std::size_t>(instance) * plane;
        int min_u = width, max_u = -1, min_v = height, max_v = -1;
        for (int v = 0; v < height; ++v) {
            const std::uint8_t* row = plane_ptr + static_cast<std::size_t>(v) * static_cast<std::size_t>(width);
            for (int u = 0; u < width; ++u) {
                if (row[u] != 0) {
                    min_u = std::min(min_u, u);
                    max_u = std::max(max_u, u);
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                }
            }
        }
        if (max_u < 0) {
            continue; // empty instance
        }
        const float center_u = static_cast<float>(min_u + max_u) * 0.5f;
        const float center_v = static_cast<float>(min_v + max_v) * 0.5f;
        const float dist = std::hypot(wrist_uv.x - center_u, wrist_uv.y - center_v);
        if (dist < best_dist) {
            best_dist = dist;
            best_instance = instance;
        }
    }
    return best_instance;
}

std::optional<float> sample_wrist_depth_da3(
    const std::vector<float>& depth, int depth_height, int depth_width, const glm::vec2& wrist_uv, const std::uint8_t* hand_mask, int radius_px,
    int min_valid_px
) {
    if (depth.empty() || depth_height <= 0 || depth_width <= 0) {
        return std::nullopt;
    }
    const int ui = static_cast<int>(std::lround(wrist_uv.x));
    const int vi = static_cast<int>(std::lround(wrist_uv.y));
    const int v0 = std::max(0, vi - radius_px), v1 = std::min(depth_height, vi + radius_px + 1);
    const int u0 = std::max(0, ui - radius_px), u1 = std::min(depth_width, ui + radius_px + 1);
    if (v0 >= v1 || u0 >= u1) {
        return std::nullopt;
    }
    const std::int64_t radius_sq = static_cast<std::int64_t>(radius_px) * static_cast<std::int64_t>(radius_px);

    auto collect = [&](bool use_mask) {
        std::vector<float> samples;
        for (int v = v0; v < v1; ++v) {
            const std::int64_t dv = v - vi;
            const std::size_t row_base = static_cast<std::size_t>(v) * static_cast<std::size_t>(depth_width);
            for (int u = u0; u < u1; ++u) {
                const std::int64_t du = u - ui;
                if (dv * dv + du * du > radius_sq) {
                    continue;
                }
                const float d = depth[row_base + static_cast<std::size_t>(u)];
                if (d <= 0.0f) {
                    continue;
                }
                if (use_mask && hand_mask[row_base + static_cast<std::size_t>(u)] == 0) {
                    continue;
                }
                samples.push_back(d);
            }
        }
        return samples;
    };

    std::vector<float> samples;
    if (hand_mask != nullptr) {
        samples = collect(true);
        if (static_cast<int>(samples.size()) < min_valid_px) {
            samples = collect(false); // fall back to the unmasked disk
        }
    } else {
        samples = collect(false);
    }
    if (static_cast<int>(samples.size()) < min_valid_px) {
        return std::nullopt;
    }
    const std::size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(mid), samples.end());
    return samples[mid];
}

std::vector<std::vector<int>> track_hand_sequences(
    const std::vector<std::vector<glm::vec2>>& wrist_uv_by_frame, const std::vector<std::vector<bool>>& is_right_by_frame, float max_track_dist_px
) {
    int next_id = 0;
    // active[0] = left-hand tracks, active[1] = right-hand tracks; each a
    // list of (track_id, last-seen wrist_uv).
    std::array<std::vector<std::pair<int, glm::vec2>>, 2> active;
    std::vector<std::vector<int>> track_ids_by_frame(wrist_uv_by_frame.size());

    for (std::size_t frame = 0; frame < wrist_uv_by_frame.size(); ++frame) {
        const std::size_t count = wrist_uv_by_frame[frame].size();
        track_ids_by_frame[frame].assign(count, -1);
        for (int side = 0; side < 2; ++side) {
            const bool is_right_side = side == 1;
            std::vector<std::pair<int, glm::vec2>>& prev_tracks = active[static_cast<std::size_t>(side)];
            std::vector<char> used(prev_tracks.size(), 0);
            std::vector<std::pair<int, glm::vec2>> new_tracks;
            for (std::size_t i = 0; i < count; ++i) {
                if (is_right_by_frame[frame][i] != is_right_side) {
                    continue;
                }
                const glm::vec2 uv = wrist_uv_by_frame[frame][i];
                int best_j = -1;
                float best_dist = max_track_dist_px;
                for (std::size_t j = 0; j < prev_tracks.size(); ++j) {
                    if (used[j] != 0) {
                        continue;
                    }
                    const float dist = glm::length(uv - prev_tracks[j].second);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_j = static_cast<int>(j);
                    }
                }
                int track_id = 0;
                if (best_j >= 0) {
                    track_id = prev_tracks[static_cast<std::size_t>(best_j)].first;
                    used[static_cast<std::size_t>(best_j)] = 1;
                } else {
                    track_id = next_id++;
                }
                track_ids_by_frame[frame][i] = track_id;
                new_tracks.emplace_back(track_id, uv);
            }
            active[static_cast<std::size_t>(side)] = std::move(new_tracks);
        }
    }
    return track_ids_by_frame;
}

std::vector<std::vector<float>> fill_missing_depths_by_track(
    const std::vector<std::vector<std::optional<float>>>& raw_z_by_frame, const std::vector<std::vector<float>>& raw_cam_t_z_by_frame,
    const std::vector<std::vector<int>>& track_ids_by_frame, float k_metric, int min_ratio_samples
) {
    std::unordered_map<int, std::vector<float>> ratios_by_track;
    for (std::size_t frame = 0; frame < raw_z_by_frame.size(); ++frame) {
        for (std::size_t i = 0; i < raw_z_by_frame[frame].size(); ++i) {
            const std::optional<float>& raw_z = raw_z_by_frame[frame][i];
            if (raw_z && raw_cam_t_z_by_frame[frame][i] > 0.0f) {
                ratios_by_track[track_ids_by_frame[frame][i]].push_back(*raw_z / raw_cam_t_z_by_frame[frame][i]);
            }
        }
    }
    std::unordered_map<int, float> k_by_track;
    for (auto& [track_id, ratios] : ratios_by_track) {
        if (static_cast<int>(ratios.size()) >= min_ratio_samples) {
            std::sort(ratios.begin(), ratios.end());
            k_by_track[track_id] = ratios[ratios.size() / 2];
        }
    }
    std::vector<std::vector<float>> filled(raw_z_by_frame.size());
    for (std::size_t frame = 0; frame < raw_z_by_frame.size(); ++frame) {
        filled[frame].resize(raw_z_by_frame[frame].size());
        for (std::size_t i = 0; i < raw_z_by_frame[frame].size(); ++i) {
            if (raw_z_by_frame[frame][i]) {
                filled[frame][i] = *raw_z_by_frame[frame][i];
                continue;
            }
            const auto found = k_by_track.find(track_ids_by_frame[frame][i]);
            const float k = found != k_by_track.end() ? found->second : k_metric;
            filled[frame][i] = k * raw_cam_t_z_by_frame[frame][i];
        }
    }
    return filled;
}
