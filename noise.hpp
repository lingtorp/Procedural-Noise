#ifndef NOISE_H
#define NOISE_H

#include <random>
#include <algorithm>
#include "vector.hpp"
#include <iostream>
#include <SDL2/SDL_log.h>

/**
 * Base class for noise generating classes
 */
class Noise {
public:
    /// 2D raw noise from the underlying noise algorithm
    virtual double get_value(double x, double y) const = 0;

    /// 3D raw noise from the underlying noise algorithm
    virtual double get_value(double x, double y, double z) const = 0;

    /// 3D turbulence noise which simulates fBm
    double turbulence(double x, double y, double zoom_factor) const {
        double value = 0;
        double zoom = zoom_factor;
        while (zoom >= 1.0) {
            value += get_value(x / zoom, y / zoom) * zoom;
            zoom /= 2;
        }
        return value / zoom_factor;
    }

    /// 3D turbulence noise which simulates fBm
    double turbulence(double x, double y, double z, double zoom_factor) const {
        double value = 0;
        double zoom = zoom_factor;
        while (zoom >= 1.0) {
            value += get_value(x / zoom, y / zoom, z / zoom) * zoom;
            zoom /= 2;
        }
        return value / zoom_factor;
    }

    /// 3D turbulence noise which simulates fBm
    double fbm(Vec3<double> in, double zoom_factor) const {
        return turbulence(in.x, in.y, in.z, zoom_factor);
    }

    /// 3D Billowy turbulence
    double turbulence_billowy(double x, double y, double z, double zoom_factor) const {
        double value = 0;
        double zoom = zoom_factor;
        while (zoom >= 1.0) {
            value += std::abs(get_value(x / zoom, y / zoom, z / zoom) * zoom);
            zoom /= 2;
        }
        return value / zoom_factor;
    }

    /// 3D Ridged turbulence
    double turbulence_ridged(double x, double y, double z, double zoom_factor) const {
        double value = 0;
        double zoom = zoom_factor;
        while (zoom >= 1.0) {
            value += (1.0f - std::abs(get_value(x / zoom, y / zoom, z / zoom) * zoom));
            zoom /= 2;
        }
        return value / zoom_factor;
    }

    /// 2D fractional Brownian motion noise of the underlying noise algorithm
    double octaves(double x, double y, int octaves, double persistance = 1.0, double amplitude = 1.0) const {
        double total = 0.0;
        double max_value = 0.0;
        double frequency = 1.0;
        for (size_t i = 0; i < octaves; ++i) {
            total += get_value(x / frequency, y / frequency) * amplitude;
            max_value += amplitude;

            amplitude *= persistance;
            frequency *= 2;
        }

        // Dividing by the max amplitude sum brings it into [-1, 1] range
        return total / max_value;
    }

    /// 3D fractional Brownian motion noise of the underlying noise algorithm
    double octaves(double x, double y, double z, int octaves, double persistance = 1.0, double amplitude = 1.0) const {
        double total = 0.0;
        double max_value = 0.0;
        double frequency = 1.0;
        for (size_t i = 0; i < octaves; ++i) {
            total += get_value(x / frequency, y / frequency, z / frequency) * amplitude;
            max_value += amplitude;

            amplitude *= persistance;
            frequency *= 2;
        }

        // Dividing by the max amplitude sum brings it into [-1, 1] range
        return total / max_value;
    }

    /// 3D fractional Brownian motion noise in which each octave gets its own amplitude
    double octaves(double x, double y, double z, const std::vector<float> &amplitudes) const {
        double total = 0.0;
        double max_value = 0.0;
        double frequency = 1.0;
        for (size_t i = 0; i < amplitudes.size(); ++i) {
            total += get_value(x / frequency, y / frequency, z / frequency) * amplitudes[i];
            max_value += amplitudes[i];
            frequency *= 2;
        }

        // Dividing by the max amplitude sum brings it into [-1, 1] range
        return total / max_value;
    }

    double domain_wrapping(double x, double y, double z, double scale) const {
        Vec3<double> p{x, y, z};
        Vec3<double> offset{50.2, 10.3, 10.5};

        Vec3<double> q{fbm(p + offset, scale), fbm(p + offset, scale), fbm(p + offset, scale)};
        Vec3<double> qq{100.0*q.x, 100.0*q.y, 100.0*q.z};

        /// Adjusting the scales in r makes a cool ripple effect through the noise
        Vec3<double> r{fbm(p + qq + Vec3<double>{1.7, 9.2, 5.1}, scale * 1),
                       fbm(p + qq + Vec3<double>{8.3, 2.8, 2.5}, scale * 1),
                       fbm(p + qq + Vec3<double>{1.2, 6.9, 8.4}, scale * 1)};
        Vec3<double> rr{100.0*r.x, 100.0*r.y, 100.0*r.z};

        return fbm(p + rr, scale);
    }

protected:
    static inline float smoothstep(float t) { return t * t * (3 - 2 * t); }
    static inline double fade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }

    /// Linear interpolation between a and b with t as a variable
    static inline double lerp(double t, double a, double b) { return (1 - t) * a + t * b; }
};

/// Simplex noise implementation, a.k.a 'improved Perlin' noise
class Simplex : public Noise {
    // Implementation details for generation of gradients
    std::mt19937 engine;
    std::uniform_real_distribution<> distr;

    /// 2D Normalized gradients table
    std::vector<Vec2<double>> grads2;

    /// 3D Normalized gradients table
    std::vector<Vec3<double>> grads3;

    /// Permutation table for indices to the gradients
    std::vector<u_char> perms;
public:
    /// Perms size is double that of grad to avoid index wrapping
    Simplex(uint64_t seed): engine(seed), grads2(256), grads3(256), distr(-1.0, 1.0), perms(512) {
        /// Fill the gradients list with random normalized vectors
        for (int i = 0; i < grads2.size(); i++) {
            double x = distr(engine);
            double y = distr(engine);
            double z = distr(engine);
            auto grad_vector = Vec2<double>{x, y}.normalize();
            grads2[i] = grad_vector;
            auto grad3_vector = Vec3<double>{x, y, z}.normalize();
            grads3[i] = grad3_vector;
        }

        /// Fill gradient lookup array with random indices to the gradients list
        /// Fill with indices from 0 to perms.size()
        std::iota(perms.begin(), perms.end(), 0);

        /// Randomize the order of the indices
        std::shuffle(perms.begin(), perms.end(), engine);
    }

    double grad(int hash, float x, float y) const {
        int h = hash & 0b0111;    // Convert low 3 bits of hash code
        float u = h < 4 ? x : y;  // into 8 simple gradient directions,
        float v = h < 4 ? y : x;  // and compute the dot product with (x,y).
        return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f*v : 2.0f*v);
    }

    double get_value(double x, double y) const {
        #define F2 0.366025403f // F2 = 0.5*(sqrt(3.0)-1.0)
        #define G2 0.211324865f // G2 = (3.0-Math.sqrt(3.0))/6.0

        float n0, n1, n2; // Noise contributions from the three corners

        // Skew the input space to determine which simplex cell we're in
        float s = (x + y) * F2; // Hairy factor for 2D
        float xs = x + s;
        float ys = y + s;
        int i = (int) std::floor(xs);
        int j = (int) std::floor(ys);

        float t = (float) (i + j) * G2;
        float X0 = i - t;  // Unskew the cell origin back to (x,y) space
        float Y0 = j - t;
        float x0 = x - X0; // The x,y distances from the cell origin
        float y0 = y - Y0;

        // For the 2D case, the simplex shape is an equilateral triangle.
        // Determine which simplex we are in.
        int i1, j1; // Offsets for second vertex of simplex in (i,j) coords
        if (x0 > y0) { i1 = 1; j1 = 0; } // lower triangle, XY order: (0,0)->(1,0)->(1,1)
        else { i1 = 0; j1 = 1; }         // upper triangle, YX order: (0,0)->(0,1)->(1,1)

        // A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
        // a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
        // c = (3-sqrt(3))/6

        float x1 = x0 - i1 + G2; // Offsets for second vertex in (x,y) unskewed coords
        float y1 = y0 - j1 + G2;
        float x2 = x0 - 1.0f + 2.0f * G2; // Offsets for last corner in (x,y) unskewed coords
        float y2 = y0 - 1.0f + 2.0f * G2;

        // Wrap the integer indices at 256, to avoid indexing perm[] out of bounds
        int ii = i % perms.size();
        int jj = j % perms.size();

        // Calculate the contribution from the three corners
        float t0 = 0.5f - x0*x0 - y0*y0;
        if (t0 < 0.0f) n0 = 0.0f;
        else {
            t0 *= t0;
            n0 = t0 * t0 * grad(perms[ii + perms[jj]], x0, y0);
        }

        float t1 = 0.5f - x1*x1 - y1*y1;
        if (t1 < 0.0f) n1 = 0.0f;
        else {
            t1 *= t1;
            n1 = t1 * t1 * grad(perms[ii + i1 + perms[jj + j1]], x1, y1);
        }

        float t2 = 0.5f - x2*x2 - y2*y2;
        if (t2 < 0.0f) n2 = 0.0f;
        else {
            t2 *= t2;
            n2 = t2 * t2 * grad(perms[ii + 1 + perms[jj + 1]], x2, y2);
        }

        // Add contributions from each corner to get the final noise value.
        // The result is scaled to return values in the interval [-1,1].
        return 20.0f * (n0 + n1 + n2); // TODO: The scale factor is preliminary!
    }

    // TODO: Implement
    double get_value(double x, double y, double z) const { exit(0); return 0.0; }
};

class ImprovedPerlin : public Noise {
    // Implementation details for generation of gradients
    std::mt19937 engine;
    std::uniform_real_distribution<> distr;

    /// 2D Normalized gradients table
    std::vector<Vec2<double>> grads2;

    /// 3D Normalized gradients table
    std::vector<Vec3<double>> grads3;

    /// Permutation table for indices to the gradients
    std::vector<u_char> perms;

    std::vector<u_char> bit_patterns;
public:
    /// Perms size is double that of grad to avoid index wrapping
    ImprovedPerlin(uint64_t seed): engine(seed), grads2(256), grads3(256), distr(-1.0, 1.0), perms(512),
                                   bit_patterns{0x15, 0x38, 0x32, 0x2C, 0x0D, 0x13, 0x07, 0x2A} {
        /// Fill the gradients list with random normalized vectors
        for (int i = 0; i < grads2.size(); i++) {
            double x = distr(engine);
            double y = distr(engine);
            double z = distr(engine);
            auto grad_vector = Vec2<double>{x, y}.normalize();
            grads2[i] = grad_vector;
            auto grad3_vector = Vec3<double>{x, y, z}.normalize();
            grads3[i] = grad3_vector;
        }

        /// Fill gradient lookup array with random indices to the gradients list
        /// Fill with indices from 0 to perms.size()
        std::iota(perms.begin(), perms.end(), 0);

        /// Randomize the order of the indices
        std::shuffle(perms.begin(), perms.end(), engine);
    }

    u_char b(int i, int j, int k, int B) {
        auto bit_index = 4*(i & (0b1 << B)) + 2*(j & (0b1 << B)) + (k & (0b1 << B));
        return bit_patterns[bit_index];
    }

    Vec3<float> grad(int i, int j, int k) {
        int bit_sum = b(i,j,k,0) + b(j,k,i,1) + b(k,i,j,2) + b(i,j,k,3) + b(j,k,i,4) + b(k,i,j,5) + b(i,j,k,6) + b(j,k,i,7);
        auto u = (bit_sum & 0b001) ? 1.0f : 0.0f;
        auto v = (bit_sum & 0b010) ? 1.0f : 0.0f;
        auto w = (bit_sum & 0b100) ? 1.0f : 0.0f;
        u = (bit_sum & 0b100000) ? -u : u;
        v = (bit_sum & 0b010000) ? -v : v;
        w = (bit_sum & 0b001000) ? -w : w;
        return {u, v, w};
    }

    double get_value(double x, double y) const {
        x += 0.01; y += 0.01; // Skew coordinates to avoid integer lines becoming zero
        /// 1. Coordinate skewing
        /// Skew the input (x, y) to (x', y')
        const auto F = (std::sqrt(2 + 1) - 1) / 2; // Scale factor: sqrt(n + 1) - 1 / n
        auto s = (x + y) * F; // s for skewed
        auto xs = x + s;
        auto ys = y + s;

        /// Determine which of the skewed unit hypercubes (x, y) lies within
        auto xs0 = (int) std::floor(xs); // Vertex A coordinates
        auto ys0 = (int) std::floor(ys);

        /// 2. Simplical subdivision - finding the simplex consisting of vertices (A, B, C)
        /// Simplex vertices
        auto xs1 = 0; // Vertex B coordinates
        auto ys1 = 0;

        // Offset to vertex B from A
        auto x_step = 0;
        auto y_step = 0;
        // Checks which simplex the point (xs, ys) is in
        if (xs > ys) { // Upper triangle
            xs1 = xs0 + F;
            ys1 = ys0 + F + 1;
            y_step = 1;
        } else {       // Lower triangle
            xs1 = xs0 + F + 1;
            ys1 = ys0 + F;
            x_step = 1;
        }

        auto xs2 = xs0 + F + F + 1; // Last vertex (C) is always the same
        auto ys2 = ys0 + F + F + 1;

        /// 3. Gradient selection
        // Hash the coordinates of the simplex to get the gradient indices for A, B, C
        auto ii = xs0 % 255;
        auto jj = ys0 % 255;
        auto gai = perms[((ii          % perms.size()) + perms[jj          % perms.size()]) % perms.size()];
        auto gbi = perms[((ii + x_step % perms.size()) + perms[jj + y_step % perms.size()]) % perms.size()];
        auto gci = perms[((ii + 1      % perms.size()) + perms[jj + 1      % perms.size()]) % perms.size()];

        auto ga = grads2[gai];
        auto gb = grads2[gbi];
        auto gc = grads2[gci];

        /// 4. Kernel (gradient) summation
        const auto G = (1 - 1/std::sqrt(2 + 1)) / 2;
        auto x1 = xs1 + (xs1 + ys1) * G;
        auto y1 = ys1 + (xs1 + ys1) * G;

        auto x2 = xs2 + (xs2 + ys2) * G;
        auto y2 = ys2 + (xs2 + ys2) * G;

        /// Displacement vectors for point (x, y)
        auto x0 = std::floor(x);
        auto y0 = std::floor(y);
        Vec2<double> da = {x0 - x, y0 - y};
        Vec2<double> db = {x1 - x, y1 - y};
        Vec2<double> dc = {x2 - x, y2 - y};
        /// Gradient contributions from the vertices
        double radius = 0.6;
        auto result_a = std::pow(std::max(0.0, radius - da.dot(da)), 1) * da.dot(ga);
        auto result_b = std::pow(std::max(0.0, radius - db.dot(db)), 1) * db.dot(gb);
        auto result_c = std::pow(std::max(0.0, radius - dc.dot(dc)), 1) * dc.dot(gc);

        return 1*(result_a + result_b + result_c);
    }

    // TODO: Implement
    double get_value(double x, double y, double z) const { exit(0); return 0.0; }
};

class ImprovedPerlin2 : public Noise {
    // Implementation details for generation of gradients
    std::mt19937 engine;
    std::uniform_real_distribution<> distr;

    /// 2D Normalized gradients table
    std::vector<Vec2<float>> grads2;

    /// 3D Normalized gradients table
    std::vector<Vec3<double>> grads3;

    /// Permutation table for indices to the gradients
    std::vector<u_char> perms;
public:
    /// Perms size is double that of grad to avoid index wrapping
    ImprovedPerlin2(uint64_t seed): engine(seed), grads2(256), grads3(256), distr(-1.0, 1.0), perms(512) {
        /// Fill the gradients list with random normalized vectors
        for (int i = 0; i < grads2.size(); i++) {
            double x = distr(engine);
            double y = distr(engine);
            double z = distr(engine);
            auto grad_vector = Vec2<float>{(float) x, (float) y}.normalize();
            grads2[i] = grad_vector;
            auto grad3_vector = Vec3<double>{x, y, z}.normalize();
            grads3[i] = grad3_vector;
        }

        /// Fill gradient lookup array with random indices to the gradients list
        /// Fill with indices from 0 to perms.size()
        std::iota(perms.begin(), perms.end(), 0);

        /// Randomize the order of the indices
        std::shuffle(perms.begin(), perms.end(), engine);
    }

    double get_value(double x, double y) const override {
        const float F = (std::sqrtf(1.0f + 2.0f) - 1.0f) / 2;
        float s = (x + y) * F;
        float xs = x + s;
        float ys = y + s;
        int i = (int) std::floorf(xs);
        int j = (int) std::floorf(ys);

        const float G = (3.0f - std::sqrtf(2.0f + 1.0f)) / 6.0f;
        auto t = (i + j) * G;
        Vec2<float> cell_origin{i - t, j - t};
        Vec2<float> vertex_a = Vec2<float>{(float)x, (float)y} - cell_origin;

        auto x_step = 0;
        auto y_step = 0;
        if (vertex_a.x > vertex_a.y) {
            // Lower triangle
            x_step = 1;
        } else {
            y_step = 1;
        }

        Vec2<float> vertex_b{vertex_a.x - x_step + G, vertex_a.y - y_step + G};
        Vec2<float> vertex_c{vertex_a.x - 1.0f + 2.0f*G, vertex_a.y - 1.0f + 2.0f*G};

        auto ii = i % 255;
        auto jj = j % 255;
        auto grad_a = grads2[perms[ii + perms[jj]]];
        auto grad_b = grads2[perms[ii + x_step + perms[jj + y_step]]];
        auto grad_c = grads2[perms[ii + 1 + perms[jj + 1]]];

        auto t0 = 0.5 - vertex_a.x*vertex_a.x - vertex_a.y*vertex_a.y;
        auto result_a = 0.0f;
        if (t0 > 0) {
            t0 *= t0;
            result_a = t0 * t0 * grad_a.dot(vertex_a);
        }

        auto t1 = 0.5 - vertex_b.x*vertex_b.x - vertex_b.y*vertex_b.y;
        auto result_b = 0.0f;
        if (t1 > 0) {
            t1 *= t1;
            result_b = t1 * t1 * grad_b.dot(vertex_b);
        }

        auto t2 = 0.5 - vertex_c.x*vertex_c.x - vertex_c.y*vertex_c.y;
        auto result_c = 0.0f;
        if (t2 > 0) {
            t2 *= t2;
            result_c = t2 * t2 * grad_c.dot(vertex_c);
        }

        return 70.0f * (result_a + result_b + result_c);
    }

    double get_value(double x, double y, double z) const override {
        return 0;
    }
};

/// Standard Perlin noise implementation
class Perlin : public Noise {
    // Implementation details for generation of gradients
    std::mt19937 engine;
    std::uniform_real_distribution<> distr;

    /// 2D Normalized gradients table
    std::vector<Vec2<double>> grads;

    /// 3D Normalized gradients table
    std::vector<Vec3<double>> grads3;

    /// Permutation table for indices to the gradients
    std::vector<u_char> perms;

public:
    Perlin(uint64_t seed): engine(seed), grads(256), grads3(256), distr(-1.0, 1.0), perms(256) {
        /// Fill the gradients list with random normalized vectors
        for (int i = 0; i < grads.size(); i++) {
            double x = distr(engine);
            double y = distr(engine);
            double z = distr(engine);
            auto grad_vector = Vec2<double>{x, y}.normalize();
            grads[i] = grad_vector;
            auto grad3_vector = Vec3<double>{x, y, z}.normalize();
            grads3[i] = grad3_vector;
        }

        /// Fill gradient lookup array with random indices to the gradients list
        /// Fill with indices from 0 to perms.size()
        std::iota(perms.begin(), perms.end(), 0);

        /// Randomize the order of the indices
        std::shuffle(perms.begin(), perms.end(), engine);
    }

    double get_value(double X, double Y) const {
        /// Compress the coordinates inside the chunk; double part + int part = point coordinate
        X += 0.01; Y += 0.01; // Skew coordinates to avoid integer lines becoming zero
        /// Grid points from the chunk in the world
        int X0 = (int) std::floor(X);
        int Y0 = (int) std::floor(Y);
        int X1 = (int) std::ceil(X);
        int Y1 = (int) std::ceil(Y);

        /// Gradients using hashed indices from lookup list
        Vec2<double> x0y0 = grads[perms[(X0 + perms[Y0 % perms.size()]) % perms.size()]];
        Vec2<double> x1y0 = grads[perms[(X1 + perms[Y0 % perms.size()]) % perms.size()]];
        Vec2<double> x0y1 = grads[perms[(X0 + perms[Y1 % perms.size()]) % perms.size()]];
        Vec2<double> x1y1 = grads[perms[(X1 + perms[Y1 % perms.size()]) % perms.size()]];

        /// Vectors from gradients to point in unit square
        auto v00 = Vec2<double>{X - X0, Y - Y0}.normalize();
        auto v10 = Vec2<double>{X - X1, Y - Y0}.normalize();
        auto v01 = Vec2<double>{X - X0, Y - Y1}.normalize();
        auto v11 = Vec2<double>{X - X1, Y - Y1}.normalize();

        /// Contribution of gradient vectors by dot product between relative vectors and gradients
        double d00 = x0y0.dot(v00);
        double d10 = x1y0.dot(v10);
        double d01 = x0y1.dot(v01);
        double d11 = x1y1.dot(v11);

        /// Interpolate dot product values at sample point using polynomial interpolation 6x^5 - 15x^4 + 10x^3
        double yf = Y - Y0; // Float offset inside the square [0, 1]
        double xf = X - X0; // Float offset inside the square [0, 1]

        auto wx = fade(xf);
        auto wy = fade(yf);

        /// Interpolate along x for the contributions from each of the gradients
        auto xa = lerp(wx, d00, d10);
        auto xb = lerp(wx, d01, d11);

        auto val = lerp(wy, xa, xb);

        return val;
    }

    double get_value(double X, double Y, double Z) const {
        /// Compress the coordinates inside the chunk; double part + int part = point coordinate
        /// Grid points from the chunk in the world
        int X0 = (int) std::floor(X);
        int Y0 = (int) std::floor(Y);
        int X1 = (int) std::ceil(X);
        int Y1 = (int) std::ceil(Y);
        int Z0 = (int) std::floor(Z);
        int Z1 = (int) std::ceil(Z);

        double yf = Y - Y0; // Float offset inside the cube [0, 1]
        double xf = X - X0; // Float offset inside the cube [0, 1]
        double zf = Z - Z0; // Float offset inside the cube [0, 1]

        /// Gradients using hashed indices from lookup list
        Vec3<double> x0y0z0 = grads3[perms[(X0 + perms[(Y0 + perms[Z0 % perms.size()]) % perms.size()]) % perms.size()]];
        Vec3<double> x1y0z0 = grads3[perms[(X1 + perms[(Y0 + perms[Z0 % perms.size()]) % perms.size()]) % perms.size()]];
        Vec3<double> x0y1z0 = grads3[perms[(X0 + perms[(Y1 + perms[Z0 % perms.size()]) % perms.size()]) % perms.size()]];
        Vec3<double> x1y1z0 = grads3[perms[(X1 + perms[(Y1 + perms[Z0 % perms.size()]) % perms.size()]) % perms.size()]];

        Vec3<double> x0y0z1 = grads3[perms[(X0 + perms[(Y0 + perms[Z1 % perms.size()]) % perms.size()]) % perms.size()]];
        Vec3<double> x1y0z1 = grads3[perms[(X1 + perms[(Y0 + perms[Z1 % perms.size()]) % perms.size()]) % perms.size()]];
        Vec3<double> x0y1z1 = grads3[perms[(X0 + perms[(Y1 + perms[Z1 % perms.size()]) % perms.size()]) % perms.size()]];
        Vec3<double> x1y1z1 = grads3[perms[(X1 + perms[(Y1 + perms[Z1 % perms.size()]) % perms.size()]) % perms.size()]];

        /// Vectors from gradients to point in unit cube
        auto v000 = Vec3<double>{X - X0, Y - Y0, Z - Z0};
        auto v100 = Vec3<double>{X - X1, Y - Y0, Z - Z0};
        auto v010 = Vec3<double>{X - X0, Y - Y1, Z - Z0};
        auto v110 = Vec3<double>{X - X1, Y - Y1, Z - Z0};

        auto v001 = Vec3<double>{X - X0, Y - Y0, Z - Z1};
        auto v101 = Vec3<double>{X - X1, Y - Y0, Z - Z1};
        auto v011 = Vec3<double>{X - X0, Y - Y1, Z - Z1};
        auto v111 = Vec3<double>{X - X1, Y - Y1, Z - Z1};

        /// Contribution of gradient vectors by dot product between relative vectors and gradients
        double d000 = x0y0z0.dot(v000);
        double d100 = x1y0z0.dot(v100);
        double d010 = x0y1z0.dot(v010);
        double d110 = x1y1z0.dot(v110);

        double d001 = x0y0z1.dot(v001);
        double d101 = x1y0z1.dot(v101);
        double d011 = x0y1z1.dot(v011);
        double d111 = x1y1z1.dot(v111);

        /// Interpolate dot product values at sample point using polynomial interpolation 6x^5 - 15x^4 + 10x^3
        auto wx = fade(xf);
        auto wy = fade(yf);
        auto wz = fade(zf);

        /// Interpolate along x for the contributions from each of the gradients
        auto xa = lerp(wx, d000, d100);
        auto xb = lerp(wx, d010, d110);

        auto xc = lerp(wx, d001, d101);
        auto xd = lerp(wx, d011, d111);

        /// Interpolate along y for the contributions from each of the gradients
        auto ya = lerp(wy, xa, xb);
        auto yb = lerp(wy, xc, xd);

        /// Interpolate along z for the contributions from each of the gradients
        auto za = lerp(wz, ya, yb);

        return za;
    }
};

#endif // NOISE_H