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
    double octaves(double x, double y, double z, const std::vector<double> &amplitudes) const {
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

    /// Warps the domain of the noise function creating more natural looking features
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
    static inline double smoothstep(double t) { return t * t * (3 - 2 * t); }
    static inline double fade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }

    /// Linear interpolation between a and b with t as a variable
    static inline double lerp(double t, double a, double b) { return (1 - t) * a + t * b; }
};

/**
 * Simplex noise/Improved Perlin noise from the 'Improved noise' patent
 * - Gradient creation on-the-fly using bit manipulation
 * - Gradient selection uses bit manipulation (from the above point)
 */
class Simplex_Patent : public Noise {
    // Implementation details
    /// Bit patterns for the creation of the gradients
    std::vector<u_char> bit_patterns;
public:
    Simplex_Patent(uint64_t seed): bit_patterns{0x15, 0x38, 0x32, 0x2C, 0x0D, 0x13, 0x07, 0x2A} { }

    /***************** Simplex 2D Noise *****************/

    /// Given a coordinate (i, j) selects the B'th bit
    u_char b(int i, int j, int B) const {
        auto bit_index = 2*(i & (0b1 << B)) + (j & (0b1 << B));
        return bit_patterns[bit_index];
    }

    /// Given a coordinate (i, j) generates a gradient vector
    Vec2<double> grad(int i, int j) const {
        int bit_sum = b(i,j,0) + b(j,i,1) + b(i,j,2) + b(j,i,3);
        auto u = (bit_sum & 0b01) ? 1.0f : 0.0f;
        auto v = (bit_sum & 0b10) ? 1.0f : 0.0f;
        u = (bit_sum & 0b1000) ? -u : u;
        v = (bit_sum & 0b0100) ? -v : v;
        return {u, v};
    }

    double get_value(double x, double y) const override {
        /// Skew
        const double F = (std::sqrt(2.0 + 1.0) - 1.0) / 2.0;
        double s = (x + y) * F;
        double xs = x + s;
        double ys = y + s;
        int i = (int) std::floor(xs);
        int j = (int) std::floor(ys);

        /// Unskew - find first vertex of the simplex
        const double G = (3.0 - std::sqrt(2.0 + 1.0)) / 6.0;
        double t = (i + j) * G;
        Vec2<double> cell_origin{i - t, j - t};
        Vec2<double> vertex_a = Vec2<double>{x, y} - cell_origin;

        // Figure out which vertex is next
        auto x_step = 0;
        auto y_step = 0;
        if (vertex_a.x > vertex_a.y) { // Lower triangle
            x_step = 1;
        } else {
            y_step = 1;
        }

        // A change of one unit step is; x = x' + (x' + y') * G <--> x = 1.0 + (1.0 + 1.0) * G <--> x = 1.0 + 2.0 * G
        Vec2<double> vertex_b{vertex_a.x - x_step + G, vertex_a.y - y_step + G};
        Vec2<double> vertex_c{vertex_a.x - 1.0 + 2.0 * G, vertex_a.y - 1.0 + 2.0 * G};

        auto grad_a = grad(i, j);
        auto grad_b = grad(i + x_step, j + y_step);
        auto grad_c = grad(i + 1, j + 1);

        /// Calculate contribution from the vertices in a circle
        // max(0, r^2 - d^2)^4 * gradient.dot(vertex)
        const double radius = 0.6 * 0.6; // Radius of the surflet circle (0.6 in patent)
        double sum = 0.0;

        double t0 = radius - vertex_a.length()*vertex_a.length();
        if (t0 > 0) {
            sum += std::pow(t0, 4) * grad_a.dot(vertex_a);
        }

        double t1 = radius - vertex_b.length()*vertex_b.length();
        if (t1 > 0) {
            sum += std::pow(t1, 4) * grad_b.dot(vertex_b);
        }

        double t2 = radius - vertex_c.length()*vertex_c.length();
        if (t2 > 0) {
            sum += std::pow(t2, 4) * grad_c.dot(vertex_c);
        }

        return 220.0 * sum;
    }

    /***************** Simplex 3D Noise *****************/

    /// Hashes a coordinate (i, j, k) then selectes one of the bit patterns
    u_char b(int i, int j, int k, int B) const {
        auto bit_index = 4*(i & (0b1 << B)) + 2*(j & (0b1 << B)) + (k & (0b1 << B));
        return bit_patterns[bit_index];
    }

    /// Given an coordinate (i, j, k) creates a gradient vector
    Vec3<double> grad(int i, int j, int k) const {
        int bit_sum = b(i,j,k,0) + b(j,k,i,1) + b(k,i,j,2) + b(i,j,k,3) + b(j,k,i,4) + b(k,i,j,5) + b(i,j,k,6) + b(j,k,i,7);
        auto u = (bit_sum & 0b001) ? 1.0f : 0.0f;
        auto v = (bit_sum & 0b010) ? 1.0f : 0.0f;
        auto w = (bit_sum & 0b100) ? 1.0f : 0.0f;
        u = (bit_sum & 0b100000) ? -u : u;
        v = (bit_sum & 0b010000) ? -v : v;
        w = (bit_sum & 0b001000) ? -w : w;
        return {u, v, w};
    }

    double get_value(double x, double y, double z) const {
        const double F = (std::sqrt(1.0 + 3.0) - 1.0) / 3;
        double s = (x + y + z) * F;
        double xs = x + s;
        double ys = y + s;
        double zs = z + s;
        int i = (int) std::floor(xs);
        int j = (int) std::floor(ys);
        int k = (int) std::floor(zs);

        const double G = (1.0 - (1.0 / sqrt(3.0 + 1.0))) / 3.0;
        double t = (i + j + k) * G;
        Vec3<double> cell_origin{i - t, j - t, k - t};

    }
};

/**
 * Simplex noise implementation using the 'Improved Noise' patent algorithm
 * - Gradient table instead of on-the-fly gradient creation
 * - Permutation table instead of bit manipulation
 * - Using modulo hashing to select the gradients
 */
class Simplex_Tables : public Noise {
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
    Simplex_Tables(uint64_t seed): engine(seed), grads2(256), grads3(256), distr(-1.0, 1.0), perms(512) {
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

    double get_value(double x, double y) const override {
        const double F = (std::sqrt(2.0 + 1.0) - 1.0) / 2.0; // F = (sqrt(n + 1) - 1) / n
        double s = (x + y) * F;
        double xs = x + s;
        double ys = y + s;
        int i = (int) std::floor(xs);
        int j = (int) std::floor(ys);

        const double G = (3.0 - std::sqrt(2.0 + 1.0)) / 6.0; // G = (1 - (1 / sqrt(n + 1)) / n
        double t = (i + j) * G;
        Vec2<double> cell_origin{i - t, j - t};
        Vec2<double> vertex_a = Vec2<double>{x, y} - cell_origin;

        auto x_step = 0;
        auto y_step = 0;
        if (vertex_a.x > vertex_a.y) { // Lower triangle
            x_step = 1;
        } else {
            y_step = 1;
        }

        Vec2<double> vertex_b{vertex_a.x - x_step + G, vertex_a.y - y_step + G};
        Vec2<double> vertex_c{vertex_a.x - 1.0 + 2.0 * G, vertex_a.y - 1.0 + 2.0 * G};

        auto ii = i % 255;
        auto jj = j % 255;
        auto grad_a = grads2[perms[ii + perms[jj]]];
        auto grad_b = grads2[perms[ii + x_step + perms[jj + y_step]]];
        auto grad_c = grads2[perms[ii + 1 + perms[jj + 1]]];

        /// Calculate contribution from the vertices in a circle
        // max(0, r^2 - d^2)^4 * gradient.dot(vertex)
        const double radius = 0.6 * 0.6; // Radius of the surflet circle (0.6 in patent)
        double sum = 0.0;

        double t0 = radius - vertex_a.length()*vertex_a.length();
        if (t0 > 0) {
            sum += std::pow(t0, 4) * grad_a.dot(vertex_a);
        }

        double t1 = radius - vertex_b.length()*vertex_b.length();
        if (t1 > 0) {
            sum += std::pow(t1, 4) * grad_b.dot(vertex_b);
        }

        double t2 = radius - vertex_c.length()*vertex_c.length();
        if (t2 > 0) {
            sum += std::pow(t2, 4) * grad_c.dot(vertex_c);
        }

        return 128.0 * sum;
    }

    // TODO: Implement
    double get_value(double x, double y, double z) const override { exit(EXIT_FAILURE); }
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
        double yf = Y - Y0; // Float offset inside the cube [0, 1]
        double xf = X - X0; // Float offset inside the cube [0, 1]
        double zf = Z - Z0; // Float offset inside the cube [0, 1]

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