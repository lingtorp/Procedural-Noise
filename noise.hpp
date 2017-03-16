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

    /// 2D fractional Brownian motion noise of the underlying noise algorithm
    double octaves(double x, double y, int octaves, double persistance = 1.0) const {
        double total = 0.0;

        double max_value = 0.0;
        for (size_t i = 0; i < octaves; ++i) {
            auto freq = std::pow(2, i);
            auto amplitude = std::pow(persistance, i);
            max_value += amplitude;
            auto noise = get_value(x * freq, y * freq);
            total += noise * amplitude;
        }

        // Dividing by the max amplitude sum bring it into [-1, 1] range
        return total / max_value;
    }

    /// 3D fractional Brownian motion noise of the underlying noise algorithm
    double octaves(double x, double y, double z, int octaves, double persistance = 1.0) const {
        double total = 0.0;

        double max_value = 0.0;
        for (size_t i = 0; i < octaves; ++i) {
            auto freq = std::pow(2, i);
            auto amplitude = std::pow(persistance, i);
            max_value += amplitude;
            auto noise = get_value(x * freq, y * freq, z * freq);
            total += noise * amplitude;
        }

        // Dividing by the max amplitude sum bring it into [-1, 1] range
        return total / max_value;
    }

protected:
    static inline float smoothstep(float t) { return t * t * (3 - 2 * t); }
    static inline double fade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }

    /// Linear interpolation between a and b with t as a variable
    static inline double lerp(double t, double a, double b) { return (1 - t) * a + t * b; }
};

/// Simplex noise implementation, a.k.a 'improved Perlin' noise
class Simplex : public Noise {
    double get_value(double x, double y) const {
        /// Skew the input (x, y) to (x', y')
        auto s = (x + y) / 2;
        auto X = x + s;
        auto Y = y + s;


    }
};

/// OpenSimplex noise implementation
class OpenSimplex : public Noise {
    double get_value(double x, double y)  const {
        return 0.0;
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
    std::vector<int> perms;

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
        X += 0.01; Y += 0.01; // Scale coordinates to avoid integer lines becoming zero
        /// Grid points from the chunk in the world
        int X0 = (int) std::floor(X);
        int Y0 = (int) std::floor(Y);
        int X1 = (int) std::ceil(X);
        int Y1 = (int) std::ceil(Y);

        double yf = Y - Y0; // Float offset inside the square [0, 1]
        double xf = X - X0; // Float offset inside the square [0, 1]

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
        X += 0.01; Y += 0.01; Z += 0.01; // Scale coordinates to avoid integer lines becoming zero
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
        Vec3<double> x0y0z0 = grads3[perms[(X0 + perms[Y0 + perms[Z0 % perms.size()]]) % perms.size()]];
        Vec3<double> x1y0z0 = grads3[perms[(X1 + perms[Y0 % perms[Z0 % perms.size()]]) % perms.size()]];
        Vec3<double> x0y1z0 = grads3[perms[(X0 + perms[Y1 % perms[Z0 % perms.size()]]) % perms.size()]];
        Vec3<double> x1y1z0 = grads3[perms[(X1 + perms[Y1 % perms[Z0 % perms.size()]]) % perms.size()]];

        Vec3<double> x0y0z1 = grads3[perms[(X0 + perms[Y0 + perms[Z1 % perms.size()]]) % perms.size()]];
        Vec3<double> x1y0z1 = grads3[perms[(X1 + perms[Y0 + perms[Z1 % perms.size()]]) % perms.size()]];
        Vec3<double> x0y1z1 = grads3[perms[(X0 + perms[Y1 + perms[Z1 % perms.size()]]) % perms.size()]];
        Vec3<double> x1y1z1 = grads3[perms[(X1 + perms[Y1 + perms[Z1 % perms.size()]]) % perms.size()]];

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