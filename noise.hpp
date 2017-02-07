#ifndef MEINEKRAFT_NOISE_H
#define MEINEKRAFT_NOISE_H

#include <random>
#include "vector.hpp"
#include <iostream>

class Noise {
    uint64_t seed;
    std::mt19937 engine;
    std::uniform_real_distribution<> distr;
    std::vector<Vec2<double>> grads; /// Normalized gradients
    std::vector<int> perms;

public:
    Perlin(uint64_t seed): engine(seed), grads(256), distr(-1, 1), perms(256) {
        /// Fill the gradients list with random normalized vectors
        for (int i = 0; i < grads.size(); i++) {
            double x = distr(engine);
            double y = distr(engine);
            auto grad_vector = Vec2<double>{x, y}.normalize();
            grads[i] = grad_vector;
        }

        /// Fill gradient lookup array with random indices to the gradients list
        /// Fill with indices from 0 to perms.size()
        std::iota(perms.begin(), perms.end(), 0);

        /// Randomize the order of the indices
        std::shuffle(perms.begin(), perms.end(), engine);
    }

    /// Octaves of 2D noise
    /// Produces fractal noise (a.k.a brown noise/brownian noise)
    double octaves_of_2d(double x, double y, int octaves, double persistance = 1.0) const {
        double total = 0.0;

        double max_value = 0.0;
        for (size_t i = 0; i < octaves; ++i) {
            auto freq = std::pow(2, i);
            auto amplitude = std::pow(persistance, i);
            max_value += amplitude;
            total += noise_2d(x * freq, y * freq) * amplitude;
        }

        return total / max_value;
    }

    /// 2D Perlin noise (x, y), chunk_pos gives the frame for the coord (x, y) and dimension is the chunks size
    double perlin_2d(double X, double Y) const {
        /// Compress the coordinates inside the chunk; double part + int part = point coordinate
        X += 0.01; Y += 0.01; // Scale coordinates to avoid integer becoming zero
        /// Grid points from the chunk in the world
        int X0 = (int) std::floor(X);
        int Y0 = (int) std::floor(Y);
        int X1 = (int) std::ceil(X);
        int Y1 = (int) std::ceil(Y);

        double yf = Y - Y0; // Float offset inside the chunk (0, 1)
        double xf = X - X0; // Float offset inside the chunk (0, 1)

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

        auto ya = lerp(wy, xa, xb);

        return ya;
    }

private:
    static inline float smoothstep(const float &t) { return t * t * (3 - 2 * t); }
    static inline double fade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }
    static inline double lerp(double t, double a, double b) { return (1 - t) * a + t * b; }
};

#endif //MEINEKRAFT_NOISE_H