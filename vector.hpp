#ifndef VECTOR_H
#define VECTOR_H

/**
 * A collection of linear algebra data types mainly vectors and their operations
 */

#include <math.h>
#include <ostream>
#include <cmath>
#include <vector>

/// Forward declarations
template<typename T>
struct Vec2;

template<typename T>
struct Vec3;

template<typename T>
struct Vec3 {
    T x, y, z;

    Vec3(T x, T y, T z): x(x), y(y), z(z) {};
    Vec3(): x(0), y(0), z(0) {};

    /// Vec3 of zeroes
    inline static Vec3 ZERO() { return Vec3(0.0, 0.0, 0.0); }

    /// Length of the vector
    inline double length() const { return std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2)); }

    /// Returns a copy of this vector normalized
    inline Vec3<T> normalize() const {
        double length = this->length();
        Vec3 result;
        result.x = x / length;
        result.y = y / length;
        result.z = z / length;
        return result;
    }

    /// Result = v x u
    inline Vec3<T> cross(Vec3<T> u) const {
        Vec3<T> result;
        result.x = y * u.z - z * u.y;
        result.y = z * u.x - x * u.z;
        result.z = x * u.y - y * u.x;
        return result;
    }

    /// Sum of the components of the vector
    inline T sum() const { return x + y + z; }

    /// Floors the components
    Vec3<T> floor() const { return {std::floor(x), std::floor(y), std::floor(z)}; }

    /// Dot product
    inline T dot(Vec3<T> u) const { return x * u.x + y * u.y + z * u.z; }

    /// Operators
    inline Vec3<T> operator+(const Vec3 &rhs) const {
        return Vec3<T>{x + rhs.x, y + rhs.y, z + rhs.z};
    }

    inline Vec3<T> operator+(const double rhs) const {
        return Vec3<T>{x + rhs, y + rhs, z + rhs};
    }

    inline Vec3<T> operator*(const Vec3 &rhs) const {
        return Vec3<T>{x * rhs.x, y * rhs.y, z * rhs.z};
    }

    inline Vec3<T> operator*(const T s) const {
        return Vec3<T>{x * s, y * s, z * s};
    }

    inline bool operator==(const Vec3 &rhs) const {
        return (x == rhs.x) && (y == rhs.y) && (z == rhs.z);
    }

    friend std::ostream &operator<<(std::ostream& os, const Vec3 &vec) {
        return os << "(x:" << vec.x << " y:" << vec.y << " z:" << vec.z << ")";
    }

    inline Vec3 operator-(const Vec3 &rhs) const {
        return Vec3{x - rhs.x, y - rhs.y, z - rhs.z};
    }
};

template<typename T>
struct Vec2 {
    T x, y = 0.0f;
    Vec2(T x, T y): x(x), y(y) {};
    Vec2(): x(0), y(0) {};

    /// Sum of the components of the vector
    inline T sum() const { return x + y; }

    /// Floors the components and returns a copy
    inline Vec2<T> floor() const { return {std::floor(x), std::floor(y)}; }

    /// Dot product
    inline T dot(Vec2<T> u) const { return x * u.x + y * u.y; }

    /// Operators
    Vec2<T> operator+(const Vec2 &rhs) const {
        return {x + rhs.x, y + rhs.y};
    }

    Vec2<T> operator-(const Vec2 &rhs) const {
        return {x - rhs.x, y - rhs.y};
    }

    bool operator==(const Vec2 &rhs) const {
        return x == rhs.x && y == rhs.y;
    }

    /// Returns a copy of this vector normalized
    inline Vec2<T> normalize() const {
        double length = this->length();
        Vec2<T> result;
        result.x = x / length;
        result.y = y / length;
        return result;
    }

    /// Length of the vector
    inline double length() const { return std::sqrt(std::pow(x, 2) + std::pow(y, 2)); }
};

#endif //VECTOR_H
