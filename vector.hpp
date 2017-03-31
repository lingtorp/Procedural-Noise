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
struct Vec4;

template<typename T>
struct Vec4 {
    T x, y, z, w;

    Vec4(T x, T y, T z, T w): x(x), y(y), z(z), w(w) { };
    Vec4(T x, T y, T z): x(x), y(y), z(z), w(0.0f) { };
    Vec4(): x(0), y(0), z(0), w(0) { };
    Vec4(Vec3<T> vec): x(vec.x), y(vec.y), z(vec.z), w(0.0) { };

    /// Operators
    /// Returns the members x, y, z, w in index order (invalid indexes returns w)
    T& operator[] (const int index) {
        switch (index) { // Should be a jump table when optimised
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            case 3:
                return w;
            default:
                return x;
        }
    }

    bool operator==(const Vec4 &rhs) {
        return x == rhs.x && y == rhs.y && z == rhs.z && w == rhs.w;
    }

    friend std::ostream &operator<<(std::ostream &os, const Vec4 &vec) {
        return os << "(x:" << vec.x << " y:" << vec.y << " z:" << vec.z << " w:" << vec.w << ")";
    }
};

template<typename T>
struct Vec3 {
    T x, y, z;

    Vec3(T x, T y, T z): x(x), y(y), z(z) {};
    Vec3(): x(0), y(0), z(0) {};

    inline static Vec3 ZERO() { return Vec3(0.0, 0.0, 0.0); }
    inline double length() const { return std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2)); }

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

    inline T dot(Vec3<T> u) const { return x * u.x + y * u.y + z * u.z; }

    /// Operators
    inline Vec3<T> operator+(const Vec3 &rhs) const {
        return Vec3<T>{x + rhs.x, y + rhs.y, z + rhs.z};
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

    /// Dot product
    inline T dot(Vec2<T> u) const { return x * u.x + y * u.y; }

    /// Operators
    Vec2<T> operator-(const Vec2 &rhs) const {
        return {x - rhs.x, y - rhs.y};
    }

    bool operator==(const Vec2 &rhs) const {
        return x == rhs.x && y == rhs.y;
    }

    inline Vec2<T> normalize() const {
        double length = this->length();
        Vec2<T> result;
        result.x = x / length;
        result.y = y / length;
        return result;
    }

    inline double length() const { return std::sqrt(std::pow(x, 2) + std::pow(y, 2)); }
};

#endif //VECTOR_H
