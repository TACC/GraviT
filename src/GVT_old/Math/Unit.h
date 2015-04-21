#ifndef Magnum_Math_Unit_h
#define Magnum_Math_Unit_h

namespace GVT { namespace Math {

/**
@brief Base class for units
@tparam T Underlying data type

@see Deg, Rad
*/
template<template<class> class Derived, class T> class Unit {
    template<template<class> class, class> friend class Unit;

    public:
        typedef T Type;             /**< @brief Underlying data type */

        Unit(): value(T(0)) {}

        explicit Unit(T value): value(value) {}

        template<class U>  explicit Unit(Unit<Derived, U> value): value(T(value.value)) {}

        operator T() const { return value; }

         bool operator==(Unit<Derived, T> other) const {
            return value == other.value;
        }

         bool operator!=(Unit<Derived, T> other) const {
            return !operator==(other);
        }

         bool operator<(Unit<Derived, T> other) const {
            return value < other.value;
        }

         bool operator>(Unit<Derived, T> other) const {
            return value > other.value;
        }

         bool operator<=(Unit<Derived, T> other) const {
            return !operator>(other);
        }

         bool operator>=(Unit<Derived, T> other) const {
            return !operator<(other);
        }

         Unit<Derived, T> operator-() const {
            return Unit<Derived, T>(-value);
        }

        Unit<Derived, T>& operator+=(Unit<Derived, T> other) {
            value += other.value;
            return *this;
        }
         Unit<Derived, T> operator+(Unit<Derived, T> other) const {
            return Unit<Derived, T>(value + other.value);
        }

        Unit<Derived, T>& operator-=(Unit<Derived, T> other) {
            value -= other.value;
            return *this;
        }

         Unit<Derived, T> operator-(Unit<Derived, T> other) const {
            return Unit<Derived, T>(value - other.value);
        }

        Unit<Derived, T>& operator*=(T number) {
            value *= number;
            return *this;
        }

         Unit<Derived, T> operator*(T number) const {
            return Unit<Derived, T>(value*number);
        }

        Unit<Derived, T>& operator/=(T number) {
            value /= number;
            return *this;
        }

         Unit<Derived, T> operator/(T number) const {
            return Unit<Derived, T>(value/number);
        }

         T operator/(Unit<Derived, T> other) const {
            return value/other.value;
        }

    private:
        T value;
};



}}

#endif
