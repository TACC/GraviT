
#ifndef GVT_ANGLE_H
#define GVT_ANGLE_H

#include <GVT/Math/Constants.h>
#include <GVT/Math/Unit.h>



namespace GVT { namespace Math {
template<class T> class Deg;
template<class T> class Rad;

template<class T> class Deg: public Unit<Deg, T> {
    public:
       Deg() {}

         explicit Deg(T value): Unit<Math::Deg, T>(value) {}

         Deg(Unit<Math::Deg, T> value): Unit<Math::Deg, T>(value) {}

        template<class U>  Deg(Unit<Math::Deg, U> value): Unit<Math::Deg, T>(value) {}

    
         Deg(Unit<Rad, T> value);
};

template<class T> class Rad: public Unit<Rad, T> {
    public:

        explicit Rad(T value): Unit<Math::Rad, T>(value) {}

        Rad(Unit<Math::Rad, T> value): Unit<Math::Rad, T>(value) {}

        template<class U>  explicit Rad(Unit<Math::Rad, U> value): Unit<Math::Rad, T>(value) {}

        Rad(Unit<Deg, T> value);
};

/*
 Rad<double> operator "" _rad(long double value) { return Rad<double>(value); }
 Rad<float> operator "" _radf(long double value) { return Rad<float>(value); }
*/

template<class T>  Deg<T>::Deg(Unit<Rad, T> value): Unit<Math::Deg, T>(T(180)*T(value)/Math::Constants<T>::pi()) {}
template<class T>  Rad<T>::Rad(Unit<Deg, T> value): Unit<Math::Rad, T>(T(value)*Math::Constants<T>::pi()/T(180)) {}

}}

#endif /* GVT_ANGLE_H */
