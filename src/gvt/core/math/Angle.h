
#ifndef GVT_CORE_MATH_ANGLE_H
#define GVT_CORE_MATH_ANGLE_H

#include <gvt/core/math/Constants.h>
#include <gvt/core/math/Unit.h>


namespace gvt { 
    namespace core {
        namespace math {
            template<class T> class Deg;
            template<class T> class Rad;

            template<class T> class Deg: public Unit<Deg, T> 
            {
            public:
               Deg() {}

               explicit Deg(T value): Unit<math::Deg, T>(value) {}

               Deg(Unit<math::Deg, T> value): Unit<math::Deg, T>(value) {}

               template<class U> Deg(Unit<math::Deg, U> value): Unit<math::Deg, T>(value) {}

               Deg(Unit<Rad, T> value);
           };

           template<class T> class Rad: 
           public Unit<Rad, T> 
           {
           public:

                explicit Rad(T value): Unit<math::Rad, T>(value) {}

                Rad(Unit<math::Rad, T> value): Unit<math::Rad, T>(value) {}

                template<class U>  explicit Rad(Unit<math::Rad, U> value): Unit<math::Rad, T>(value) {}

                Rad(Unit<Deg, T> value);
           };

           template<class T>  Deg<T>::Deg(Unit<Rad, T> value): Unit<math::Deg, T>(T(180)*T(value)/Constants<T>::pi()) {}
           template<class T>  Rad<T>::Rad(Unit<Deg, T> value): Unit<math::Rad, T>(T(value)*Constants<T>::pi()/T(180)) {}
        }
    }
}
#endif /* GVT_CORE_MATH_ANGLE_H */
