/* ======================================================================================= 
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin  
   All rights reserved.
                                                                                           
   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file     
   except in compliance with the License.                                                  
   A copy of the License is included with this software in the file LICENSE.               
   If your copy does not contain the License, you may obtain a copy of the License at:     
                                                                                           
       http://opensource.org/licenses/BSD-3-Clause                                         
                                                                                           
   Unless required by applicable law or agreed to in writing, software distributed under   
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
   KIND, either express or implied.                                                        
   See the License for the specific language governing permissions and limitations under   
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards ACI-1339863, 
   ACI-1339881 and ACI-1339840
   ======================================================================================= */

#include <gvt/core/Variant.h>

using namespace gvt::core;
using namespace gvt::core::math;

Variant::Variant(int i): coreData(i) {}
Variant::Variant(long l): coreData(l) {}
Variant::Variant(float f): coreData(f) {}
Variant::Variant(double d): coreData(d) {}
Variant::Variant(bool b): coreData(b) {}
Variant::Variant(String s): coreData(s) {}
Variant::Variant(Uuid u): coreData(u) {}
Variant::Variant(Vector3f v): coreData(v) {}
Variant::Variant(Vector4f v): coreData(v) {}
Variant::Variant(Point4f p): coreData(p) {}
Variant::Variant(AffineTransformMatrix<float>* atm): coreData(atm) {}
Variant::Variant(Matrix3f* m): coreData(m) {}


int Variant::toInt()
{ 
   return boost::get<int>(coreData); 
}

long Variant::toLong() 
{ 
   return boost::get<long>(coreData); 
}

float Variant::toFloat() 
{ 
   return boost::get<float>(coreData); 
}

double Variant::toDouble() 
{ 
   return boost::get<double>(coreData); 
}

bool Variant::toBool() 
{ 
   return boost::get<bool>(coreData); 
}

String Variant::toString() 
{ 
   return boost::get<String>(coreData); 
}

Uuid Variant::toUuid() 
{ 
   return boost::get<Uuid>(coreData); 
}

Vector3f toVector3f() 
{ 
   return boost::get<Vector3f>(coreData); 
}

Vector4f toVector4f() 
{ 
    return boost::get<Vector4f>(coreData); 
}

Point4f toPoint4f() 
{ 
   return boost::get<Point4f>(coreData); 
}

AffineTransformMatrix<float>* toAffineTransformMatrixFloat() 
{ 
   return boost::get<AffineTransformMatrix<float>*>(coreData); 
}

Matrix3f* toMatrix3f() 
{ 
   return boost::get<Matrix3f*>(coreData); 
}


bool operator==(const Variant& v) const
{
   return coreData == v.coreData;
}

bool operator!=(const Variant& v) const
{
   return coreData != v.coreData;
}

bool operator>(const Variant& v) const
{
   return coreData > v.coreData;
}

bool operator>=(const Variant& v) const
{
   return coreData >= v.coreData;
}

bool operator<(const Variant& v) const
{
   return coreData < v.coreData;
}

bool operator<=(const Variant& v) const
{
   return coreData <= v.coreData;
}

