#include "../research/wb94/ReferenceRK4.h"
#include <iostream>
#include <limits>
int main() {
  const auto c=WB94Reference::controls(0.000299792458);
  if(c[0]>1e-8 || c[1]<=1e-8 || c[2]<=1e-8)return 1;
  bool rejected=false;
  try {WB94Reference::integrate(WB94Reference::State{0,0,0,0,0},0,1,.5,.01,0,
    [](double,double,double){return WB94Reference::Field{std::numeric_limits<double>::quiet_NaN(),0,0};});}
  catch(const std::runtime_error&){rejected=true;}
  if(!rejected)return 2;
  std::cout<<"RK affine/analytic helix/sign/units/nonfinite controls PASS: "<<c[0]<<" "<<c[1]<<" "<<c[2]<<"\n";
}
