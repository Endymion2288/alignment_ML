#include "../research/wb95/PrecisionControl.h"
#include "../research/wb94/ReferenceRK4.h"
#include "../research/wb95/CompensatedRK4.h"
#include <iostream>
int main() {
  WB95Math::Corners nodes;
  for(int c=0;c<8;++c){double x=(c>>2)&1,y=(c>>1)&1,z=c&1;nodes[c]={x+2*y+3*z,4*x-y+z,7.};}
  const auto value=WB95Math::trilinear(nodes,{.2,.3,.4},2);
  if(std::abs(value[0]-4)>1e-12 || std::abs(value[1]-1.8)>1e-12 || value[2]!=14)return 1;
  auto swapped=WB95Math::trilinear(nodes,{.4,.3,.2},2);
  if(std::abs(swapped[0]-value[0])<.1)return 2;
  auto wrongScale=WB95Math::trilinear(nodes,{.2,.3,.4},2000);
  if(std::abs(wrongScale[0]-value[0])<1)return 3;
  bool reject=false;try{WB95Math::trilinear(nodes,{-1,.3,.4},2);}catch(const std::runtime_error&){reject=true;}
  if(!reject)return 4;
  const auto rk=WB94Reference::controls(.000299792458);
  if(rk[0]>1e-8 || rk[1]<=1e-8 || rk[2]<=1e-8)return 5;
  // Exact affine result remains unchanged by additional integration cuts.
  auto zero=[](double,double,double){return WB94Reference::Field{0,0,0};};
  WB94Reference::State s{0,0,.01,.02,0};double from=0;
  for(double to:{5.,10.,100.}){s=WB95Reference::integrate(s,from,to,.03125,.01,0,zero);from=to;}
  if(std::abs(s[0]-1)>1e-11 || std::abs(s[1]-2)>1e-11)return 6;
  auto by=[](double,double,double){return WB94Reference::Field{0,.000299792458,0};};
  const double w=.01*.000299792458,z=1000.,c=std::sqrt(1-w*w*z*z);
  for(double dz:{.5,.25,.125,.0625,.03125}) {
    const auto h=WB95Reference::integrate(WB94Reference::State{0,0,0,0,0},0,z,dz,.01,0,by);
    if(std::abs(h[0]+w*z*z/(1+c))>1e-8 || std::abs(h[2]+w*z/c)/.001>1e-8)return 7;
  }
  std::cout<<"WB95 trilinear/coordinate/scale/domain/helix/segmentation controls PASS\n";
}
