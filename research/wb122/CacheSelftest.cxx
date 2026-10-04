#include "Trace.h"
#include <cassert>
int main(){
  // Construct actual cache with value-initialized map pointer; never query a map.
  MagField::FaserFieldCache c{};WB122Trace::expectedMap=nullptr;
  const Acts::Vector3 p(0.5,0.5,0.5);
  auto a=WB122Trace::cacheState(c,p);assert(!a["valid_cell"].get<bool>());assert(a["float_corner_fields"].is_null());
  // Synthetic fixture uses public setters only. Reader never writes.
  auto& cell=c.*member(WB122Trace::Cell{});
  cell.setRange(0,1,0,2,0,4);cell.setBscale(0.25);
  for(int i=0;i<8;++i)cell.setField(i,BFieldVector<double>(i,i+1,i+2));
  auto b=WB122Trace::cacheState(c,p);assert(b["valid_cell"].get<bool>()&&b["contains_query"].get<bool>());
  assert(b["float_inverse_widths"]==WB122Trace::Json::array({1.,.5,.25}));
  assert(b["float_corner_fields"][2][7]==9.);assert(b["float_bscale_kT"]==.25);
  double before[3],after[3];cell.getB(p.data(),before);auto repeated=WB122Trace::cacheState(c,p);cell.getB(p.data(),after);
  assert(b==repeated);for(int i=0;i<3;++i)assert(before[i]==after[i]);
  assert(WB122Trace::cacheState(c,Acts::Vector3(1,2,4))["contains_query"].get<bool>());
  assert(!WB122Trace::cacheState(c,Acts::Vector3(1.001,2,4))["contains_query"].get<bool>());
  cell.invalidate();assert(WB122Trace::cacheState(c,p)["float_corner_fields"].is_null());
  std::cout<<WB122Trace::Json{{"status","PASS"},{"actual_headers",true},{"reader_nonmutation",true},{"initial_invalid_float_fields_not_read",true},{"closed_boundary",true}}.dump()<<std::endl;
}
