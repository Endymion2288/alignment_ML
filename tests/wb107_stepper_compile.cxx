// Compile-only compatibility probe. Does not construct or propagate a track.
#include "research/wb107/BoundTrace.h"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
using DiagnosticPropagator=Acts::Propagator<WB107Trace::Stepper,Acts::Navigator>;
using Options=Acts::PropagatorOptions<Acts::ActionList<Acts::MaterialInteractor>,
    Acts::AbortList<Acts::EndOfWorldReached>>;
auto instantiateTargetOverload(const DiagnosticPropagator& p,
    const Acts::BoundTrackParameters& start,const Acts::Surface& target,const Options& options) {
  return p.propagate(start,target,options);
}
