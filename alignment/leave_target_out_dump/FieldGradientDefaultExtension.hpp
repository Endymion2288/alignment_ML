#ifndef ALIGNMENT_ML_FIELD_GRADIENT_DEFAULT_EXTENSION_HPP
#define ALIGNMENT_ML_FIELD_GRADIENT_DEFAULT_EXTENSION_HPP

#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/Utilities/VectorHelpers.hpp"
#include "FaserActsGeometry/FASERMagneticFieldWrapper.h"

#include <array>

/// Diagnostic-only RKN4 extension.  Mean k_i is identical to ACTS 32.0.2
/// GenericDefaultExtension.  transportMatrix completes the same mean map
///   dT/ds = (q/p) T × B(x)
/// by including official FASERMagneticFieldWrapper::getFieldGradient.
/// It does not change energy-loss, process noise, or the official
/// EigenStepper<> used for h_i(theta).
struct FieldGradientDefaultExtension {
  using Scalar = double;
  using ThisVector3 = Eigen::Matrix<Scalar, 3, 1>;

  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  int bid(const propagator_state_t&, const stepper_t&,
          const navigator_t&) const {
    return 1;
  }

  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  bool k(const propagator_state_t& state, const stepper_t& stepper,
         const navigator_t&, ThisVector3& knew, const Acts::Vector3& bField,
         std::array<Scalar, 4>& kQoP, const int i = 0, const double h = 0.,
         const ThisVector3& kprev = ThisVector3::Zero()) {
    auto qop = stepper.qOverP(state.stepping);
    if (i == 0) {
      knew = qop * stepper.direction(state.stepping).cross(bField);
      kQoP = {0., 0., 0., 0.};
    } else {
      knew = qop * (stepper.direction(state.stepping) + h * kprev).cross(bField);
    }
    return true;
  }

  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  bool finalize(propagator_state_t& state, const stepper_t& stepper,
                const navigator_t& navigator, const double h) const {
    propagateTime(state, stepper, navigator, h);
    return true;
  }

  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  bool finalize(propagator_state_t& state, const stepper_t& stepper,
                const navigator_t& navigator, const double h,
                Acts::FreeMatrix& D) const {
    propagateTime(state, stepper, navigator, h);
    return transportMatrix(state, stepper, navigator, h, D);
  }

 private:
  static Acts::ActsMatrix<3, 3> skew(const Acts::Vector3& a) {
    Acts::ActsMatrix<3, 3> m;
    m << 0., -a.z(), a.y(), a.z(), 0., -a.x(), -a.y(), a.x(), 0.;
    return m;
  }

  template <typename propagator_state_t>
  static Acts::ActsMatrix<3, 3> officialGradient(
      const Acts::Vector3& position, propagator_state_t& state, bool& ok) {
    FASERMagneticFieldWrapper wrapper;
    Acts::ActsMatrix<3, 3> gradient = Acts::ActsMatrix<3, 3>::Zero();
    auto field = wrapper.getFieldGradient(position, gradient,
                                         state.stepping.fieldCache);
    if (!field.ok()) {
      ok = false;
      return Acts::ActsMatrix<3, 3>::Zero();
    }
    return gradient;
  }

  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  void propagateTime(propagator_state_t& state, const stepper_t& stepper,
                     const navigator_t&, const double h) const {
    using std::hypot;
    auto m = stepper.particleHypothesis(state.stepping).mass();
    auto p = stepper.absoluteMomentum(state.stepping);
    auto dtds = hypot(1, m / p);
    state.stepping.pars[Acts::eFreeTime] += h * dtds;
    if (state.stepping.covTransport) {
      state.stepping.derivative(3) = dtds;
    }
  }

  /// Tangent of the ACTS 32.0.2 mean RKN4 map, including ∂B/∂x.
  /// Stage positions match EigenStepper.ipp.  If every G=0 this reduces
  /// to GenericDefaultExtension::transportMatrix.
  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  bool transportMatrix(propagator_state_t& state, const stepper_t& stepper,
                       const navigator_t&, const double h,
                       Acts::FreeMatrix& D) const {
    using std::hypot;
    auto m = state.stepping.particleHypothesis.mass();
    auto& sd = state.stepping.stepData;
    auto pos = stepper.position(state.stepping);
    auto dir = stepper.direction(state.stepping);
    auto qop = stepper.qOverP(state.stepping);
    auto p = stepper.absoluteMomentum(state.stepping);
    auto dtds = hypot(1, m / p);
    const double half_h = h * 0.5;
    const double h2 = h * h;
    const Acts::Vector3 pos1 = pos + half_h * dir + h2 * 0.125 * sd.k1;
    const Acts::Vector3 pos2 = pos + h * dir + h2 * 0.5 * sd.k3;
    const Acts::Vector3 t2 = dir + half_h * sd.k1;
    const Acts::Vector3 t3 = dir + half_h * sd.k2;
    const Acts::Vector3 t4 = dir + h * sd.k3;

    bool gradientOk = true;
    const Acts::ActsMatrix<3, 3> g0 = officialGradient(pos, state, gradientOk);
    const Acts::ActsMatrix<3, 3> g1 = officialGradient(pos1, state, gradientOk);
    const Acts::ActsMatrix<3, 3> g2 = officialGradient(pos2, state, gradientOk);
    (void)gradientOk;

    Acts::ActsMatrix<3, 3> dk1dx = qop * skew(dir) * g0;
    Acts::ActsMatrix<3, 3> dk1dT = -qop * skew(sd.B_first);
    Acts::Vector3 dk1dL = dir.cross(sd.B_first);

    const Acts::ActsMatrix<3, 3> dpos1dx =
        Acts::ActsMatrix<3, 3>::Identity() + (h2 * 0.125) * dk1dx;
    const Acts::ActsMatrix<3, 3> dpos1dT =
        half_h * Acts::ActsMatrix<3, 3>::Identity() + (h2 * 0.125) * dk1dT;
    const Acts::Vector3 dpos1dL = (h2 * 0.125) * dk1dL;
    const Acts::ActsMatrix<3, 3> dt2dx = half_h * dk1dx;
    const Acts::ActsMatrix<3, 3> dt2dT =
        Acts::ActsMatrix<3, 3>::Identity() + half_h * dk1dT;
    const Acts::Vector3 dt2dL = half_h * dk1dL;

    const Acts::ActsMatrix<3, 3> extra2 = qop * skew(t2) * g1;
    Acts::ActsMatrix<3, 3> dk2dx =
        qop * Acts::VectorHelpers::cross(dt2dx, sd.B_middle) + extra2 * dpos1dx;
    Acts::ActsMatrix<3, 3> dk2dT =
        qop * Acts::VectorHelpers::cross(dt2dT, sd.B_middle) + extra2 * dpos1dT;
    Acts::Vector3 dk2dL =
        t2.cross(sd.B_middle) + qop * dt2dL.cross(sd.B_middle) + extra2 * dpos1dL;

    const Acts::ActsMatrix<3, 3> dt3dx = half_h * dk2dx;
    const Acts::ActsMatrix<3, 3> dt3dT =
        Acts::ActsMatrix<3, 3>::Identity() + half_h * dk2dT;
    const Acts::Vector3 dt3dL = half_h * dk2dL;
    const Acts::ActsMatrix<3, 3> extra3 = qop * skew(t3) * g1;
    Acts::ActsMatrix<3, 3> dk3dx =
        qop * Acts::VectorHelpers::cross(dt3dx, sd.B_middle) + extra3 * dpos1dx;
    Acts::ActsMatrix<3, 3> dk3dT =
        qop * Acts::VectorHelpers::cross(dt3dT, sd.B_middle) + extra3 * dpos1dT;
    Acts::Vector3 dk3dL =
        t3.cross(sd.B_middle) + qop * half_h * dk2dL.cross(sd.B_middle) +
        extra3 * dpos1dL;

    const Acts::ActsMatrix<3, 3> dpos2dx =
        Acts::ActsMatrix<3, 3>::Identity() + (h2 * 0.5) * dk3dx;
    const Acts::ActsMatrix<3, 3> dpos2dT =
        h * Acts::ActsMatrix<3, 3>::Identity() + (h2 * 0.5) * dk3dT;
    const Acts::Vector3 dpos2dL = (h2 * 0.5) * dk3dL;
    const Acts::ActsMatrix<3, 3> dt4dx = h * dk3dx;
    const Acts::ActsMatrix<3, 3> dt4dT =
        Acts::ActsMatrix<3, 3>::Identity() + h * dk3dT;
    const Acts::Vector3 dt4dL = h * dk3dL;
    const Acts::ActsMatrix<3, 3> extra4 = qop * skew(t4) * g2;
    Acts::ActsMatrix<3, 3> dk4dx =
        qop * Acts::VectorHelpers::cross(dt4dx, sd.B_last) + extra4 * dpos2dx;
    Acts::ActsMatrix<3, 3> dk4dT =
        qop * Acts::VectorHelpers::cross(dt4dT, sd.B_last) + extra4 * dpos2dT;
    Acts::Vector3 dk4dL =
        t4.cross(sd.B_last) + qop * h * dk3dL.cross(sd.B_last) + extra4 * dpos2dL;

    D = Acts::FreeMatrix::Identity();
    auto dFdT = D.block<3, 3>(0, 4);
    auto dFdL = D.block<3, 1>(0, 7);
    auto dFdx = D.block<3, 3>(0, 0);
    auto dGdT = D.block<3, 3>(4, 4);
    auto dGdL = D.block<3, 1>(4, 7);
    auto dGdx = D.block<3, 3>(4, 0);

    dFdx = Acts::ActsMatrix<3, 3>::Identity() +
           (h2 / 6.) * (dk1dx + dk2dx + dk3dx);
    dFdT = h * Acts::ActsMatrix<3, 3>::Identity() +
           (h2 / 6.) * (dk1dT + dk2dT + dk3dT);
    dFdL = (h2 / 6.) * (dk1dL + dk2dL + dk3dL);
    dGdx = (h / 6.) * (dk1dx + 2. * (dk2dx + dk3dx) + dk4dx);
    dGdT = Acts::ActsMatrix<3, 3>::Identity() +
           (h / 6.) * (dk1dT + 2. * (dk2dT + dk3dT) + dk4dT);
    dGdL = (h / 6.) * (dk1dL + 2. * (dk2dL + dk3dL) + dk4dL);
    D(3, 7) = h * m * m * qop / dtds;
    return true;
  }
};

#endif
