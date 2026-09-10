#ifndef ALIGNMENT_ML_INDEPENDENT_MEAN_ODE_INTEGRATOR_HPP
#define ALIGNMENT_ML_INDEPENDENT_MEAN_ODE_INTEGRATOR_HPP

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "FaserActsGeometry/FASERMagneticFieldWrapper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

/// Diagnostic-only independent integrator of the pinned ACTS mean ODE
///
///   dr/ds = T
///   dT/ds = (q/p) T × B(x)
///   d(q/p)/ds = 0
///
/// It is not EigenStepper, not GenericDefaultExtension, and is never
/// used for official h_i(θ).  B(x) and ∂B/∂x come from the official
/// FASERMagneticFieldWrapper.  Tolerances below are pre-registered
/// from geometry / loc0-invariance floors and must not be retuned
/// after seeing Jacobian agreement.
struct IndependentMeanOdeIntegrator {
  static constexpr const char* kMethod = "dormand_prince_5_4";
  static constexpr double kAbsTolPosMm = 1.0e-6;
  static constexpr double kAbsTolDir = 1.0e-10;
  static constexpr double kAbsTolQop = 1.0e-14;
  static constexpr double kRelTol = 1.0e-9;
  static constexpr double kInitialStepMm = 1.0;
  static constexpr double kMinStepMm = 1.0e-6;
  static constexpr double kMaxStepMm = 50.0;
  static constexpr double kMaxPathMm = 20000.0;
  static constexpr int kMaxSteps = 100000;
  static constexpr double kPlaneHitAbsMm = 1.0e-6;
  static constexpr const char* kToleranceJustification =
      "abs_tol_pos_mm equals the frozen loc0 invariance floor 1e-6 mm; "
      "direction/qop and rel_tol are one to five orders tighter than "
      "production stepTolerance 1e-4; max/min step follow the FASER "
      "magnet-cell scale (O(10 mm)) and the 1e-6 mm floor.  Registered "
      "before any 86 Jacobian comparison.  Do not retune from agreement.";

  struct State {
    Acts::Vector3 pos = Acts::Vector3::Zero();
    Acts::Vector3 dir = Acts::Vector3::UnitZ();
    double qop = 0.0;
  };

  struct Result {
    bool ok = false;
    State end;
    double pathLength = 0.0;
    int nSteps = 0;
    int nRejected = 0;
    int nFieldFails = 0;
    double nDotDir = 0.0;
    double distanceToPlaneMm = 0.0;
    std::string abortReason;
    Eigen::Matrix<double, 7, 7> jacobian = Eigen::Matrix<double, 7, 7>::Identity();
  };

  explicit IndependentMeanOdeIntegrator(
      const Acts::MagneticFieldContext& magFieldContext)
      : m_cache(m_wrapper.makeCache(magFieldContext)) {}

  bool field(const Acts::Vector3& pos, Acts::Vector3& B) {
    auto res = m_wrapper.getField(pos, m_cache);
    if (!res.ok()) {
      return false;
    }
    B = *res;
    return true;
  }

  bool fieldGradient(const Acts::Vector3& pos, Acts::Vector3& B,
                     Acts::ActsMatrix<3, 3>& G) {
    G.setZero();
    auto res = m_wrapper.getFieldGradient(pos, G, m_cache);
    if (!res.ok()) {
      return false;
    }
    B = *res;
    return true;
  }

  static Acts::ActsMatrix<3, 3> skew(const Acts::Vector3& a) {
    Acts::ActsMatrix<3, 3> m;
    m << 0., -a.z(), a.y(), a.z(), 0., -a.x(), -a.y(), a.x(), 0.;
    return m;
  }

  void meanRhs(const State& y, const Acts::Vector3& B, State& f) const {
    f.pos = y.dir;
    f.dir = y.qop * y.dir.cross(B);
    f.qop = 0.0;
  }

  void variationalRhs(const State& y, const Acts::Vector3& B,
                      const Acts::ActsMatrix<3, 3>& G,
                      const Eigen::Matrix<double, 7, 7>& J,
                      Eigen::Matrix<double, 7, 7>& dJ) const {
    dJ.setZero();
    dJ.block<3, 7>(0, 0) = J.block<3, 7>(3, 0);
    const Acts::ActsMatrix<3, 3> dT_dx = y.qop * skew(y.dir) * G;
    const Acts::ActsMatrix<3, 3> dT_dT = -y.qop * skew(B);
    const Acts::Vector3 dT_dL = y.dir.cross(B);
    dJ.block<3, 7>(3, 0) = dT_dx * J.block<3, 7>(0, 0) +
                           dT_dT * J.block<3, 7>(3, 0) +
                           dT_dL * J.block<1, 7>(6, 0);
  }

  static void axpyState(State& y, double a, const State& f) {
    y.pos += a * f.pos;
    y.dir += a * f.dir;
    y.qop += a * f.qop;
  }

  static State scaled(double a, const State& f) {
    State o;
    o.pos = a * f.pos;
    o.dir = a * f.dir;
    o.qop = a * f.qop;
    return o;
  }

  static State addScaled(const State& a, double c, const State& b) {
    State o;
    o.pos = a.pos + c * b.pos;
    o.dir = a.dir + c * b.dir;
    o.qop = a.qop + c * b.qop;
    return o;
  }

  Result integrateToPlane(State start, const Acts::Vector3& planeCenter,
                          const Acts::Vector3& planeNormal, double pathSign) {
    Result out;
    State y = start;
    if (y.dir.norm() > 0.0) {
      y.dir.normalize();
    }
    Eigen::Matrix<double, 7, 7> J = Eigen::Matrix<double, 7, 7>::Identity();
    double s = 0.0;
    double h = pathSign * kInitialStepMm;
    const auto distance = [&](const Acts::Vector3& pos) {
      return planeNormal.dot(pos - planeCenter);
    };
    double d0 = distance(y.pos);
    for (int step = 0; step < kMaxSteps; ++step) {
      if (std::abs(s) > kMaxPathMm) {
        out.abortReason = "max_path";
        return out;
      }
      const double dNow = distance(y.pos);
      if (std::abs(dNow) <= kPlaneHitAbsMm) {
        out.ok = true;
        out.end = y;
        out.pathLength = s;
        out.nSteps = step;
        out.nDotDir = planeNormal.dot(y.dir);
        out.distanceToPlaneMm = dNow;
        out.jacobian = J;
        return out;
      }
      const double denom = planeNormal.dot(y.dir);
      if (std::abs(denom) > 1.0e-12) {
        const double sHit = -dNow / denom;
        if (sHit * h > 0.0 && std::abs(sHit) < std::abs(h)) {
          h = sHit;
        }
      }
      if (std::abs(h) < kMinStepMm) {
        h = pathSign * kMinStepMm;
      }
      if (std::abs(h) > kMaxStepMm) {
        h = pathSign * kMaxStepMm;
      }

      State k1{}, k2{}, k3{}, k4{}, k5{}, k6{}, k7{};
      Eigen::Matrix<double, 7, 7> j1, j2, j3, j4, j5, j6, j7;
      Acts::Vector3 B;
      Acts::ActsMatrix<3, 3> G = Acts::ActsMatrix<3, 3>::Zero();
      const auto stage = [&](const State& ys, const Eigen::Matrix<double, 7, 7>& Js,
                             State& ks, Eigen::Matrix<double, 7, 7>& js) {
        if (!fieldGradient(ys.pos, B, G)) {
          return false;
        }
        meanRhs(ys, B, ks);
        variationalRhs(ys, B, G, Js, js);
        return true;
      };
      if (!stage(y, J, k1, j1)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      State y2 = addScaled(y, h * (1.0 / 5.0), k1);
      auto J2 = J + (h * (1.0 / 5.0)) * j1;
      if (!stage(y2, J2, k2, j2)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      State y3 = y;
      axpyState(y3, h * (3.0 / 40.0), k1);
      axpyState(y3, h * (9.0 / 40.0), k2);
      auto J3 = J + (h * (3.0 / 40.0)) * j1 + (h * (9.0 / 40.0)) * j2;
      if (!stage(y3, J3, k3, j3)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      State y4 = y;
      axpyState(y4, h * (44.0 / 45.0), k1);
      axpyState(y4, h * (-56.0 / 15.0), k2);
      axpyState(y4, h * (32.0 / 9.0), k3);
      auto J4 = J + (h * (44.0 / 45.0)) * j1 + (h * (-56.0 / 15.0)) * j2 +
                (h * (32.0 / 9.0)) * j3;
      if (!stage(y4, J4, k4, j4)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      State y5 = y;
      axpyState(y5, h * (19372.0 / 6561.0), k1);
      axpyState(y5, h * (-25360.0 / 2187.0), k2);
      axpyState(y5, h * (64448.0 / 6561.0), k3);
      axpyState(y5, h * (-212.0 / 729.0), k4);
      auto J5 = J + (h * (19372.0 / 6561.0)) * j1 +
                (h * (-25360.0 / 2187.0)) * j2 +
                (h * (64448.0 / 6561.0)) * j3 + (h * (-212.0 / 729.0)) * j4;
      if (!stage(y5, J5, k5, j5)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      State y6 = y;
      axpyState(y6, h * (9017.0 / 3168.0), k1);
      axpyState(y6, h * (-355.0 / 33.0), k2);
      axpyState(y6, h * (46732.0 / 5247.0), k3);
      axpyState(y6, h * (49.0 / 176.0), k4);
      axpyState(y6, h * (-5103.0 / 18656.0), k5);
      auto J6 = J + (h * (9017.0 / 3168.0)) * j1 + (h * (-355.0 / 33.0)) * j2 +
                (h * (46732.0 / 5247.0)) * j3 + (h * (49.0 / 176.0)) * j4 +
                (h * (-5103.0 / 18656.0)) * j5;
      if (!stage(y6, J6, k6, j6)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      State y5th = y;
      axpyState(y5th, h * (35.0 / 384.0), k1);
      axpyState(y5th, h * (500.0 / 1113.0), k3);
      axpyState(y5th, h * (125.0 / 192.0), k4);
      axpyState(y5th, h * (-2187.0 / 6784.0), k5);
      axpyState(y5th, h * (11.0 / 84.0), k6);
      auto J5th = J + (h * (35.0 / 384.0)) * j1 + (h * (500.0 / 1113.0)) * j3 +
                  (h * (125.0 / 192.0)) * j4 + (h * (-2187.0 / 6784.0)) * j5 +
                  (h * (11.0 / 84.0)) * j6;
      if (!stage(y5th, J5th, k7, j7)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      State y4th = y;
      axpyState(y4th, h * (5179.0 / 57600.0), k1);
      axpyState(y4th, h * (7571.0 / 16695.0), k3);
      axpyState(y4th, h * (393.0 / 640.0), k4);
      axpyState(y4th, h * (-92097.0 / 339200.0), k5);
      axpyState(y4th, h * (187.0 / 2100.0), k6);
      axpyState(y4th, h * (1.0 / 40.0), k7);

      const double errPos = (y5th.pos - y4th.pos).norm();
      const double errDir = (y5th.dir - y4th.dir).norm();
      const double errQ = std::abs(y5th.qop - y4th.qop);
      const double scalePos =
          kAbsTolPosMm + kRelTol * std::max(y.pos.norm(), y5th.pos.norm());
      const double scaleDir = kAbsTolDir + kRelTol * 1.0;
      const double scaleQ =
          kAbsTolQop + kRelTol * std::max(std::abs(y.qop), std::abs(y5th.qop));
      const double err = std::max({errPos / scalePos, errDir / scaleDir,
                                   errQ / std::max(scaleQ, 1.0e-18)});
      if (err > 1.0 && std::abs(h) > kMinStepMm) {
        h *= std::max(0.2, 0.9 * std::pow(err, -0.2));
        ++out.nRejected;
        continue;
      }
      y = y5th;
      J = J5th;
      const double nrm = y.dir.norm();
      if (nrm > 0.0) {
        y.dir /= nrm;
        const Acts::ActsMatrix<3, 3> proj =
            Acts::ActsMatrix<3, 3>::Identity() - y.dir * y.dir.transpose();
        J.block<3, 7>(3, 0) = proj * J.block<3, 7>(3, 0);
      }
      s += h;
      const double dNew = distance(y.pos);
      if (d0 * dNew <= 0.0 || std::abs(dNew) <= kPlaneHitAbsMm) {
        const double frac = (std::abs(dNew - dNow) > 1.0e-18)
                                ? (-dNow / (dNew - dNow))
                                : 1.0;
        const double alpha = std::clamp(frac, 0.0, 1.0);
        y.pos = (1.0 - alpha) * (y.pos - h * y.dir) + alpha * y.pos;
        // last accepted dir/qop already at this step; snap onto plane
        const double dSnap = distance(y.pos);
        if (std::abs(denom) > 1.0e-12) {
          y.pos -= (dSnap / denom) * y.dir;
        }
        out.ok = true;
        out.end = y;
        out.pathLength = s - h + alpha * h;
        out.nSteps = step + 1;
        out.nDotDir = planeNormal.dot(y.dir);
        out.distanceToPlaneMm = distance(y.pos);
        out.jacobian = J;
        return out;
      }
      d0 = dNew;
      double fac = (err > 0.0) ? 0.9 * std::pow(err, -0.2) : 4.0;
      fac = std::clamp(fac, 0.2, 4.0);
      h *= fac;
    }
    out.abortReason = "max_steps";
    out.end = y;
    out.pathLength = s;
    out.nSteps = kMaxSteps;
    out.jacobian = J;
    return out;
  }

 private:
  FASERMagneticFieldWrapper m_wrapper;
  Acts::MagneticFieldProvider::Cache m_cache;
};

#endif
