#ifndef ALIGNMENT_ML_COMMON_GRID_SHADOW_INTEGRATOR_HPP
#define ALIGNMENT_ML_COMMON_GRID_SHADOW_INTEGRATOR_HPP

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/PdgParticle.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/ParticleHypothesis.hpp"
#include "Acts/Material/Interactions.hpp"
#include "Acts/Material/Material.hpp"
#include "Acts/Material/MaterialSlab.hpp"
#include "FaserActsGeometry/FASERMagneticFieldWrapper.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

/// Diagnostic-only common-grid shadow of the pinned production mean map.
///
/// Field-only subsegments:
///   dr/ds = T
///   dT/ds = (q/p) T × B(x)
///   d(q/p)/ds = 0
///
/// At frozen surface-material nodes the official mean energy-loss map
///   computeEnergyLossMean (Bethe + radiative)
/// is applied.  Multiple scattering / process noise is not applied.
///
/// The mesh is built from the nominal unperturbed hop only.  Every
/// ±h…±h/8 arm reuses that frozen node sequence, the same material
/// partition, and the same classical RK4 stages.  Not EigenStepper.
/// Tolerances and grid sizes are pre-registered and must not be
/// retuned after seeing Jacobian agreement.
struct CommonGridShadowIntegrator {
  static constexpr const char* kMethod = "classical_rk4_common_grid";
  static constexpr double kCoarseMaxStepMm = 20.0;
  static constexpr double kNominalMaxStepMm = 10.0;
  static constexpr double kFineMaxStepMm = 5.0;
  static constexpr double kMaxPathMm = 20000.0;
  static constexpr int kMaxNodes = 20000;
  static constexpr double kPlaneHitAbsMm = 1.0e-6;
  static constexpr double kMeanLoc0AbsMm = 1.0e-3;
  static constexpr double kMeanPathAbsMm = 1.0e-2;
  static constexpr double kMeanPosAbsMm = 1.0e-2;
  static constexpr double kMeanDirAbs = 1.0e-6;
  static constexpr double kMeanQopRel = 1.0e-4;
  static constexpr double kGridDerivRelMax = 0.05;
  static constexpr double kGridMeanLoc0AbsMm = 1.0e-3;
  static constexpr const char* kMeshJustification =
      "nominal_max_step_mm = 10 mm is one FASER magnet-cell scale; "
      "coarse/fine are exact 2x/0.5x of that same rule.  Mean-closure "
      "loc0 1e-3 mm is 23x below SCT sigma (0.08/sqrt(12)) and 44x "
      "tighter than the WB124 vacuum residual (0.044 mm), so a vacuum "
      "ODE cannot pass the mean contract.  Registered before any 86 "
      "Jacobian comparison.  Do not retune from agreement.";

  struct State {
    Acts::Vector3 pos = Acts::Vector3::Zero();
    Acts::Vector3 dir = Acts::Vector3::UnitZ();
    double qop = 0.0;
  };

  struct Node {
    double ds = 0.0;
    bool material = false;
    bool volumeOnly = false;
    Acts::MaterialSlab slab;
    double sAbs = 0.0;
  };

  struct Mesh {
    std::string name;
    double maxStepMm = 0.0;
    std::vector<Node> nodes;
    int nField = 0;
    int nMaterial = 0;
    int nVolumeRecorded = 0;
  };

  struct Result {
    bool ok = false;
    State end;
    double pathLength = 0.0;
    int nFieldSteps = 0;
    int nMaterialMaps = 0;
    int nFieldFails = 0;
    double nDotDir = 0.0;
    double distanceToPlaneMm = 0.0;
    std::string abortReason;
  };

  explicit CommonGridShadowIntegrator(
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

  void meanRhs(const State& y, const Acts::Vector3& B, State& f) const {
    f.pos = y.dir;
    f.dir = y.qop * y.dir.cross(B);
    f.qop = 0.0;
  }

  static State addScaled(const State& a, double c, const State& b) {
    State o;
    o.pos = a.pos + c * b.pos;
    o.dir = a.dir + c * b.dir;
    o.qop = a.qop + c * b.qop;
    return o;
  }

  static void applyEnergyLossMean(State& y, const Acts::MaterialSlab& slab,
                                  double pathSign) {
    if (!slab) {
      return;
    }
    const float mass = static_cast<float>(Acts::ParticleHypothesis::muon().mass());
    const float absQ = 1.0f;
    const float qop = static_cast<float>(y.qop);
    const float eloss = Acts::computeEnergyLossMean(
        slab, Acts::PdgParticle::eMuon, mass, qop, absQ);
    const double navDir = pathSign >= 0.0 ? 1.0 : -1.0;
    const double p = (std::abs(y.qop) > 0.0) ? (1.0 / std::abs(y.qop)) : 0.0;
    const double nextE = std::hypot(static_cast<double>(mass), p) - eloss * navDir;
    double nextP = (mass < nextE)
                       ? std::sqrt(nextE * nextE - static_cast<double>(mass) * mass)
                       : 0.0;
    nextP = std::max(10.0 * Acts::UnitConstants::MeV, nextP);
    y.qop = std::copysign(1.0 / nextP, y.qop);
  }

  static Mesh buildMesh(double pathSign, double pathLength,
                        const std::vector<std::pair<double, Node>>& crossings,
                        double maxStepMm, const std::string& name) {
    Mesh mesh;
    mesh.name = name;
    mesh.maxStepMm = maxStepMm;
    const double sEnd = std::min(std::abs(pathLength), kMaxPathMm);
    const double sign = pathSign >= 0.0 ? 1.0 : -1.0;
    std::vector<std::pair<double, Node>> events = crossings;
    std::sort(events.begin(), events.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    double s = 0.0;
    std::size_t ie = 0;
    int guard = 0;
    while (s < sEnd - 1.0e-12 && guard < kMaxNodes) {
      ++guard;
      double nextFill = std::min(s + maxStepMm, sEnd);
      double nextEvent = sEnd;
      if (ie < events.size()) {
        nextEvent = std::min(events[ie].first, sEnd);
      }
      if (ie < events.size() && events[ie].first <= s + 1.0e-9) {
        Node mat = events[ie].second;
        mat.ds = 0.0;
        mat.sAbs = s;
        mesh.nodes.push_back(mat);
        if (mat.volumeOnly) {
          ++mesh.nVolumeRecorded;
        } else {
          ++mesh.nMaterial;
        }
        ++ie;
        continue;
      }
      const double target = std::min(nextFill, nextEvent);
      Node field;
      field.ds = sign * (target - s);
      field.material = false;
      field.sAbs = target;
      mesh.nodes.push_back(field);
      ++mesh.nField;
      s = target;
    }
    return mesh;
  }

  Result integrate(State start, const Mesh& mesh, const Acts::Vector3& planeCenter,
                   const Acts::Vector3& planeNormal, double pathSign) {
    Result out;
    State y = start;
    if (y.dir.norm() > 0.0) {
      y.dir.normalize();
    }
    double s = 0.0;
    const auto distance = [&](const Acts::Vector3& pos) {
      return planeNormal.dot(pos - planeCenter);
    };
    for (const Node& node : mesh.nodes) {
      if (std::abs(s) > kMaxPathMm) {
        out.abortReason = "max_path";
        return out;
      }
      if (node.material) {
        if (!node.volumeOnly) {
          applyEnergyLossMean(y, node.slab, pathSign);
          ++out.nMaterialMaps;
        }
        continue;
      }
      const double h = node.ds;
      if (std::abs(h) < 1.0e-18) {
        continue;
      }
      State k1{}, k2{}, k3{}, k4{};
      Acts::Vector3 B = Acts::Vector3::Zero();
      if (!field(y.pos, B)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      meanRhs(y, B, k1);
      State y2 = addScaled(y, 0.5 * h, k1);
      if (!field(y2.pos, B)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      meanRhs(y2, B, k2);
      State y3 = addScaled(y, 0.5 * h, k2);
      if (!field(y3.pos, B)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      meanRhs(y3, B, k3);
      State y4 = addScaled(y, h, k3);
      if (!field(y4.pos, B)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      meanRhs(y4, B, k4);
      y.pos += (h / 6.0) * (k1.pos + 2.0 * k2.pos + 2.0 * k3.pos + k4.pos);
      y.dir += (h / 6.0) * (k1.dir + 2.0 * k2.dir + 2.0 * k3.dir + k4.dir);
      y.qop += (h / 6.0) * (k1.qop + 2.0 * k2.qop + 2.0 * k3.qop + k4.qop);
      if (y.dir.norm() > 0.0) {
        y.dir.normalize();
      }
      s += h;
      ++out.nFieldSteps;
    }
    const double dNow = distance(y.pos);
    const double denom = planeNormal.dot(y.dir);
    if (std::abs(dNow) > kPlaneHitAbsMm && std::abs(denom) > 1.0e-12) {
      y.pos -= (dNow / denom) * y.dir;
    }
    out.ok = true;
    out.end = y;
    out.pathLength = s;
    out.nDotDir = planeNormal.dot(y.dir);
    out.distanceToPlaneMm = distance(y.pos);
    return out;
  }

 private:
  FASERMagneticFieldWrapper m_wrapper;
  Acts::MagneticFieldProvider::Cache m_cache;
};

#endif
