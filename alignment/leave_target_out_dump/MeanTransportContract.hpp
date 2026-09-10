#ifndef ALIGNMENT_ML_MEAN_TRANSPORT_CONTRACT_HPP
#define ALIGNMENT_ML_MEAN_TRANSPORT_CONTRACT_HPP

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Common.hpp"
#include "Acts/Definitions/PdgParticle.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/ParticleHypothesis.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Material/ISurfaceMaterial.hpp"
#include "Acts/Material/Interactions.hpp"
#include "Acts/Material/Material.hpp"
#include "Acts/Material/MaterialSlab.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/detail/PointwiseMaterialInteraction.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "FaserActsGeometry/FASERMagneticFieldWrapper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

/// WB126/WB127 mean-only shadow of the pinned ACTS 32.0.2 EigenStepper mean.
///
/// Shadow P reproduces production physics semantics:
///   RKN4 / Nyström field update from EigenStepper.ipp
///   k_i from GenericDefaultExtension.hpp
///   explicit direction.normalize() after each accepted field step
///   surface energy loss after arrival, q/p only
///
/// WB127: production MaterialInteractor does NOT use
/// computeEnergyLossMean or computeEnergyLossMode.  Compiled
/// PointwiseMaterialInteraction::evaluatePointwiseMaterialInteraction
/// (libActsCore.so v32.0.2) writes Eloss from computeEnergyLossBethe
/// only.  Mean = Bethe + Radiative; Mode = 0.9*Landau + 0.15*Radiative.
/// updateState runs iff evaluateMaterialSlab returns a valid slab
/// (start=PostUpdate, target=PreUpdate, else FullUpdate; ISurfaceMaterial
/// factor may zero the slab).  Shadow copies that gating, Bethe Eloss,
/// and the updateState p/E -> q/p conversion.  Eloss is never fitted
/// to an endpoint and never inferred from loc0.
///
/// Shadow D is a pre-registered mean-only refinement
///   10, 5, 2.5, 1.25 mm
/// defined from RKN4 order and the FASER magnet-cell scale.  It is not
/// the frozen-failed WB125 20/10/5 mm recipe and is not tuned from
/// Jacobian agreement.
///
/// WB128 may evaluate a common-grid independent FD of this same
/// certified mean function.  The caller freezes the nominal production
/// accepted-step / material-gating partition and reuses replay().
/// Mean semantics in this header stay Bethe + updateState gating.
struct MeanTransportShadow {
  static constexpr const char* kMethodP =
      "official_rkn4_nystrom_physics_semantics";
  static constexpr const char* kMethodD =
      "official_rkn4_nystrom_discretization_audit";
  static constexpr double kDStepsMm[4] = {10.0, 5.0, 2.5, 1.25};
  static constexpr int kNDSteps = 4;
  static constexpr double kMaxPathMm = 20000.0;
  static constexpr int kMaxNodes = 40000;
  static constexpr double kPlaneHitAbsMm = 1.0e-6;
  static constexpr double kMeanLoc0AbsMm = 1.0e-3;
  static constexpr double kMeanPathAbsMm = 1.0e-2;
  static constexpr double kMeanPosAbsMm = 1.0e-2;
  static constexpr double kMeanDirAbs = 1.0e-6;
  static constexpr double kMeanQopRel = 1.0e-4;
  static constexpr double kDivPosAbsMm = 1.0e-4;
  static constexpr double kDivDirAbs = 1.0e-7;
  static constexpr double kDivQopRel = 1.0e-5;
  static constexpr const char* kMeshJustification =
      "Shadow D uses Delta s, Delta s/2, Delta s/4, Delta s/8 starting at "
      "10 mm = one FASER magnet-cell scale.  Official mean stepping is "
      "RKN4 / Nystrom (global error O(h^4)); four halvings reach 1.25 mm "
      "inside a cell so a residual that vanishes is resolution and a "
      "stable offset is a physics-semantics mismatch.  Registered from "
      "method order and field-map scale before any 86 evaluation.  "
      "derivative_not_evaluated; jacobian_agreement_not_read; "
      "grid_selected_from_mean_contract_only.  Not the WB125 20/10/5 mm "
      "recipe and not tuned from Jacobian agreement.";

  struct State {
    Acts::Vector3 pos = Acts::Vector3::Zero();
    Acts::Vector3 dir = Acts::Vector3::UnitZ();
    double qop = 0.0;
  };

  struct MaterialNode {
    double sAbs = 0.0;
    bool volumeOnly = false;
    bool updateStateCalled = false;
    std::uint64_t geometryId = 0;
    double surfaceZ = 0.0;
    double pathCorrection = 1.0;
    double incidenceAngle = 0.0;
    Acts::MaterialSlab slab;
    Acts::Vector3 position = Acts::Vector3::Zero();
    Acts::Vector3 direction = Acts::Vector3::Zero();
  };

  struct Event {
    std::string kind;
    double sBefore = 0.0;
    double sAfter = 0.0;
    Acts::Vector3 posBefore = Acts::Vector3::Zero();
    Acts::Vector3 posAfter = Acts::Vector3::Zero();
    Acts::Vector3 dirBefore = Acts::Vector3::Zero();
    Acts::Vector3 dirAfter = Acts::Vector3::Zero();
    double qopBefore = 0.0;
    double qopAfter = 0.0;
    double dirNormBefore = 1.0;
    double dirNormAfter = 1.0;
    double stepSize = 0.0;
    Acts::Vector3 bBefore = Acts::Vector3::Zero();
    Acts::Vector3 bMiddle = Acts::Vector3::Zero();
    Acts::Vector3 bEnd = Acts::Vector3::Zero();
    bool fieldOk = true;
    std::uint64_t geometryId = 0;
    double surfaceZ = 0.0;
    double pathCorrection = 1.0;
    double incidenceAngle = 0.0;
    double slabThickness = 0.0;
    double thicknessInX0 = 0.0;
    double thicknessInL0 = 0.0;
    double materialX0 = 0.0;
    double materialL0 = 0.0;
    double materialAr = 0.0;
    double materialZ = 0.0;
    Acts::MaterialSlab slab;
    double qopAtInteraction = 0.0;
    double momentumBefore = 0.0;
    double energyBefore = 0.0;
    double betaBefore = 0.0;
    double gammaBefore = 0.0;
    double elossMean = 0.0;
    double elossMode = 0.0;
    double elossBethe = 0.0;
    double elossLandau = 0.0;
    double elossRadiative = 0.0;
    double elossEvaluatePointwise = 0.0;
    double elossApplied = 0.0;
    double productionDeltaE = 0.0;
    double deltaQop = 0.0;
    double absCharge = 1.0;
    double massGeV = 0.0;
    int absPdg = 0;
    double stageFactor = 1.0;
    bool volumeOnly = false;
    bool slabValid = false;
    bool updateStateCalled = false;
    bool hasSurfaceMaterialPointer = false;
    bool isStartSurface = false;
    bool isTargetSurface = false;
    std::string updateStage;
    std::string particleHypothesis = "muon";
    std::string elossSourceFunction;
  };

  struct Ledger {
    bool ok = false;
    State start;
    State end;
    double pathLength = 0.0;
    double loc0 = 0.0;
    bool projected = false;
    int nField = 0;
    int nMaterial = 0;
    int nFieldFails = 0;
    double nDotDir = 0.0;
    double distanceToPlaneMm = 0.0;
    double maxAbsDirNormMinusOne = 0.0;
    std::string abortReason;
    std::string method;
    std::vector<Event> events;
  };

  struct Residual {
    bool ok = false;
    bool projected = false;
    double loc0 = 0.0;
    double path = 0.0;
    double pos = 0.0;
    double dir = 0.0;
    double qopRel = 0.0;
    bool closed = false;
  };

  explicit MeanTransportShadow(const Acts::MagneticFieldContext& magFieldContext)
      : m_cache(m_wrapper.makeCache(magFieldContext)) {}

  bool field(const Acts::Vector3& pos, Acts::Vector3& B) {
    auto res = m_wrapper.getField(pos, m_cache);
    if (!res.ok()) {
      return false;
    }
    B = *res;
    return true;
  }

  static double muonMass() {
    return static_cast<double>(Acts::ParticleHypothesis::muon().mass());
  }

  static void kinematics(double qop, double& p, double& energy, double& beta,
                         double& gamma) {
    const double mass = muonMass();
    p = (std::abs(qop) > 0.0) ? (1.0 / std::abs(qop)) : 0.0;
    energy = std::hypot(mass, p);
    gamma = (mass > 0.0) ? (energy / mass) : 0.0;
    beta = (energy > 0.0) ? (p / energy) : 0.0;
  }

  static double productionDeltaEnergy(double qopBefore, double qopAfter,
                                      double pathSign) {
    double p0 = 0.0;
    double e0 = 0.0;
    double beta = 0.0;
    double gamma = 0.0;
    kinematics(qopBefore, p0, e0, beta, gamma);
    double p1 = 0.0;
    double e1 = 0.0;
    kinematics(qopAfter, p1, e1, beta, gamma);
    const double navDir = pathSign >= 0.0 ? 1.0 : -1.0;
    return (e0 - e1) * navDir;
  }

  static double qopAfterEnergyLoss(double qopBefore, double eloss,
                                   double pathSign, double absCharge) {
    const double mass = muonMass();
    const double navDir = pathSign >= 0.0 ? 1.0 : -1.0;
    const double p = (std::abs(qopBefore) > 0.0) ? (1.0 / std::abs(qopBefore))
                                                 : 0.0;
    const double nextE = std::hypot(mass, p) - eloss * navDir;
    double nextP = (mass < nextE) ? std::sqrt(nextE * nextE - mass * mass) : 0.0;
    nextP = std::max(10.0 * Acts::UnitConstants::MeV, nextP);
    const double absQ = (absCharge > 0.0) ? absCharge : 1.0;
    return std::copysign(absQ / nextP, qopBefore);
  }

  static void fillEnergyLoss(Event& ev, const State& y,
                             const Acts::MaterialSlab& slab, double pathSign) {
    ev.qopAtInteraction = y.qop;
    kinematics(y.qop, ev.momentumBefore, ev.energyBefore, ev.betaBefore,
               ev.gammaBefore);
    ev.slabThickness = slab.thickness();
    ev.thicknessInX0 = slab.thicknessInX0();
    ev.thicknessInL0 = slab.thicknessInL0();
    ev.materialX0 = slab.material().X0();
    ev.materialL0 = slab.material().L0();
    ev.materialAr = slab.material().Ar();
    ev.materialZ = slab.material().Z();
    ev.slab = slab;
    ev.slabValid = static_cast<bool>(slab);
    ev.massGeV = muonMass();
    if (ev.particleHypothesis.empty()) {
      ev.particleHypothesis = "muon";
    }
    if (ev.absPdg == 0) {
      ev.absPdg = static_cast<int>(Acts::PdgParticle::eMuon);
    }
    if (ev.absCharge == 0.0) {
      ev.absCharge = 1.0;
    }
    ev.elossSourceFunction =
        "Acts::detail::PointwiseMaterialInteraction::"
        "evaluatePointwiseMaterialInteraction -> Acts::computeEnergyLossBethe";
    if (!slab) {
      ev.elossMean = 0.0;
      ev.elossMode = 0.0;
      ev.elossBethe = 0.0;
      ev.elossLandau = 0.0;
      ev.elossRadiative = 0.0;
      ev.elossApplied = 0.0;
      return;
    }
    const float mass = static_cast<float>(muonMass());
    const float absQ = static_cast<float>(ev.absCharge);
    const float qop = static_cast<float>(y.qop);
    const auto pdg = static_cast<Acts::PdgParticle>(ev.absPdg);
    ev.elossBethe = Acts::computeEnergyLossBethe(slab, mass, qop, absQ);
    ev.elossLandau = Acts::computeEnergyLossLandau(slab, mass, qop, absQ);
    ev.elossRadiative =
        Acts::computeEnergyLossRadiative(slab, pdg, mass, qop, absQ);
    ev.elossMean = Acts::computeEnergyLossMean(slab, pdg, mass, qop, absQ);
    ev.elossMode = Acts::computeEnergyLossMode(slab, pdg, mass, qop, absQ);
    ev.elossApplied = ev.elossBethe;
    (void)pathSign;
  }

  static void applyProductionEnergyLoss(State& y, const Acts::MaterialSlab& slab,
                                        double pathSign, Event& ev) {
    fillEnergyLoss(ev, y, slab, pathSign);
    if (!slab || !ev.updateStateCalled) {
      ev.qopAfter = y.qop;
      ev.deltaQop = 0.0;
      ev.elossApplied = 0.0;
      return;
    }
    y.qop = qopAfterEnergyLoss(y.qop, ev.elossApplied, pathSign, ev.absCharge);
    ev.qopAfter = y.qop;
    ev.deltaQop = ev.qopAfter - ev.qopBefore;
  }

  bool rkn4Step(State& y, double h, Event& ev, bool normalize) {
    ev.kind = "field_propagation_interval";
    ev.posBefore = y.pos;
    ev.dirBefore = y.dir;
    ev.qopBefore = y.qop;
    ev.dirNormBefore = y.dir.norm();
    ev.stepSize = h;
    Acts::Vector3 B1 = Acts::Vector3::Zero();
    if (!field(y.pos, B1)) {
      ev.fieldOk = false;
      return false;
    }
    ev.bBefore = B1;
    const Acts::Vector3 k1 = y.qop * y.dir.cross(B1);
    const double h2 = h * h;
    const double half = 0.5 * h;
    const Acts::Vector3 pos1 = y.pos + half * y.dir + h2 * 0.125 * k1;
    Acts::Vector3 B2 = Acts::Vector3::Zero();
    if (!field(pos1, B2)) {
      ev.fieldOk = false;
      return false;
    }
    ev.bMiddle = B2;
    const Acts::Vector3 k2 = y.qop * (y.dir + half * k1).cross(B2);
    const Acts::Vector3 k3 = y.qop * (y.dir + half * k2).cross(B2);
    const Acts::Vector3 pos2 = y.pos + h * y.dir + h2 * 0.5 * k3;
    Acts::Vector3 B3 = Acts::Vector3::Zero();
    if (!field(pos2, B3)) {
      ev.fieldOk = false;
      return false;
    }
    ev.bEnd = B3;
    const Acts::Vector3 k4 = y.qop * (y.dir + h * k3).cross(B3);
    y.pos += h * y.dir + h2 / 6.0 * (k1 + k2 + k3);
    y.dir += (h / 6.0) * (k1 + 2.0 * (k2 + k3) + k4);
    if (normalize && y.dir.norm() > 0.0) {
      y.dir.normalize();
    }
    ev.posAfter = y.pos;
    ev.dirAfter = y.dir;
    ev.qopAfter = y.qop;
    ev.dirNormAfter = y.dir.norm();
    ev.sAfter = ev.sBefore + h;
    ev.fieldOk = true;
    return true;
  }

  static Residual residualAgainst(const Ledger& shadow, double officialLoc0,
                                  double officialPath,
                                  const Acts::Vector3& officialPos,
                                  const Acts::Vector3& officialDir,
                                  double officialQop) {
    Residual r;
    r.ok = shadow.ok;
    r.projected = shadow.projected;
    r.loc0 = shadow.projected ? (officialLoc0 - shadow.loc0) : 1.0e9;
    r.path = officialPath - shadow.pathLength;
    r.pos = (officialPos - shadow.end.pos).norm();
    r.dir = (officialDir - shadow.end.dir).norm();
    r.qopRel = std::abs(officialQop - shadow.end.qop) /
               std::max(std::abs(officialQop), 1.0e-12);
    r.closed = shadow.ok && shadow.projected &&
               std::abs(r.loc0) <= kMeanLoc0AbsMm &&
               std::abs(r.path) <= kMeanPathAbsMm && r.pos <= kMeanPosAbsMm &&
               r.dir <= kMeanDirAbs && r.qopRel <= kMeanQopRel;
    return r;
  }

  static bool diverged(const Event& prod, const Event& sh) {
    const double dPos = (prod.posAfter - sh.posAfter).norm();
    const double dDir = (prod.dirAfter - sh.dirAfter).norm();
    const double dQ = std::abs(prod.qopAfter - sh.qopAfter) /
                      std::max(std::abs(prod.qopAfter), 1.0e-12);
    return dPos > kDivPosAbsMm || dDir > kDivDirAbs || dQ > kDivQopRel;
  }

  Ledger replay(const State& start, const std::vector<Event>& production,
                const Acts::Vector3& planeCenter,
                const Acts::Vector3& planeNormal, double pathSign,
                bool applyMaterial, bool normalize, const std::string& method) {
    Ledger out;
    out.method = method;
    State y = start;
    if (y.dir.norm() > 0.0) {
      y.dir.normalize();
    }
    out.start = y;
    double s = 0.0;
    const auto distance = [&](const Acts::Vector3& pos) {
      return planeNormal.dot(pos - planeCenter);
    };
    for (const Event& src : production) {
      if (std::abs(s) > kMaxPathMm) {
        out.abortReason = "max_path";
        return out;
      }
      Event ev = src;
      ev.sBefore = s;
      if (src.kind == "surface_material" || src.kind == "material_interaction") {
        ev.kind = "surface_material";
        ev.posBefore = y.pos;
        ev.dirBefore = y.dir;
        ev.qopBefore = y.qop;
        ev.dirNormBefore = y.dir.norm();
        if (applyMaterial && !src.volumeOnly) {
          applyProductionEnergyLoss(y, src.slab, pathSign, ev);
          if (ev.updateStateCalled && src.slab) {
            ++out.nMaterial;
          }
        } else {
          ev.qopAfter = y.qop;
          ev.elossApplied = 0.0;
        }
        ev.posAfter = y.pos;
        ev.dirAfter = y.dir;
        ev.dirNormAfter = y.dir.norm();
        ev.sAfter = s;
        out.maxAbsDirNormMinusOne =
            std::max(out.maxAbsDirNormMinusOne, std::abs(ev.dirNormAfter - 1.0));
        out.events.push_back(ev);
        continue;
      }
      const double h = src.stepSize;
      if (std::abs(h) < 1.0e-18) {
        continue;
      }
      ev.sBefore = s;
      if (!rkn4Step(y, h, ev, normalize)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      s = ev.sAfter;
      ++out.nField;
      out.maxAbsDirNormMinusOne =
          std::max(out.maxAbsDirNormMinusOne, std::abs(ev.dirNormAfter - 1.0));
      out.events.push_back(ev);
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

  Ledger integrateUniform(const State& start,
                          const std::vector<MaterialNode>& materials,
                          double maxStepMm, const Acts::Vector3& planeCenter,
                          const Acts::Vector3& planeNormal, double pathSign,
                          double pathLength, bool applyMaterial,
                          bool normalize) {
    Ledger out;
    out.method = kMethodD;
    State y = start;
    if (y.dir.norm() > 0.0) {
      y.dir.normalize();
    }
    out.start = y;
    double s = 0.0;
    const double sEnd = std::min(std::abs(pathLength), kMaxPathMm);
    const double sign = pathSign >= 0.0 ? 1.0 : -1.0;
    std::vector<MaterialNode> events = materials;
    std::sort(events.begin(), events.end(),
              [](const MaterialNode& a, const MaterialNode& b) {
                return a.sAbs < b.sAbs;
              });
    std::size_t ie = 0;
    int guard = 0;
    const auto distance = [&](const Acts::Vector3& pos) {
      return planeNormal.dot(pos - planeCenter);
    };
    while (s < sEnd - 1.0e-12 && guard < kMaxNodes) {
      ++guard;
      if (ie < events.size() && events[ie].sAbs <= s + 1.0e-9) {
        Event ev;
        ev.kind = "surface_material";
        ev.sBefore = s;
        ev.sAfter = s;
        ev.posBefore = y.pos;
        ev.dirBefore = y.dir;
        ev.qopBefore = y.qop;
        ev.dirNormBefore = y.dir.norm();
        ev.geometryId = events[ie].geometryId;
        ev.surfaceZ = events[ie].surfaceZ;
        ev.pathCorrection = events[ie].pathCorrection;
        ev.incidenceAngle = events[ie].incidenceAngle;
        ev.volumeOnly = events[ie].volumeOnly;
        ev.updateStateCalled = events[ie].updateStateCalled;
        ev.slab = events[ie].slab;
        if (applyMaterial && !events[ie].volumeOnly) {
          applyProductionEnergyLoss(y, events[ie].slab, pathSign, ev);
          if (ev.updateStateCalled && events[ie].slab) {
            ++out.nMaterial;
          }
        } else {
          ev.qopAfter = y.qop;
        }
        ev.posAfter = y.pos;
        ev.dirAfter = y.dir;
        ev.dirNormAfter = y.dir.norm();
        out.maxAbsDirNormMinusOne =
            std::max(out.maxAbsDirNormMinusOne, std::abs(ev.dirNormAfter - 1.0));
        out.events.push_back(ev);
        ++ie;
        continue;
      }
      double nextFill = std::min(s + maxStepMm, sEnd);
      if (ie < events.size()) {
        nextFill = std::min(nextFill, events[ie].sAbs);
      }
      const double h = sign * (nextFill - s);
      Event ev;
      ev.sBefore = s * sign;
      if (!rkn4Step(y, h, ev, normalize)) {
        ++out.nFieldFails;
        out.abortReason = "field_query_failed";
        return out;
      }
      s = nextFill;
      ev.sAfter = s * sign;
      ++out.nField;
      out.maxAbsDirNormMinusOne =
          std::max(out.maxAbsDirNormMinusOne, std::abs(ev.dirNormAfter - 1.0));
      out.events.push_back(ev);
    }
    const double dNow = distance(y.pos);
    const double denom = planeNormal.dot(y.dir);
    if (std::abs(dNow) > kPlaneHitAbsMm && std::abs(denom) > 1.0e-12) {
      y.pos -= (dNow / denom) * y.dir;
    }
    out.ok = true;
    out.end = y;
    out.pathLength = sign * s;
    out.nDotDir = planeNormal.dot(y.dir);
    out.distanceToPlaneMm = distance(y.pos);
    return out;
  }

 private:
  FASERMagneticFieldWrapper m_wrapper;
  Acts::MagneticFieldProvider::Cache m_cache;
};

/// Read-only snapshot before / after MaterialInteractor in one actor cycle.
struct MeanStateSnapshot {
  double path = 0.0;
  Acts::Vector3 pos = Acts::Vector3::Zero();
  Acts::Vector3 dir = Acts::Vector3::Zero();
  double qop = 0.0;
  double dirNorm = 0.0;
  std::uint64_t geometryId = 0;
  double surfaceZ = 0.0;
  bool hasSurface = false;
  bool hasSurfaceMaterial = false;
};

template <int Tag>
struct MeanSnapshotActor {
  struct result_type {
    std::vector<MeanStateSnapshot> snaps;
  };

  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  void operator()(propagator_state_t& state, const stepper_t& stepper,
                  const navigator_t& navigator, result_type& result,
                  const Acts::Logger& /*logger*/) const {
    if (state.stage == Acts::PropagatorStage::postPropagation) {
      return;
    }
    MeanStateSnapshot snap;
    snap.path = state.pathLength;
    snap.pos = stepper.position(state.stepping);
    snap.dir = stepper.direction(state.stepping);
    snap.qop = stepper.qOverP(state.stepping);
    snap.dirNorm = snap.dir.norm();
    const Acts::Surface* surface = navigator.currentSurface(state.navigation);
    if (surface != nullptr) {
      snap.hasSurface = true;
      snap.geometryId = surface->geometryId().value();
      snap.surfaceZ = surface->center(state.geoContext).z();
      snap.hasSurfaceMaterial = surface->surfaceMaterial() != nullptr;
    }
    result.snaps.push_back(snap);
  }
};

using MeanPreMaterialActor = MeanSnapshotActor<0>;
using MeanPostMaterialActor = MeanSnapshotActor<1>;

/// Read-only probe immediately before MaterialInteractor in the same cycle.
/// Constructs PointwiseMaterialInteraction and calls the compiled
/// evaluateMaterialSlab / evaluatePointwiseMaterialInteraction.  Does not
/// call updateState and does not change the production stepper state.
struct MeanMaterialProbeActor {
  struct Probe {
    bool hasSurface = false;
    bool hasSurfaceMaterial = false;
    bool slabValid = false;
    bool updateStateWouldRun = false;
    bool isStartSurface = false;
    bool isTargetSurface = false;
    std::uint64_t geometryId = 0;
    double surfaceZ = 0.0;
    double pathCorrection = 0.0;
    double stageFactor = 1.0;
    double qop = 0.0;
    double momentum = 0.0;
    double massGeV = 0.0;
    double absCharge = 1.0;
    int absPdg = 0;
    double elossEvaluatePointwise = 0.0;
    double elossBethe = 0.0;
    double elossLandau = 0.0;
    double elossRadiative = 0.0;
    double elossMean = 0.0;
    double elossMode = 0.0;
    std::string updateStage;
    std::string particleHypothesis = "muon";
    std::string elossSourceFunction;
    Acts::MaterialSlab slab;
  };

  struct result_type {
    std::vector<Probe> probes;
  };

  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  void operator()(propagator_state_t& state, const stepper_t& stepper,
                  const navigator_t& navigator, result_type& result,
                  const Acts::Logger& /*logger*/) const {
    if (state.stage == Acts::PropagatorStage::postPropagation) {
      return;
    }
    Probe p;
    const Acts::Surface* surface = navigator.currentSurface(state.navigation);
    if (surface == nullptr) {
      result.probes.push_back(p);
      return;
    }
    p.hasSurface = true;
    p.geometryId = surface->geometryId().value();
    p.surfaceZ = surface->center(state.geoContext).z();
    p.hasSurfaceMaterial = surface->surfaceMaterial() != nullptr;
    p.isStartSurface = surface == navigator.startSurface(state.navigation);
    p.isTargetSurface = surface == navigator.targetSurface(state.navigation);
    if (p.isStartSurface) {
      p.updateStage = "PostUpdate";
    } else if (p.isTargetSurface) {
      p.updateStage = "PreUpdate";
    } else {
      p.updateStage = "FullUpdate";
    }
    if (!p.hasSurfaceMaterial) {
      result.probes.push_back(p);
      return;
    }
    Acts::MaterialUpdateStage stage = Acts::MaterialUpdateStage::FullUpdate;
    if (p.isStartSurface) {
      stage = Acts::MaterialUpdateStage::PostUpdate;
    } else if (p.isTargetSurface) {
      stage = Acts::MaterialUpdateStage::PreUpdate;
    }
    p.stageFactor =
        surface->surfaceMaterial()->factor(state.options.direction, stage);
    Acts::detail::PointwiseMaterialInteraction interaction(surface, state,
                                                           stepper);
    p.qop = interaction.qOverP;
    p.momentum = interaction.momentum;
    p.massGeV = interaction.mass;
    p.absCharge = interaction.absQ;
    p.absPdg = static_cast<int>(interaction.absPdg);
    p.particleHypothesis =
        (interaction.absPdg == Acts::PdgParticle::eMuon) ? "muon" : "other";
    p.slabValid = interaction.evaluateMaterialSlab(state, navigator);
    p.slab = interaction.slab;
    p.pathCorrection = interaction.pathCorrection;
    p.updateStateWouldRun = p.slabValid;
    p.elossSourceFunction =
        "Acts::detail::PointwiseMaterialInteraction::"
        "evaluatePointwiseMaterialInteraction -> Acts::computeEnergyLossBethe";
    if (p.slabValid) {
      interaction.evaluatePointwiseMaterialInteraction(true, true);
      p.elossEvaluatePointwise = interaction.Eloss;
      const float mass = interaction.mass;
      const float qop = interaction.qOverP;
      const float absQ = interaction.absQ;
      p.elossBethe =
          Acts::computeEnergyLossBethe(interaction.slab, mass, qop, absQ);
      p.elossLandau =
          Acts::computeEnergyLossLandau(interaction.slab, mass, qop, absQ);
      p.elossRadiative = Acts::computeEnergyLossRadiative(
          interaction.slab, interaction.absPdg, mass, qop, absQ);
      p.elossMean = Acts::computeEnergyLossMean(
          interaction.slab, interaction.absPdg, mass, qop, absQ);
      p.elossMode = Acts::computeEnergyLossMode(
          interaction.slab, interaction.absPdg, mass, qop, absQ);
    }
    result.probes.push_back(p);
  }
};

#endif
