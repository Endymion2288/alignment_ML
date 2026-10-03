#include "AthenaBaseComps/AthAlgorithm.h"
#include "xAODEventInfo/EventInfo.h"
#include "xAODTruth/TruthParticleContainer.h"
#include "TrackerPrepRawData/FaserSCT_ClusterContainer.h"
#include "TrackerSimData/TrackerSimDataCollection.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "TrackerRIO_OnTrack/FaserSCT_ClusterOnTrack.h"
#include "TrkTrack/TrackCollection.h"
#include "TrkParameters/TrackParameters.h"
#include "AtlasHepMC/GenParticle.h"
#include "AtlasHepMC/GenEvent.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <map>
#include <set>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>

namespace WB113 {
using Json=nlohmann::json;
double finite(double v){if(!std::isfinite(v))throw std::runtime_error("nonfinite EDM value");return v;}
template<class V> Json vector(const V& v){Json j=Json::array();for(int i=0;i<v.size();++i)j.push_back(finite(v[i]));return j;}
class PersistedProvenance final:public AthAlgorithm {
 public:
  PersistedProvenance(const std::string& n,ISvcLocator* s):AthAlgorithm(n,s){}
  StatusCode initialize() override {
    ATH_CHECK(detStore()->retrieve(m_id,"FaserSCT_ID"));
    try{std::ifstream f(m_fixture.value());f>>m_f;if(!f)throw std::runtime_error("fixture read");
      std::ifstream c(m_control.value());c>>m_c;if(!c)throw std::runtime_error("control read");
    }catch(const std::exception& e){ATH_MSG_ERROR(e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
  StatusCode execute() override {
    try{audit();}catch(const std::exception& e){ATH_MSG_ERROR("WB113 fail closed: "<<e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
 private:
  Gaudi::Property<std::string> m_fixture{this,"FixturePath",""},m_control{this,"ControlPath",""},m_output{this,"AuditPath",""};
  const FaserSCT_ID* m_id=nullptr;Json m_f,m_c;
  Json link(const HepMcParticleLink& l) {
    const bool valid=l.isValid();
    Json j{{"barcode",l.barcode()},{"event_index",l.eventIndex()},
      {"event_collection",std::string(1,l.getEventCollectionAsChar())},{"valid",valid},
      {"event_position",nullptr},{"resolved_particle",nullptr}};
    if(valid){auto position=l.getEventPositionInCollection(evtStore().get());
      if(position!=HepMcParticleLink::ExtendedBarCode::UNDEFINED)j["event_position"]=position;
      auto p=l.cptr();
      if(!p)throw std::runtime_error("valid link without particle");
      auto e=p->parent_event();
      j["resolved_particle"]={{"pdg",p->pdg_id()},{"momentum_native",Json::array({finite(p->momentum().px()),finite(p->momentum().py()),finite(p->momentum().pz())})},
        {"parent_event_number",e?Json(e->event_number()):Json(nullptr)},
        {"momentum_unit",e?Json(e->momentum_unit()==HepMC3::Units::MEV?"MEV":"GEV"):Json("UNKNOWN")}};
    }
    return j;
  }
  void audit() {
    const xAOD::EventInfo* header=nullptr;
    if(evtStore()->retrieve(header,"EventInfo").isFailure())throw std::runtime_error("header retrieval failed");
    if(!header || header->runNumber()!=m_f.at("actual_run").get<unsigned>() || header->eventNumber()!=m_f.at("actual_event").get<uint64_t>())throw std::runtime_error("event identity changed");
    const Tracker::FaserSCT_ClusterContainer* clusters=nullptr;
    if(evtStore()->retrieve(clusters,"SCT_ClusterContainer").isFailure() || !clusters)throw std::runtime_error("missing cluster container");
    const TrackerSimDataCollection* sdo=nullptr;const xAOD::TruthParticleContainer* truth=nullptr;
    if(evtStore()->contains<TrackerSimDataCollection>("SCT_SDO_Map")){
      if(evtStore()->retrieve(sdo,"SCT_SDO_Map").isFailure())throw std::runtime_error("SDO retrieval failed");}
    if(evtStore()->contains<xAOD::TruthParticleContainer>("TruthParticles")){
      if(evtStore()->retrieve(truth,"TruthParticles").isFailure())throw std::runtime_error("truth retrieval failed");}
    std::map<uint64_t,int> allowed;std::map<uint64_t,const Tracker::FaserSCT_Cluster*> found;
    for(const auto& r:m_f.at("references"))for(const auto& id:r.at("clusters"))
      if(!allowed.emplace(id.get<uint64_t>(),r.at("station").get<int>()).second)throw std::runtime_error("duplicate allowlist cluster");
    for(const auto* coll:*clusters)for(const auto* c:*coll)if(allowed.count(c->identify().get_compact())){
      auto id=c->identify().get_compact();if(!found.emplace(id,c).second)throw std::runtime_error("duplicate persisted cluster");
      if(m_id->station(c->identify())!=allowed.at(id))throw std::runtime_error("cluster station changed");}
    if(found.size()!=allowed.size())throw std::runtime_error("allowlisted cluster missing");
    Json rows=Json::array();std::set<int> barcodes;
    for(const auto& [id,c]:found){Json rdos=Json::array();std::set<uint64_t> ids;
      for(const auto rid:c->rdoList()){
        if(!ids.insert(rid.get_compact()).second)throw std::runtime_error("duplicate cluster RDO");
        Json r{{"rdo_id",rid.get_compact()},{"sdo_present",false},{"deposits",Json::array()}};
        if(sdo){auto it=sdo->find(rid);if(it!=sdo->end()){
          r["sdo_present"]=true;r["sdo_word"]=it->second.word();
          for(const auto& d:it->second.getdeposits()){auto j=link(d.first);j["weight_native"]=finite(d.second);r["deposits"].push_back(j);barcodes.insert(d.first.barcode());}}}
        rdos.push_back(r);
      }
      rows.push_back(Json{{"cluster_id",id},{"station",allowed.at(id)},{"rdos",rdos}});
    }
    Json particles=Json::array();
    if(truth)for(const auto* p:*truth)if(barcodes.count(p->barcode()))
      particles.push_back(Json{{"barcode",p->barcode()},{"pdg",p->pdgId()},{"status",p->status()},
        {"charge_e",finite(p->charge())},{"momentum_native",Json::array({finite(p->px()),finite(p->py()),finite(p->pz())})},
        {"momentum_unit","UNKNOWN_XAOD_UNIT_UNTIL_SOURCE_OR_RESOLVED_LINK_CHECK"}});
    Json tracks=Json::array();
    for(const auto& key:m_c.at("track_keys")){
      const std::string k=key.get<std::string>();const TrackCollection* collection=nullptr;
      if(evtStore()->retrieve(collection,k).isFailure() || !collection)throw std::runtime_error("metadata track container retrieval failed");
      Json matches=Json::array();size_t index=0;
      for(const auto* track:*collection){const size_t ordinal=index++;if(!track)throw std::runtime_error("null persisted track");
        Json membership=Json::array();std::set<uint64_t> overlap;auto measurements=track->measurementsOnTrack();
        if(measurements)for(const auto* m:*measurements){auto c=dynamic_cast<const Tracker::FaserSCT_ClusterOnTrack*>(m);
          if(!c)continue;const auto* prd=c->prepRawData();auto id=prd?prd->identify().get_compact():c->identify().get_compact();
          membership.push_back(Json{{"cluster_id",id},{"prd_link_resolved",prd!=nullptr}});if(allowed.count(id))overlap.insert(id);}
        if(overlap.empty())continue;
        Json parameters=Json::array();auto pars=track->trackParameters();
        if(pars)for(const auto* p:*pars){if(!p)throw std::runtime_error("null track parameters");
          parameters.push_back(Json{{"native_parameters",vector(p->parameters())},{"global_position_native",vector(p->position())},
            {"global_momentum_native",vector(p->momentum())},{"charge_e",finite(p->charge())}});}
        matches.push_back(Json{{"track_index",ordinal},{"allowlist_overlap",overlap},{"cluster_membership",membership},{"parameters",parameters}});
      }
      tracks.push_back(Json{{"key",k},{"container_size",collection->size()},{"matching_tracks",matches}});
    }
    std::set<std::string> libraries;std::ifstream maps("/proc/self/maps");std::string line;
    while(std::getline(maps,line)){auto s=line.find('/');if(s!=std::string::npos&&line.find(".so",s)!=std::string::npos)libraries.insert(line.substr(s));}
    Json result{{"schema","wb113_persisted_provenance_v1"},{"identity",Json{{"actual_run",header->runNumber()},{"actual_event",header->eventNumber()},
      {"input_xaod",m_f.at("input_xaod")},{"ordinal",m_f.at("ordinal")}}},
      {"sdo_container_present",sdo!=nullptr},{"truth_container_present",truth!=nullptr},{"clusters",rows},
      {"related_truth_particles",particles},{"persisted_tracks",tracks},{"loaded_libraries",libraries},
      {"new_reconstruction_calls",0},{"new_propagation_calls",0},{"algorithm_field_queries",0}};
    const int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0)throw std::runtime_error("exclusive output failed");::close(fd);
    std::ofstream out(m_output.value());out<<result.dump(2)<<std::endl;if(!out)throw std::runtime_error("write failed");
  }
};
}
DECLARE_COMPONENT(WB113::PersistedProvenance)
