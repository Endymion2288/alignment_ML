  SG::ReadCondHandleKey<FaserFieldMapCondObj> m_mapKey{this,"MapKey","fieldMapCondObj","actual map"};
  SG::ReadCondHandleKey<FaserFieldCacheCondObj> m_cacheKey{this,"CacheKey","fieldCondObj","actual cache"};

  Acts::BoundTrackParameters propagate(const Acts::BoundTrackParameters& start,
      const Acts::Transform3& frame,const EventContext& ctx,const Acts::GeometryContext& g) const {
    auto plane=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);
    auto r=m_active->propagate(ctx,start,*plane,Acts::Direction::Forward);
    if(!r)throw std::runtime_error("official continuation failed");
    const Acts::Vector3 pos=frame.inverse()*r->position(g);
    if(std::abs(pos.z())>1e-6 || r->direction().z()<=0 ||
       std::abs(r->parameters()[Acts::eBoundQOverP]-start.parameters()[Acts::eBoundQOverP])>1e-14)
      throw std::runtime_error("continuation plane/qop/direction contract");
    return *r; // complete returned state, including time and particle hypothesis
  }
  Json state(const Acts::BoundTrackParameters& s,const Acts::GeometryContext& g) const {
    const auto x=s.position(g),u=s.direction();V4 h;h<<x.x(),x.y(),u.x()/u.z(),u.y()/u.z();
    return {{"h",encode(h)},{"position_mm",encode(x)},{"direction",encode(u)},
      {"q_over_p_per_MeV",s.parameters()[Acts::eBoundQOverP]*Acts::UnitConstants::MeV},
      {"time_Acts",s.time()},{"bound_parameters",encode(s.parameters())}};
  }
  void audit() {
    const auto& ctx=Gaudi::Hive::currentContext();const xAOD::EventInfo* header=nullptr;
    if(evtStore()->retrieve(header,"EventInfo").isFailure() ||
       header->runNumber()!=m_fixture.at("actual_run").get<unsigned>() ||
       header->eventNumber()!=m_fixture.at("actual_event").get<unsigned long long>() ||
       header->runNumber()!=ctx.eventID().run_number() || header->eventNumber()!=ctx.eventID().event_number())
      throw std::runtime_error("actual event identity mismatch");
    const auto& p=m_fixture.at("wb94_protocol");
    SG::ReadCondHandle<FaserFieldMapCondObj> mh{m_mapKey,ctx};
    SG::ReadCondHandle<FaserFieldCacheCondObj> ch{m_cacheKey,ctx};
    if(!mh.isValid() || !ch.isValid() || !mh->fieldMap())throw std::runtime_error("missing actual field conditions");
    const auto* map=mh->fieldMap();EventIDRange mr,cr;
    if(!mh.range(mr) || !ch.range(cr))throw std::runtime_error("missing field IOV");
    std::ostringstream mi,ci;mi<<mr;ci<<cr;
    const auto g=m_tools[0]->trackingGeometryTool()->getGeometryContext(ctx).context();
    if(m_tools[0]->getMagneticFieldContext(ctx).get<const FaserFieldCacheCondObj*>()!=ch.cptr())
      throw std::runtime_error("wrapper conditions differ from actual cache");
    V5 seed;seed.head<4>()=vector<4>(m_fixture.at("references").at(0).at("fixed_z_state"));
    seed[4]=m_fixture.at("references").at(0).at("q_over_p_per_MeV");
    const double z=m_fixture.at("references").at(0).at("z_state_mm");
    const double edge=p.at("expected_zone").at("min_mm").at(2);
    const double inner=edge+p.at("interior_restart_offset_mm").get<double>();
    Acts::Transform3 entryFrame=Acts::Transform3::Identity(),innerFrame=entryFrame;
    entryFrame.translation().z()=edge;innerFrame.translation().z()=inner;
    const auto* zone=map->findBFieldZone(seed[0],seed[1],edge+1);
    if(!zone || zone->id()!=p.at("expected_zone").at("id"))throw std::runtime_error("unexpected actual zone");
    Json minimum=Json::array(),maximum=Json::array();
    for(int k=0;k<3;++k){minimum.push_back(zone->min(k));maximum.push_back(zone->max(k));}
    if(minimum!=p.at("expected_zone").at("min_mm") || maximum!=p.at("expected_zone").at("max_mm"))
      throw std::runtime_error("actual zone bounds changed");
    FASERMagneticFieldWrapper field;
    auto mctx=m_tools[0]->getMagneticFieldContext(ctx);auto persistent=field.makeCache(mctx);
    auto query=[&](const Acts::Vector3& pos,auto& cache){auto b=field.getField(pos,cache);
      if(!b.ok())throw std::runtime_error("actual field query failed");return b.value();};
    auto sample=[&](const Acts::Vector3& pos,auto& cache){auto fresh=field.makeCache(mctx);
      const auto* q=map->findBFieldZone(pos.x(),pos.y(),pos.z());
      return Json{{"position_mm",encode(pos)},{"zone_id",q?q->id():-1},
        {"field_T",encode(query(pos,cache)/Acts::UnitConstants::T)},
        {"fresh_field_T",encode(query(pos,fresh)/Acts::UnitConstants::T)}};};
    Json probes=Json::array();
    for(const auto& d:p.at("boundary_probe_offsets_mm"))probes.push_back(sample(Acts::Vector3(seed[0],seed[1],edge+d.get<double>()),persistent));
    Json reverse=Json::array();auto reverseCache=field.makeCache(mctx);
    for(auto it=p.at("boundary_probe_offsets_mm").rbegin();it!=p.at("boundary_probe_offsets_mm").rend();++it)
      reverse.push_back(sample(Acts::Vector3(seed[0],seed[1],edge+it->get<double>()),reverseCache));
    Json controls=Json::array();
    for(const auto& pos:std::vector<Acts::Vector3>{Acts::Vector3(seed[0],seed[1],z),
        Acts::Vector3(seed[0],seed[1],inner),Acts::Vector3(edge+1,seed[0],seed[1]),
        Acts::Vector3(201,seed[1],inner),Acts::Vector3(seed[0],201,inner)})controls.push_back(sample(pos,persistent));
    const auto math=WB94Reference::controls(Acts::UnitConstants::T);
    Json out={{"input_xaod",m_fixture.at("input_xaod")},{"ordinal",m_fixture.at("ordinal")},
      {"actual_run",header->runNumber()},{"actual_event",header->eventNumber()},
      {"seed",encode(seed)},{"seed_z_mm",z},
      {"conditions",{{"map_key",m_mapKey.key()},{"cache_key",m_cacheKey.key()},{"map_IOV",mi.str()},
        {"cache_IOV",ci.str()},{"scale",ch->dipoleFieldScaleFactor()},{"zone_id",zone->id()},
        {"min_mm",minimum},{"max_mm",maximum},{"wrapper_cache_identity",true}}},
      {"probes",probes},{"reverse_probes",reverse},{"domain_controls",controls},
      {"math_control",{{"max_scaled_error",math[0]},{"wrong_sign_error",math[1]},{"wrong_unit_error",math[2]}}},
      {"references",Json::array()},{"caps",Json::array()}};
    const V5 xs=vector<5>(p.at("seed_steps"));
    std::vector<V5> seeds{seed};Json labels=Json::array({Json{{"name","nominal"}}});
    for(int k=0;k<6;++k){V5 d=V5::Zero();if(k<5)d[k]=xs[k];else{d=xs;for(int a=1;a<5;a+=2)d[a]*=-1;}
      for(const auto& pair:std::vector<std::pair<std::string,double>>{{"central_plus",.5},{"central_minus",-.5},{"full",.25},{"half",.125}}){
        seeds.push_back(seed+pair.second*d);labels.push_back({{"direction",p.at("directions").at(k)},{"name",pair.first}});}}
    Acts::BoundTrackParameters insideStart=bound(seed,z,g);
    for(const auto& step:p.at("reference_rk4_dz_mm")) {
      ATH_MSG_INFO("WB94 independent reference dz_mm="<<step);
      Json ladder={{"dz_mm",step},{"samples",Json::array()}};
      for(size_t n=0;n<seeds.size();++n) {
        const auto x=seeds[n];WB94Reference::State r{x[0],x[1],x[2],x[3],0};double current=z;
        auto cache=field.makeCache(mctx);size_t outside=0,inside=0;
        bool before=true;const double eps=p.at("reference_boundary_side_epsilon_mm");
        auto get=[&](double xx,double yy,double zz){
          if(std::abs(zz-edge)<eps/2)zz=edge+(before?-eps:eps);
          const auto* q=map->findBFieldZone(xx,yy,zz);
          if(before){++outside;if(q)throw std::runtime_error("reference pre-entry unexpectedly in map");}
          else {++inside;if(!q || q->id()!=zone->id())throw std::runtime_error("reference leaves actual map");}
          const auto b=query(Acts::Vector3(xx,yy,zz),cache);return WB94Reference::Field{b.x(),b.y(),b.z()};};
        auto go=[&](double to){r=WB94Reference::integrate(r,current,to,step.get<double>(),x[4]/Acts::UnitConstants::MeV,
          Acts::ParticleHypothesis::muon().mass(),get);current=to;V4 h;for(int a=0;a<4;++a)h[a]=r[a];
          return Json{{"z_mm",to},{"h",encode(h)},{"time_Acts",r[4]},{"q_over_p_per_MeV",x[4]}};};
        Json row={{"label",labels[n]},{"entry",go(edge)}};before=false;row["interior"]=go(inner);row["targets"]=Json::array();
        if(n==0 && step==p.at("reference_rk4_dz_mm").back()) {
          V5 v;for(int a=0;a<4;++a)v[a]=r[a];v[4]=x[4];const auto temp=bound(v,inner,g);
          auto parameters=temp.parameters();parameters[Acts::eBoundTime]=r[4];
          insideStart=Acts::BoundTrackParameters(temp.referenceSurface().getSharedPtr(),parameters,std::nullopt,Acts::ParticleHypothesis::muon());
        }
        for(const auto& ref:m_fixture.at("references"))if(ref.at("station")!=0)row["targets"].push_back(go(ref.at("z_state_mm")));
        row["outside_stages"]=outside;row["inside_stages"]=inside;ladder["samples"].push_back(row);
      }
      out["references"].push_back(ladder);
    }
    for(size_t ti=0;ti<m_tools.size();++ti) {
      ATH_MSG_INFO("WB94 official cap_m="<<p.at("max_step_sizes_m").at(ti));
      m_active=m_tools[ti].get();if(m_active->getMagneticFieldContext(ctx).get<const FaserFieldCacheCondObj*>()!=ch.cptr())throw std::runtime_error("tool cache context mismatch");
      Json cap={{"cap_m",p.at("max_step_sizes_m").at(ti)},{"targets",Json::array()}};
      const auto geometry=m_active->trackingGeometryTool();const auto ids=geometry->getIdentifierMap();
      for(int station=1;station<=3;++station) {
        ATH_MSG_INFO("WB94 target station="<<station);
        const auto& old=m_fixture.at("wb92_targets").at(station);const auto frame=transform(old.at("frame"));
        Json sensors=Json::array();
        for(const auto& sensor:old.at("sensors")) {
          const Identifier strip(static_cast<Identifier::value_type>(sensor.at("strip").get<unsigned long long>()));const auto wafer=m_helper->wafer_id(strip);
          if(ids->count(wafer)!=1 || m_helper->station(wafer)!=station || wafer.get_compact()!=sensor.at("wafer") || ids->at(wafer).value()!=sensor.at("geometry_id"))throw std::runtime_error("sensor identity mismatch");
          const auto* surface=geometry->trackingGeometry()->findSurface(ids->at(wafer));
          if(!surface || (surface->transform(g).matrix()-transform(sensor.at("transform")).matrix()).cwiseAbs().maxCoeff()>1e-9)throw std::runtime_error("sensor transform changed");
          sensors.push_back({{"wafer",wafer.get_compact()},{"transform",encode(surface->transform(g).matrix())}});
        }
        auto paths=[&](const V5& x){const auto start=bound(x,z,g),entry=propagate(start,entryFrame,ctx,g);
          const auto ep=entry.position(g);if(!map->findBFieldZone(ep.x(),ep.y(),edge+1e-6) || map->findBFieldZone(ep.x(),ep.y(),edge-1e-6))throw std::runtime_error("entry domain is not verified");
          return Json{{"direct",state(propagate(start,frame,ctx,g),g)},
            {"boundary_split",state(propagate(entry,frame,ctx,g),g)},{"entry",state(entry,g)}};};
        Json target={{"station",station},{"frame",old.at("frame")},{"sensors",sensors},{"y",m_fixture.at("references").at(station).at("fixed_z_state")},
          {"nominal",paths(seed)},{"directions",Json::array()},{"traces",Json::array()}};
        const auto nominalStart=bound(seed,z,g),sham=propagate(nominalStart,innerFrame,ctx,g);
        target["interior_split"]=state(propagate(sham,frame,ctx,g),g);
        target["fixed_reference_start"]=state(propagate(insideStart,frame,ctx,g),g);
        target["interior_prefix"]=state(sham,g);target["fixed_start"]=state(insideStart,g);
        for(int k=0;k<6;++k) {
          V5 d=V5::Zero();if(k<5)d[k]=xs[k];else{d=xs;for(int a=1;a<5;a+=2)d[a]*=-1;}
          Json dir={{"name",p.at("directions").at(k)},{"samples",Json::array()}};
          for(const auto& multiplier:p.at("step_multipliers")){const V5 delta=multiplier.get<double>()*d;
            dir["samples"].push_back({{"multiplier",multiplier},{"central_plus",paths(seed+.5*delta)},
              {"central_minus",paths(seed-.5*delta)},{"full",paths(seed+.25*delta)},{"half",paths(seed+.125*delta)}});}
          target["directions"].push_back(dir);
        }
        for(bool split:{false,true}) {
          auto traceStart=split?propagate(nominalStart,entryFrame,ctx,g):nominalStart;
          Json trace={{"path",split?"boundary_split_continuation":"direct"},{"start",state(traceStart,g)},{"steps",Json::array()}};
          auto plane=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);
          const auto steps=m_active->propagationSteps(ctx,traceStart,*plane,Acts::Direction::Forward);
          if(steps.first.empty())throw std::runtime_error("empty official trace");
          auto tc=field.makeCache(mctx);
          for(const auto& t:steps.first){auto q=sample(t.position,tc);q["geometry_id"]=t.geoID.value();q["momentum_Acts"]=encode(t.momentum);trace["steps"].push_back(q);}
          trace["endpoint"]=state(propagate(traceStart,frame,ctx,g),g);target["traces"].push_back(trace);
        }
        cap["targets"].push_back(target);
      }
      out["caps"].push_back(cap);
    }
    std::ifstream maps("/proc/self/maps");std::string line;std::set<std::string> libraries;
    while(std::getline(maps,line)){auto slash=line.find('/');if(slash!=std::string::npos && line.find(".so",slash)!=std::string::npos)libraries.insert(line.substr(slash));}
    out["loaded_libraries"]=libraries;
    const int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive result failed");::close(fd);
    std::ofstream stream(m_output.value());stream<<out.dump(2)<<std::endl;if(!stream)throw std::runtime_error("result write failed");
  }
