  SG::ReadCondHandleKey<FaserFieldMapCondObj> m_mapKey{this,"MapKey","fieldMapCondObj","actual map"};
  SG::ReadCondHandleKey<FaserFieldCacheCondObj> m_cacheKey{this,"CacheKey","fieldCondObj","actual cache"};
  Json state(const Acts::BoundTrackParameters& s,const Acts::GeometryContext& g)const {
    const auto x=s.position(g),u=s.direction();V4 h;h<<x.x(),x.y(),u.x()/u.z(),u.y()/u.z();
    if(u.z()<=0)throw std::runtime_error("nonforward returned state");
    return {{"h",encode(h)},{"position_mm",encode(x)},{"direction",encode(u)},
      {"q_over_p_per_MeV",s.parameters()[Acts::eBoundQOverP]*Acts::UnitConstants::MeV},
      {"time_Acts",s.time()},{"bound_parameters",encode(s.parameters())}};
  }
  Json mathControls(const Acts::GeometryContext& g,const Acts::MagneticFieldContext& m)const {
    Json out=Json::array();
    for(double tolerance:{1e-4,1e-5,1e-6}) {
      double worst=0,wrongSign=0,wrongUnit=0;
      for(bool zero:{true,false}) {
        auto b=std::make_shared<Acts::ConstantBField>(zero?Acts::Vector3::Zero().eval():Acts::Vector3(0,Acts::UnitConstants::T,0));
        Acts::Propagator<Acts::EigenStepper<>,Acts::VoidNavigator> engine{Acts::EigenStepper<>(b),Acts::VoidNavigator{}};
        Acts::PropagatorOptions<> o(g,m);o.stepTolerance=tolerance;o.maxStepSize=1.;o.maxSteps=1000;o.loopProtection=false;
        Acts::Transform3 t=Acts::Transform3::Identity();t.translation().z()=100.;auto target=Acts::Surface::makeShared<Acts::PlaneSurface>(t);
        V5 x=V5::Zero();x[4]=1e-5;
        const double w=.01*Acts::UnitConstants::T,c=std::sqrt(1-w*w*10000);
        V4 exact=V4::Zero();if(!zero){exact[0]=-w*10000/(1+c);exact[2]=-w*100/c;}
        auto calc=[&](const V5& input){auto r=engine.propagate(bound(input,0,g),*target,o);
          if(!r.ok() || !r->endParameters)throw std::runtime_error("analytic ACTS control failed");
          return vector<4>(state(*r->endParameters,g).at("h"));};
        V4 scales;scales<<1,1,.001,.001;const auto got=calc(x);worst=std::max(worst,((got-exact).array()/scales.array()).abs().maxCoeff());
        if(!zero){auto a=x;a[4]*=-1;wrongSign=((calc(a)-exact).array()/scales.array()).abs().maxCoeff();
          a=x;a[4]*=1000;wrongUnit=((calc(a)-exact).array()/scales.array()).abs().maxCoeff();}
      }
      out.push_back({{"tolerance",tolerance},{"max_scaled_error",worst},{"wrong_sign_error",wrongSign},{"wrong_unit_error",wrongUnit}});
    }
    return out;
  }
  void audit() {
    const auto& ctx=Gaudi::Hive::currentContext();const xAOD::EventInfo* header=nullptr;
    if(evtStore()->retrieve(header,"EventInfo").isFailure() || header->runNumber()!=m_fixture.at("actual_run").get<unsigned>() ||
      header->eventNumber()!=m_fixture.at("actual_event").get<unsigned long long>() || header->runNumber()!=ctx.eventID().run_number() ||
      header->eventNumber()!=ctx.eventID().event_number())throw std::runtime_error("event identity mismatch");
    const auto& p=m_fixture.at("wb96_protocol");
    SG::ReadCondHandle<FaserFieldMapCondObj> mh{m_mapKey,ctx};SG::ReadCondHandle<FaserFieldCacheCondObj> ch{m_cacheKey,ctx};
    if(!mh.isValid() || !ch.isValid() || !mh->fieldMap())throw std::runtime_error("missing actual field");
    const auto g=m_tools[0]->trackingGeometryTool()->getGeometryContext(ctx).context();auto mctx=m_tools[0]->getMagneticFieldContext(ctx);
    if(mctx.get<const FaserFieldCacheCondObj*>()!=ch.cptr())throw std::runtime_error("context/cache mismatch");
    const auto tracking=m_tools[0]->trackingGeometryTool()->trackingGeometry();
    auto field=std::make_shared<WB96Observation::Field>(mh->fieldMap());
    Acts::Navigator::Config navConfig{tracking};
    using Engine=Acts::Propagator<Acts::EigenStepper<>,Acts::Navigator>;
    Engine engine{Acts::EigenStepper<>(field),Acts::Navigator(navConfig),Acts::getDefaultLogger("WB96Engine",Acts::Logging::ERROR)};
    Acts::PropagatorPlainOptions defaults;
    if(defaults.stepTolerance!=p.at("step_tolerances").at(0) || defaults.surfaceTolerance!=p.at("surface_tolerance_mm") ||
      defaults.maxRungeKuttaStepTrials!=p.at("max_runge_kutta_step_trials") || defaults.stepSizeCutOff!=p.at("step_size_cutoff"))throw std::runtime_error("actual defaults changed");
    std::map<uint64_t,Json> surfaceMap;
    V5 seed;seed.head<4>()=vector<4>(m_fixture.at("references").at(0).at("fixed_z_state"));seed[4]=m_fixture.at("references").at(0).at("q_over_p_per_MeV");
    const double z=m_fixture.at("references").at(0).at("z_state_mm");
    V5 steps=vector<5>(p.at("seed_steps"));std::vector<V5> seeds{seed};Json labels=Json::array({Json{{"name","nominal"}}});
    for(int k=0;k<6;++k){V5 d=V5::Zero();if(k<5)d[k]=steps[k];else{d=steps;for(int j=1;j<5;j+=2)d[j]*=-1;}
      for(auto pair:std::vector<std::pair<std::string,double>>{{"central_plus",.5},{"central_minus",-.5},{"full",.25},{"half",.125}}){
        seeds.push_back(seed+pair.second*d);labels.push_back({{"direction",p.at("directions").at(k)},{"name",pair.first}});}}
    EventIDRange mr,cr;if(!mh.range(mr) || !ch.range(cr))throw std::runtime_error("missing field IOV");std::ostringstream mi,ci;mi<<mr;ci<<cr;
    const auto* zone=mh->fieldMap()->findBFieldZone(seed[0],seed[1],-1662.3);if(!zone)throw std::runtime_error("missing actual zone");
    Json min=Json::array(),max=Json::array();for(int k=0;k<3;++k){min.push_back(zone->min(k));max.push_back(zone->max(k));}
    Json out={{"input_xaod",m_fixture.at("input_xaod")},{"ordinal",m_fixture.at("ordinal")},{"actual_run",header->runNumber()},
      {"actual_event",header->eventNumber()},{"seed",encode(seed)},{"seed_z_mm",z},{"defaults",WB96Observation::options(defaults)},
      {"navigator",{{"resolveSensitive",navConfig.resolveSensitive},{"resolveMaterial",navConfig.resolveMaterial},{"resolvePassive",navConfig.resolvePassive}}},
      {"conditions",{{"map_key",m_mapKey.key()},{"cache_key",m_cacheKey.key()},{"map_IOV",mi.str()},{"cache_IOV",ci.str()},
        {"scale",ch->dipoleFieldScaleFactor()},{"zone_id",zone->id()},{"bscale_kT",zone->bscale()},{"min_mm",min},{"max_mm",max},{"wrapper_cache_identity",true}}},
      {"math_control",mathControls(g,mctx)},{"sensors",Json::array()},{"settings",Json::array()}};
    const auto ids=m_tools[0]->trackingGeometryTool()->getIdentifierMap();
    for(const auto& old:m_fixture.at("wb92_targets"))for(const auto& sensor:old.at("sensors")) {
      Identifier strip(static_cast<Identifier::value_type>(sensor.at("strip").get<unsigned long long>()));auto wafer=m_helper->wafer_id(strip);
      if(ids->count(wafer)!=1 || m_helper->station(wafer)!=old.at("station") || wafer.get_compact()!=sensor.at("wafer") || ids->at(wafer).value()!=sensor.at("geometry_id"))throw std::runtime_error("sensor identity changed");
      const auto* s=tracking->findSurface(ids->at(wafer));if(!s || (s->transform(g).matrix()-transform(sensor.at("transform")).matrix()).cwiseAbs().maxCoeff()>1e-9)throw std::runtime_error("sensor transform changed");
      out["sensors"].push_back(sensor);
    }
    auto fixed=vector<5>(m_fixture.at("wb96_fixed_start").at("seed"));const double inner=m_fixture.at("wb96_fixed_start").at("z_mm");
    auto tmp=bound(fixed,inner,g);auto pars=tmp.parameters();pars[Acts::eBoundTime]=m_fixture.at("wb96_fixed_start").at("time_Acts");
    const Acts::BoundTrackParameters fixedStart(tmp.referenceSurface().getSharedPtr(),pars,std::nullopt,Acts::ParticleHypothesis::muon());
    auto call=[&](const Acts::BoundTrackParameters& start,const Acts::Transform3& frame,double tolerance,double cap,bool save,bool verify){
      field->reset(save);WB96Observation::Trace trace;trace.save=save;
      using Options=Acts::PropagatorOptions<Acts::ActionList<Acts::MaterialInteractor,WB96Observation::Action>,Acts::AbortList<Acts::EndOfWorldReached>>;
      Options options(g,mctx);options.stepTolerance=tolerance;options.maxStepSize=cap*Acts::UnitConstants::m;options.maxSteps=p.at("max_steps");
      options.loopProtection=Acts::VectorHelpers::perp(start.momentum())<p.at("pt_loopers_MeV").get<double>()*Acts::UnitConstants::MeV;
      auto& mat=options.actionList.get<Acts::MaterialInteractor>();mat.multipleScattering=false;mat.energyLoss=false;mat.recordInteractions=false;
      auto& action=options.actionList.get<WB96Observation::Action>();action.trace=&trace;action.field=field.get();action.surfaces=&surfaceMap;
      auto plane=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);auto result=engine.propagate(start,*plane,options);
      Json row={{"status","FAIL"},{"options",WB96Observation::options(options)},{"accepted_steps",trace.accepted},
        {"rejected_trials",trace.rejected},{"max_accepted_error_estimate",trace.maxError},{"sensitive_sequence",trace.sensitive},
        {"last_free_state",trace.lastFree},{"field_queries",field->raw},{"accepted_trace",trace.steps},
        {"field_counts",{{"total",field->queries},{"inside",field->inside},{"outside",field->outside},{"gradient_calls",field->gradients}}}};
      if(!result.ok()){row["error_code"]=result.error().value();row["error_message"]=result.error().message();return row;}
      if(!result->endParameters){row["error_message"]="no endParameters";return row;}
      row["status"]="PASS";row["state"]=state(*result->endParameters,g);row["propagator_steps_counter"]=result->steps;row["path_length_mm"]=result->pathLength;
      if(verify){auto official=m_active->propagate(ctx,start,*plane,Acts::Direction::Forward);
        if(official)row["official_default_state"]=state(*official,g);else row["official_default_error"]="nullopt";}
      return row;
    };
    for(const auto& toleranceValue:p.at("step_tolerances"))for(size_t ti=0;ti<m_tools.size();++ti) {
      const double tolerance=toleranceValue,cap=p.at("max_step_sizes_m").at(ti);m_active=m_tools[ti].get();
      if(m_active->getMagneticFieldContext(ctx).get<const FaserFieldCacheCondObj*>()!=ch.cptr())throw std::runtime_error("official tool conditions differ");
      const bool baseline=tolerance==p.at("step_tolerances").at(0);ATH_MSG_INFO("WB96 tolerance="<<tolerance<<" cap_m="<<cap);
      Json setting={{"tolerance",tolerance},{"cap_m",cap},{"targets",Json::array()}};
      Acts::Transform3 entry=Acts::Transform3::Identity();entry.translation().z()=zone->min(2);
      setting["entry_nominal"]=call(bound(seed,z,g),entry,tolerance,cap,true,baseline);
      for(int station=1;station<=3;++station) {
        const auto& old=m_fixture.at("wb92_targets").at(station);const auto frame=transform(old.at("frame"));
        Json target={{"station",station},{"frame",old.at("frame")},{"y",m_fixture.at("references").at(station).at("fixed_z_state")},{"samples",Json::array()}};
        for(size_t n=0;n<seeds.size();++n){bool save=false;for(auto index:p.at("trace_sample_indices"))if(n==index)save=true;
          auto row=call(bound(seeds[n],z,g),frame,tolerance,cap,save,baseline);row["label"]=labels[n];row["seed"]=encode(seeds[n]);target["samples"].push_back(row);}
        target["fixed_reference_start_nominal"]=call(fixedStart,frame,tolerance,cap,true,baseline);target["fixed_start"]=state(fixedStart,g);
        setting["targets"].push_back(target);
      }
      if(baseline) {
        const auto& saved=m_fixture.at("wb96_saved_baseline").at(ti);const V4 scales=vector<4>(p.at("output_scales"));
        auto guard=[&](const Json& row,const Json* historical){
          if(row.at("status")!="PASS" || !row.contains("official_default_state"))throw std::runtime_error("baseline engine/official propagation failed; tightened calls forbidden");
          const auto h=vector<4>(row.at("state").at("h"));
          double difference=((h-vector<4>(row.at("official_default_state").at("h"))).array()/scales.array()).abs().maxCoeff();
          if(historical)difference=std::max(difference,((h-vector<4>(historical->at("h"))).array()/scales.array()).abs().maxCoeff());
          if(difference>p.at("baseline_reproduction_scaled_tolerance").get<double>()) {
            std::ostringstream msg;msg<<"baseline same-path guard failed, scaled difference="<<difference<<"; tightened calls forbidden";throw std::runtime_error(msg.str());}
        };
        guard(setting.at("entry_nominal"),&saved.at("entry"));
        for(size_t st=0;st<3;++st){const auto& target=setting.at("targets").at(st);
          for(size_t n=0;n<25;++n)guard(target.at("samples").at(n),&saved.at("targets").at(st).at(n));
          guard(target.at("fixed_reference_start_nominal"),nullptr);}
      }
      out["settings"].push_back(setting);
    }
    out["observed_surfaces"]=Json::array();for(const auto& [id,s]:surfaceMap)out["observed_surfaces"].push_back(s);
    std::ifstream maps("/proc/self/maps");std::string line;std::set<std::string> libraries;
    while(std::getline(maps,line)){auto slash=line.find('/');if(slash!=std::string::npos && line.find(".so",slash)!=std::string::npos)libraries.insert(line.substr(slash));}
    out["loaded_libraries"]=libraries;
    const int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive result failed");::close(fd);
    std::ofstream stream(m_output.value());stream<<out.dump(2)<<std::endl;if(!stream)throw std::runtime_error("result write failed");
  }
