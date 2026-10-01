  void audit() {
    const auto& ctx=Gaudi::Hive::currentContext();
    const xAOD::EventInfo* header=nullptr;
    if(evtStore()->retrieve(header,"EventInfo").isFailure())throw std::runtime_error("missing actual EventInfo");
    if(header->runNumber()!=m_fixture.at("actual_run").get<unsigned>() ||
       header->eventNumber()!=m_fixture.at("actual_event").get<unsigned long long>() ||
       header->runNumber()!=ctx.eventID().run_number() || header->eventNumber()!=ctx.eventID().event_number())
      throw std::runtime_error("actual source event header mismatch");
    const auto g=m_tools[0]->trackingGeometryTool()->getGeometryContext(ctx).context();
    const auto& p=m_fixture.at("wb93_protocol");
    V5 seed;seed.head<4>()=vector<4>(m_fixture.at("references").at(0).at("fixed_z_state"));
    seed[4]=m_fixture.at("references").at(0).at("q_over_p_per_MeV");
    const double z=m_fixture.at("references").at(0).at("z_state_mm");
    const V5 xs=vector<5>(p.at("seed_steps"));
    Json out={{"input_xaod",m_fixture.at("input_xaod")},{"ordinal",m_fixture.at("ordinal")},
      {"actual_run",header->runNumber()},{"actual_event",header->eventNumber()},
      {"seed",encode(seed)},{"seed_z_mm",z},{"caps",Json::array()}};
    if(m_tools.size()!=p.at("max_step_sizes_m").size())throw std::runtime_error("tool/cap multiplicity mismatch");
    for(size_t toolIndex=0;toolIndex<m_tools.size();++toolIndex) {
      m_active=m_tools[toolIndex].get();
      FASERMagneticFieldWrapper field;auto cache=field.makeCache(m_active->getMagneticFieldContext(ctx));
      Json cap={{"max_step_size_m",p.at("max_step_sizes_m").at(toolIndex)},
        {"tool_name",m_tools[toolIndex].name()},{"targets",Json::array()}};
      for(const auto& ref:m_fixture.at("references")) {
        const int station=ref.at("station");if(station==0)continue;
        Acts::Transform3 frame=Acts::Transform3::Identity();frame.translation().z()=ref.at("z_state_mm");
        const auto& baseline=m_fixture.at("wb92_targets").at(station);
        if((frame.matrix()-transform(baseline.at("frame")).matrix()).cwiseAbs().maxCoeff()>1e-12)
          throw std::runtime_error("frozen reference frame mismatch");
        const auto geometry=m_active->trackingGeometryTool();
        const auto ids=geometry->getIdentifierMap();
        Json sensors=Json::array();
        for(const auto& sensor:baseline.at("sensors")) {
          Identifier strip(static_cast<Identifier::value_type>(sensor.at("strip").get<unsigned long long>()));
          const auto wafer=m_helper->wafer_id(strip);
          if(m_helper->station(wafer)!=station || ids->count(wafer)!=1 ||
             wafer.get_compact()!=sensor.at("wafer") || ids->at(wafer).value()!=sensor.at("geometry_id"))
            throw std::runtime_error("actual sensor identity changed");
          const auto* surface=geometry->trackingGeometry()->findSurface(ids->at(wafer));
          if(!surface)throw std::runtime_error("missing actual sensor surface");
          const Acts::Transform3 actual=surface->transform(g);
          if((actual.matrix()-transform(sensor.at("transform")).matrix()).cwiseAbs().maxCoeff()>1e-9)
            throw std::runtime_error("absolute sensor geometry differs from WB92");
          sensors.push_back({{"wafer",wafer.get_compact()},{"transform",encode(actual.matrix())}});
        }
        auto h=[&](const V5& x){return prediction(x,z,frame,ctx,g);};
        const V4 nominal=h(seed);
        Json target={{"station",station},{"frame",encode(frame.matrix())},{"sensors",sensors},
          {"y",ref.at("fixed_z_state")},{"h",encode(nominal)},{"repeat_h",encode(h(seed))},
          {"directions",Json::array()},{"traces",Json::array()}};
        for(int directionIndex=0;directionIndex<6;++directionIndex) {
          V5 d=V5::Zero();
          if(directionIndex<5)d[directionIndex]=xs[directionIndex];
          else {d=xs;for(int k=1;k<5;k+=2)d[k]*=-1;}
          Json direction={{"name",p.at("directions").at(directionIndex)},{"base_delta",encode(d)},{"samples",Json::array()}};
          for(const auto& value:p.at("step_multipliers")) {
            const double s=value;const V5 delta=s*d;
            direction["samples"].push_back({{"multiplier",s},
              {"central_plus",encode(h(seed+0.5*delta))},{"central_minus",encode(h(seed-0.5*delta))},
              {"full",encode(h(seed+0.25*delta))},{"half",encode(h(seed+0.125*delta))}});
          }
          target["directions"].push_back(direction);
        }
        V5 mixed=xs;for(int k=1;k<5;k+=2)mixed[k]*=-1;
        const V5 probe=seed+0.25*mixed;
        target["repeat_mixed_full"]=encode(h(probe));
        for(int traceIndex=0;traceIndex<3;++traceIndex) {
          const V5 x=seed+(traceIndex==0?0.:traceIndex==1?0.25:0.125)*mixed;
          const auto start=bound(x,z,g);auto plane=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);
          const auto steps=m_active->propagationSteps(ctx,start,*plane,Acts::Direction::Forward);
          Json trace={{"name",p.at("trace_variants").at(traceIndex)},{"steps",Json::array()}};
          for(const auto& step:steps.first) {
            auto b=field.getField(step.position,cache);
            if(!b.ok())throw std::runtime_error("trace field query failed");
            trace["steps"].push_back({{"position_mm",encode(step.position)},
              {"momentum_Acts",encode(step.momentum)},{"geometry_id",step.geoID.value()},
              {"field_T",encode(b.value()/Acts::UnitConstants::T)}});
          }
          const V4 end=h(x);const Acts::Vector3 endpoint=frame*Acts::Vector3(end[0],end[1],0);
          auto b=field.getField(endpoint,cache);if(!b.ok())throw std::runtime_error("endpoint field failed");
          trace["endpoint"]={{"position_mm",encode(endpoint)},{"h",encode(end)},
            {"field_T",encode(b.value()/Acts::UnitConstants::T)}};
          if(steps.first.empty())throw std::runtime_error("official trace empty");
          target["traces"].push_back(trace);
        }
        cap["targets"].push_back(target);
      }
      out["caps"].push_back(cap);
    }
    std::ifstream maps("/proc/self/maps");std::string line;std::set<std::string> libraries;
    while(std::getline(maps,line)) {const auto slash=line.find('/');
      if(slash!=std::string::npos && line.find(".so",slash)!=std::string::npos)libraries.insert(line.substr(slash));}
    out["loaded_libraries"]=libraries;
    const int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0)throw std::runtime_error("exclusive result creation failed");
    ::close(fd);
    std::ofstream result(m_output.value());result<<out.dump(2)<<std::endl;
    if(!result)throw std::runtime_error("result write failed");
  }
