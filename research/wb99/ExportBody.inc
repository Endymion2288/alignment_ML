  void doubling(const Json& current,const Acts::GeometryContext& g,const Acts::MagneticFieldContext& mctx,const MagField::FaserFieldMap* map) {
    static_assert(std::is_trivially_copyable_v<MagField::FaserFieldCache>,"byte snapshot requires trivially copyable cache");
    const auto& p=m_fixture.at("wb99_protocol");
    std::ifstream input(m_fixture.at("wb99_saved_raw").get<std::string>());Json old;input>>old;if(!input)throw std::runtime_error("WB99 historical raw read");
    double worst=0,fieldWorst=0;size_t defaultCount=0,historicalCount=0;
    const V4 scales=vector<4>(m_fixture.at("wb96_protocol").at("output_scales"));
    auto guard=[&](const Json& a,const Json& b,bool baseline){
      if(a.at("status")!="PASS"||b.at("status")!="PASS"||a.at("accepted_steps")!=b.at("accepted_steps")||a.at("rejected_trials")!=b.at("rejected_trials")||
        a.at("sensitive_sequence")!=b.at("sensitive_sequence")||a.at("field_counts")!=b.at("field_counts"))throw std::runtime_error("WB99 propagation identity changed");
      worst=std::max(worst,((vector<4>(a.at("state").at("h"))-vector<4>(b.at("state").at("h"))).array()/scales.array()).abs().maxCoeff());++historicalCount;
      if(baseline){if(!a.contains("official_default_state"))throw std::runtime_error("missing official default");
        worst=std::max(worst,((vector<4>(a.at("state").at("h"))-vector<4>(a.at("official_default_state").at("h"))).array()/scales.array()).abs().maxCoeff());++defaultCount;}
      if(a.at("accepted_trace").size()!=b.at("accepted_trace").size()||a.at("field_queries").size()!=b.at("field_queries").size())throw std::runtime_error("WB99 saved trace count");
      for(size_t k=0;k<a.at("accepted_trace").size();++k){const auto& x=a.at("accepted_trace").at(k);const auto& y=b.at("accepted_trace").at(k);
        if(x.at("query_begin")!=y.at("query_begin")||x.at("query_end")!=y.at("query_end")||x.at("rejected_trials")!=y.at("rejected_trials")||
          (vector<3>(x.at("position_mm"))-vector<3>(y.at("position_mm"))).cwiseAbs().maxCoeff()>p.at("closure_position_mm").get<double>()||
          (vector<3>(x.at("direction"))-vector<3>(y.at("direction"))).cwiseAbs().maxCoeff()>p.at("closure_direction").get<double>()||
          std::abs(x.at("h_mm").get<double>()-y.at("h_mm").get<double>())>p.at("closure_position_mm").get<double>())throw std::runtime_error("WB99 all-step default source changed");}
      for(size_t k=0;k<a.at("field_queries").size();++k){const auto& x=a.at("field_queries").at(k);const auto& y=b.at("field_queries").at(k);
        fieldWorst=std::max(fieldWorst,(vector<3>(x.at("field_T"))-vector<3>(y.at("field_T"))).cwiseAbs().maxCoeff());
        if(x.at("zone_id")!=y.at("zone_id")||(vector<3>(x.at("position_mm"))-vector<3>(y.at("position_mm"))).cwiseAbs().maxCoeff()>p.at("closure_position_mm").get<double>())throw std::runtime_error("WB99 all-query position changed");}
    };
    for(size_t i=0;i<current.at("settings").size();++i){const auto& a=current.at("settings").at(i);const auto& b=old.at("settings").at(i);
      if(a.at("tolerance")!=b.at("tolerance")||a.at("cap_m")!=b.at("cap_m"))throw std::runtime_error("WB99 setting mismatch");
      const bool base=a.at("tolerance")==1e-4;guard(a.at("entry_nominal"),b.at("entry_nominal"),base);
      for(size_t st=0;st<3;++st){const auto& x=a.at("targets").at(st);const auto& y=b.at("targets").at(st);
        for(size_t n=0;n<25;++n)guard(x.at("samples").at(n),y.at("samples").at(n),base);
        guard(x.at("fixed_reference_start_nominal"),y.at("fixed_reference_start_nominal"),base);}}
    if(defaultCount!=316||historicalCount!=948||worst>p.at("official_reproduction_scaled").get<double>()||fieldWorst>p.at("field_replay_T").get<double>())throw std::runtime_error("WB99 full reproduction budget");
    const std::string dest=m_output.value()+".doubling.ndjson";int fd=::open(dest.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);if(fd<0)throw std::runtime_error("exclusive WB99 export");::close(fd);std::ofstream stream(dest);
    stream<<Json{{"record","fidelity"},{"historical_calls",historicalCount},{"official_calls",defaultCount},{"scaled_max",worst},{"field_max_T",fieldWorst},{"gate","PASS"}}.dump()<<'\n';
    FASERMagneticFieldWrapper wrapper;size_t traces=0,total=0;double replayMax=0;
    auto analyze=[&](const Json& setting,const Json& target,const Json& call,const std::string& path,int station,int sample){
      Acts::BoundTrackParameters start=path=="fixed_start"?Acts::BoundTrackParameters(Acts::Surface::makeShared<Acts::PlaneSurface>(WB99::V(0,0,vector<3>(target.at("fixed_start").at("position_mm"))[2]),WB99::V(0,0,1)),
        vector<6>(target.at("fixed_start").at("bound_parameters")),std::nullopt,Acts::ParticleHypothesis::muon()):bound(path=="entry"?vector<5>(old.at("seed")):vector<5>(call.at("seed")),old.at("seed_z_mm"),g);
      auto history=wrapper.makeCache(mctx);WB99::V pos=start.position(g),dir=start.direction();const double q=start.parameters()[Acts::eBoundQOverP];size_t cursor=0,index=0;
      auto replay=[&](size_t until){while(cursor<until){const auto& query=call.at("field_queries").at(cursor++);auto got=wrapper.getField(vector<3>(query.at("position_mm")),history);
          if(!got.ok())throw std::runtime_error("WB99 history field error");replayMax=std::max(replayMax,(*got/Acts::UnitConstants::T-vector<3>(query.at("field_T"))).cwiseAbs().maxCoeff());}};
      auto snapshot=[&](auto& c){std::memcpy(&c.template as<FASERMagneticFieldWrapper::Cache>().fieldCache,&history.template as<FASERMagneticFieldWrapper::Cache>().fieldCache,sizeof(MagField::FaserFieldCache));};
      for(const auto& saved:call.at("accepted_trace")){
        replay(saved.at("query_begin"));auto fullCache=wrapper.makeCache(mctx),halfCache=wrapper.makeCache(mctx);snapshot(fullCache);snapshot(halfCache);
        auto get=[&](auto& cache,const WB99::V& x){auto b=wrapper.getField(x,cache);if(!b.ok())throw std::runtime_error("WB99 new stage field error");return WB99::V(*b);};
        const double h=saved.at("h_mm");auto full=WB99::step(pos,dir,h,q,[&](const auto& x){return get(fullCache,x);});
        auto half1=WB99::step(pos,dir,h/2,q,[&](const auto& x){return get(halfCache,x);});
        auto half2=WB99::step(half1.p,half1.u,h/2,q,[&](const auto& x){return get(halfCache,x);});
        const size_t first=saved.at("query_begin").get<size_t>()+(index==0?int(call.at("options").at("loopProtection").get<bool>()):0);
        replay(saved.at("query_end").get<size_t>()-2);auto acceptedCache=wrapper.makeCache(mctx);snapshot(acceptedCache);bool firstQuery=true;
        auto accepted=WB99::step(pos,dir,h,q,[&](const WB99::V& x){if(firstQuery){firstQuery=false;return WB99::V(vector<3>(call.at("field_queries").at(first).at("field_T"))*Acts::UnitConstants::T);}return get(acceptedCache,x);});
        const auto ep=vector<3>(saved.at("position_mm")),eu=vector<3>(saved.at("direction"));
        Json row={{"record","step"},{"trace",traces},{"index",index},{"tolerance",setting.at("tolerance")},{"cap_m",setting.at("cap_m")},{"path",path},{"station",station},{"sample",sample},
          {"start_position_mm",WB99::array(pos)},{"start_direction",WB99::array(dir)},{"q_over_p_Acts",q},{"h_mm",h},{"full",WB99::save(full)},{"half1",WB99::save(half1)},{"half2",WB99::save(half2)},{"accepted_branch",WB99::save(accepted)},
          {"full_position_closure_mm",(full.p-ep).cwiseAbs().maxCoeff()},{"full_direction_closure",(full.u-eu).cwiseAbs().maxCoeff()},
          {"accepted_position_closure_mm",(accepted.p-ep).cwiseAbs().maxCoeff()},{"accepted_direction_closure",(accepted.u-eu).cwiseAbs().maxCoeff()}};
        for(const std::string key:{"full","half1","half2","accepted_branch"})for(auto& query:row[key]["queries"]){const auto x=vector<3>(query["position_mm"]);const auto* zone=map->findBFieldZone(x[0],x[1],x[2]);query["zone_id"]=zone?zone->id():-1;}
        stream<<row.dump()<<'\n';++index;++total;replay(saved.at("query_end"));pos=ep;dir=eu;
      }
      stream.flush();ATH_MSG_INFO("WB99 trace="<<traces<<" steps="<<index);++traces;
    };
    for(const auto& setting:old.at("settings")){analyze(setting,Json::object(),setting.at("entry_nominal"),"entry",0,0);
      for(const auto& target:setting.at("targets")){for(int sample:p.at("trace_sample_indices"))analyze(setting,target,target.at("samples").at(sample),"direct",target.at("station"),sample);
        analyze(setting,target,target.at("fixed_reference_start_nominal"),"fixed_start",target.at("station"),0);}}
    if(traces!=p.at("expected_traces")||total!=p.at("expected_steps")||replayMax>p.at("field_replay_T").get<double>())throw std::runtime_error("WB99 coverage/cache replay");
    stream<<Json{{"record","terminal"},{"traces",traces},{"steps",total},{"replay_max_T",replayMax},{"qualification","NOT_EVALUATED"}}.dump()<<'\n';if(!stream)throw std::runtime_error("WB99 stream write");
  }
