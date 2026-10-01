  SG::ReadCondHandleKey<FaserFieldMapCondObj> m_mapKey{this,"MapKey","fieldMapCondObj","actual map"};
  SG::ReadCondHandleKey<FaserFieldCacheCondObj> m_cacheKey{this,"CacheKey","fieldCondObj","actual cache"};
  void audit() {
    const auto& ctx=Gaudi::Hive::currentContext();const xAOD::EventInfo* header=nullptr;
    if(evtStore()->retrieve(header,"EventInfo").isFailure() ||
       header->runNumber()!=m_fixture.at("actual_run").get<unsigned>() ||
       header->eventNumber()!=m_fixture.at("actual_event").get<unsigned long long>() ||
       header->runNumber()!=ctx.eventID().run_number() || header->eventNumber()!=ctx.eventID().event_number())
      throw std::runtime_error("actual event identity mismatch");
    const auto& p=m_fixture.at("wb95_protocol");
    SG::ReadCondHandle<FaserFieldMapCondObj> mh{m_mapKey,ctx};SG::ReadCondHandle<FaserFieldCacheCondObj> ch{m_cacheKey,ctx};
    if(!mh.isValid() || !ch.isValid() || !mh->fieldMap())throw std::runtime_error("missing field conditions");
    const auto* map=mh->fieldMap();EventIDRange mr,cr;if(!mh.range(mr) || !ch.range(cr))throw std::runtime_error("missing field IOV");
    std::ostringstream mi,ci;mi<<mr;ci<<cr;
    const auto g=m_tools[0]->trackingGeometryTool()->getGeometryContext(ctx).context();
    const auto mctx=m_tools[0]->getMagneticFieldContext(ctx);
    if(mctx.get<const FaserFieldCacheCondObj*>()!=ch.cptr())throw std::runtime_error("wrapper context differs");
    V5 seed;seed.head<4>()=vector<4>(m_fixture.at("references").at(0).at("fixed_z_state"));seed[4]=m_fixture.at("references").at(0).at("q_over_p_per_MeV");
    const double z=m_fixture.at("references").at(0).at("z_state_mm"),edge=p.at("expected_zone").at("min_mm").at(2);
    const double inner=edge+p.at("interior_restart_offset_mm").get<double>(),scale=ch->dipoleFieldScaleFactor();
    const auto* zone=map->findBFieldZone(seed[0],seed[1],edge+1);
    if(!zone || zone->id()!=p.at("expected_zone").at("id") || scale!=p.at("expected_scale") || zone->bscale()!=p.at("expected_bscale_kT"))throw std::runtime_error("zone/scale identity changed");
    std::array<std::vector<double>,3> mesh;Json minimum=Json::array(),maximum=Json::array(),meshJson=Json::array();
    for(int axis=0;axis<3;++axis) {
      minimum.push_back(zone->min(axis));maximum.push_back(zone->max(axis));
      if(zone->nmesh(axis)!=p.at("expected_mesh_dimensions").at(axis))throw std::runtime_error("mesh dimensions changed");
      for(unsigned j=0;j<zone->nmesh(axis);++j)mesh[axis].push_back(zone->mesh(axis,j));
      meshJson.push_back(mesh[axis]);
    }
    if(minimum!=p.at("expected_zone").at("min_mm") || maximum!=p.at("expected_zone").at("max_mm") || zone->nfield()!=p.at("expected_nodes"))throw std::runtime_error("map payload structure changed");
    const std::string nodesPath=m_output.value()+".nodes_le_i16.bin";
    const int nf=::open(nodesPath.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(nf<0)throw std::runtime_error("exclusive nodes export failed");::close(nf);
    std::ofstream nodeStream(nodesPath,std::ios::binary);
    for(unsigned j=0;j<zone->nfield();++j)for(int k=0;k<3;++k){const auto raw=static_cast<unsigned short>(zone->field(j)[k]);
      const char bytes[2]={static_cast<char>(raw&255),static_cast<char>(raw>>8)};nodeStream.write(bytes,2);}
    nodeStream.close();if(!nodeStream)throw std::runtime_error("nodes export failed");
    FASERMagneticFieldWrapper field;
    auto floatField=[&](const Acts::Vector3& pos,auto& cache)->Acts::Vector3 {
      auto b=field.getField(pos,cache);if(!b.ok())throw std::runtime_error("official field query failed");return b.value();};
    auto doubleField=[&](const Acts::Vector3& pos)->Acts::Vector3 {
      const auto* q=map->findBFieldZone(pos.x(),pos.y(),pos.z());
      if(!q)return Acts::Vector3(1e-5,1e-5,1e-5)*Acts::UnitConstants::T;
      if(q!=zone)throw std::runtime_error("different actual zone");
      double xyz[3]={pos.x(),pos.y(),pos.z()},b[3];q->getB(xyz,b);
      return Acts::Vector3(b[0],b[1],b[2])*scale*1000.*Acts::UnitConstants::T;};
    auto cell=[&](const Acts::Vector3& pos){std::array<int,3> index;
      for(int axis=0;axis<3;++axis){auto it=std::lower_bound(mesh[axis].begin(),mesh[axis].end(),pos[axis]);
        index[axis]=std::max(0,std::min(static_cast<int>(mesh[axis].size())-2,static_cast<int>(it-mesh[axis].begin())-1));}
      return index;};
    auto corners=[&](const std::array<int,3>& ix){WB95Math::Corners nodes;
      for(int c=0;c<8;++c){int x=ix[0]+((c>>2)&1),y=ix[1]+((c>>1)&1),zz=ix[2]+(c&1);
        const unsigned j=(x*mesh[1].size()+y)*mesh[2].size()+zz;
        for(int k=0;k<3;++k)nodes[c][k]=zone->field(j)[k];}
      return nodes;};
    auto cache=field.makeCache(mctx);
    auto probe=[&](const Acts::Vector3& pos){const auto* q=map->findBFieldZone(pos.x(),pos.y(),pos.z());
      const Acts::Vector3 fb=floatField(pos,cache)/Acts::UnitConstants::T,db=doubleField(pos)/Acts::UnitConstants::T;
      Json row={{"position_mm",encode(pos)},{"zone_id",q?q->id():-1},{"float_T",encode(fb)},{"double_T",encode(db)}};
      if(q){const auto ix=cell(pos);WB95Math::V fraction;
        for(int a=0;a<3;++a)fraction[a]=(pos[a]-mesh[a][ix[a]])/(mesh[a][ix[a]+1]-mesh[a][ix[a]]);
        const auto nodes=corners(ix);
        const auto independent=WB95Math::trilinear(nodes,fraction,zone->bscale()*scale*1000.);
        row["cell"]=ix;row["fractions"]=fraction;row["nodes"]=nodes;row["independent_double_T"]=independent;
      }
      return row;};
    const auto math=WB94Reference::controls(Acts::UnitConstants::T);
    Json out={{"input_xaod",m_fixture.at("input_xaod")},{"ordinal",m_fixture.at("ordinal")},{"actual_run",header->runNumber()},
      {"actual_event",header->eventNumber()},{"seed",encode(seed)},{"seed_z_mm",z},
      {"conditions",{{"map_key",m_mapKey.key()},{"cache_key",m_cacheKey.key()},{"map_IOV",mi.str()},{"cache_IOV",ci.str()},
        {"wrapper_cache_identity",true},{"scale",scale},{"bscale_kT",zone->bscale()},{"zone_id",zone->id()},
        {"min_mm",minimum},{"max_mm",maximum}}},{"mesh_mm",meshJson},{"node_count",zone->nfield()},
      {"nodes_binary",nodesPath},{"math_control",{{"max_scaled_error",math[0]},{"wrong_sign_error",math[1]},{"wrong_unit_error",math[2]}}},
      {"probes",Json::array()},{"modes",Json::array()},{"sensors",Json::array()}};
    for(double zz:mesh[2])for(const auto& d:p.at("probe_offsets_mm"))out["probes"].push_back(probe(Acts::Vector3(seed[0],seed[1],zz+d.get<double>())));
    for(size_t j=1;j<mesh[2].size();++j)out["probes"].push_back(probe(Acts::Vector3(seed[0],seed[1],.5*(mesh[2][j-1]+mesh[2][j]))));
    const auto si=cell(Acts::Vector3(seed[0],seed[1],inner));
    for(int axis=0;axis<2;++axis)for(int j:{si[axis],si[axis]+1})for(const auto& d:p.at("probe_offsets_mm")) {
      Acts::Vector3 pos(seed[0],seed[1],inner);pos[axis]=mesh[axis][j]+d.get<double>();out["probes"].push_back(probe(pos));}
    out["domain_controls"]=Json::array();
    for(const auto& pos:std::vector<Acts::Vector3>{Acts::Vector3(seed[0],seed[1],z),Acts::Vector3(seed[0],seed[1],inner),
        Acts::Vector3(edge+1,seed[0],seed[1]),Acts::Vector3(201,seed[1],inner),Acts::Vector3(seed[0],201,inner)})out["domain_controls"].push_back(probe(pos));
    const auto geometry=m_tools[0]->trackingGeometryTool();const auto ids=geometry->getIdentifierMap();
    for(const auto& old:m_fixture.at("wb92_targets"))for(const auto& sensor:old.at("sensors")) {
      const Identifier strip(static_cast<Identifier::value_type>(sensor.at("strip").get<unsigned long long>()));const auto wafer=m_helper->wafer_id(strip);
      if(ids->count(wafer)!=1 || m_helper->station(wafer)!=old.at("station") || wafer.get_compact()!=sensor.at("wafer") || ids->at(wafer).value()!=sensor.at("geometry_id"))throw std::runtime_error("sensitive identity mismatch");
      const auto* surface=geometry->trackingGeometry()->findSurface(ids->at(wafer));
      if(!surface || (surface->transform(g).matrix()-transform(sensor.at("transform")).matrix()).cwiseAbs().maxCoeff()>1e-9)throw std::runtime_error("sensitive geometry changed");
      out["sensors"].push_back({{"wafer",wafer.get_compact()},{"transform",encode(surface->transform(g).matrix())}});
    }
    const V5 xs=vector<5>(p.at("seed_steps"));std::vector<V5> seeds{seed};Json labels=Json::array({Json{{"name","nominal"}}});
    for(int k=0;k<6;++k){V5 d=V5::Zero();if(k<5)d[k]=xs[k];else{d=xs;for(int a=1;a<5;a+=2)d[a]*=-1;}
      for(const auto& pair:std::vector<std::pair<std::string,double>>{{"central_plus",.5},{"central_minus",-.5},{"full",.25},{"half",.125}}){
        seeds.push_back(seed+pair.second*d);labels.push_back({{"direction",p.at("directions").at(k)},{"name",pair.first}});}}
    for(const auto& modeValue:p.at("modes")) {
      const std::string mode=modeValue;const bool meshSplit=mode!="original_float",useDouble=mode=="mesh_z_double";
      Json modeResult={{"name",mode},{"ladders",Json::array()}};
      const auto& steps=p.at(meshSplit?"mesh_rk4_dz_mm":"original_rk4_dz_mm");
      for(const auto& step:steps) {
        ATH_MSG_INFO("WB95 mode="<<mode<<" dz_mm="<<step);
        Json ladder={{"dz_mm",step},{"samples",Json::array()}};
        for(size_t n=0;n<seeds.size();++n) {
          const auto x=seeds[n];WB94Reference::State r{x[0],x[1],x[2],x[3],0};double current=z;bool before=true;
          auto localCache=field.makeCache(mctx);size_t outside=0,inside=0,transverseChanges=0;std::array<int,3> previous{-1,-1,-1};
          double maxFieldDifference=0.;const double eps=p.at("boundary_side_epsilon_mm");
          auto get=[&](double xx,double yy,double zz){if(std::abs(zz-edge)<eps/2)zz=edge+(before?-eps:eps);
            const Acts::Vector3 pos(xx,yy,zz);const auto* q=map->findBFieldZone(xx,yy,zz);
            if(before){++outside;if(q)throw std::runtime_error("reference pre-entry in map");}
            else {++inside;if(q!=zone)throw std::runtime_error("reference leaves actual zone");
              const auto ix=cell(pos);if(previous[0]>=0 && (previous[0]!=ix[0] || previous[1]!=ix[1]))++transverseChanges;previous=ix;}
            const Acts::Vector3 f=floatField(pos,localCache),d=doubleField(pos);maxFieldDifference=std::max(maxFieldDifference,(f-d).cwiseAbs().maxCoeff()/Acts::UnitConstants::T);
            const Acts::Vector3 b=useDouble?d:f;return WB94Reference::Field{b.x(),b.y(),b.z()};};
          auto integrateTo=[&](double to){
            if(meshSplit)r=WB95Reference::integrate(r,current,to,step.get<double>(),x[4]/Acts::UnitConstants::MeV,Acts::ParticleHypothesis::muon().mass(),get);
            else r=WB94Reference::integrate(r,current,to,step.get<double>(),x[4]/Acts::UnitConstants::MeV,Acts::ParticleHypothesis::muon().mass(),get);
            current=to;};
          auto go=[&](double to){if(meshSplit && current>=edge) {
              for(auto it=std::upper_bound(mesh[2].begin(),mesh[2].end(),current);it!=mesh[2].end() && *it<to;++it)integrateTo(*it);
            }
            integrateTo(to);V4 h;for(int a=0;a<4;++a)h[a]=r[a];
            return Json{{"z_mm",to},{"h",encode(h)},{"time_Acts",r[4]},{"q_over_p_per_MeV",x[4]}};};
          Json row={{"label",labels[n]},{"seed",encode(x)},{"entry",go(edge)}};before=false;row["interior"]=go(inner);row["targets"]=Json::array();
          for(const auto& ref:m_fixture.at("references"))if(ref.at("station")!=0)row["targets"].push_back(go(ref.at("z_state_mm")));
          row["outside_stages"]=outside;row["inside_stages"]=inside;row["transverse_stage_cell_transitions"]=transverseChanges;
          row["max_same_point_field_difference_T"]=maxFieldDifference;ladder["samples"].push_back(row);
        }
        modeResult["ladders"].push_back(ladder);
      }
      out["modes"].push_back(modeResult);
    }
    std::ifstream maps("/proc/self/maps");std::string line;std::set<std::string> libraries;
    while(std::getline(maps,line)){auto slash=line.find('/');if(slash!=std::string::npos && line.find(".so",slash)!=std::string::npos)libraries.insert(line.substr(slash));}
    out["loaded_libraries"]=libraries;
    const int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive result failed");::close(fd);
    std::ofstream stream(m_output.value());stream<<out.dump(2)<<std::endl;if(!stream)throw std::runtime_error("result write failed");
  }
