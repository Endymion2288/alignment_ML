"""Versioned, checked transformations of frozen diagnostic components."""
from wb101_sources import files as parent_files, replace_one
from wb95_contract import generated_source as reference_source


def files():
    result = parent_files()
    direction = result.pop('WB101Diagnostic/DirectionAcceptanceAudit.cxx')
    direction = replace_one(direction, 'namespace WB101 {', 'namespace WB102 {')
    direction = direction.replace('DirectionAcceptanceAudit', 'JacobianScaleAudit')
    direction = direction.replace('WB101::JacobianScaleAudit', 'WB102::JacobianScaleAudit')
    direction = replace_one(direction, 'V5 steps=vector<5>(p.at("seed_steps"));',
                            'V5 steps=vector<5>(p.at("seed_steps"))*m_fixture.at("wb102_lambda").get<double>();')
    direction = replace_one(direction, 'if(historical)difference=',
                            'if(historical && m_fixture.at("wb102_lambda")==1.)difference=')
    direction = replace_one(direction, '        structure(setting.at("entry_nominal"));',
                            '        if(m_fixture.at("wb102_lambda")==1.) {\n        structure(setting.at("entry_nominal"));')
    direction = replace_one(direction, 'if(guardIndex!=79)throw std::runtime_error("WB101 disabled guard population");',
                            'if(guardIndex!=79)throw std::runtime_error("WB101 disabled guard population");\n        }')
    direction = replace_one(direction, '    out["wb101_calls"]=callId;',
                            '    out["wb101_calls"]=callId;out["wb102_lambda"]=m_fixture.at("wb102_lambda");')
    reference = reference_source().replace('namespace WB95', 'namespace WB102Reference')
    reference = reference.replace('FieldPrecisionAudit', 'ScaleReferenceAudit').replace(
        'WB95::ScaleReferenceAudit', 'WB102Reference::ScaleReferenceAudit')
    reference = replace_one(reference, 'const V5 xs=vector<5>(p.at("seed_steps"));',
                            'const V5 xs=vector<5>(p.at("seed_steps"))*m_fixture.at("wb102_lambda").get<double>();')
    reference = replace_one(reference, '    out["loaded_libraries"]=libraries;',
                            '    out["loaded_libraries"]=libraries;out["wb102_lambda"]=m_fixture.at("wb102_lambda");')
    result = {key.replace('WB101Diagnostic', 'WB102Diagnostic'): value for key, value in result.items()}
    result['CMakeLists.txt'] = result['CMakeLists.txt'].replace('WB101', 'WB102')
    cmake = result['WB102Diagnostic/CMakeLists.txt'].replace('WB101Diagnostic', 'WB102Diagnostic')
    cmake = replace_one(cmake, 'DirectionAcceptanceAudit.cxx', 'JacobianScaleAudit.cxx ScaleReferenceAudit.cxx')
    result['WB102Diagnostic/CMakeLists.txt'] = cmake
    result['WB102Diagnostic/JacobianScaleAudit.cxx'] = direction
    result['WB102Diagnostic/ScaleReferenceAudit.cxx'] = reference
    return result
