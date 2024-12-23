import cvxpy as cp
import numpy as np
from Read_file import gbv

def Constraint(sys, Province, power_sys):
    constraint = [
        # 需求约束
        sys.ToELEC()[1:] >= gbv.demand.loc[Province, 2025:].values * 3600,
        
        
        # cp.bmat(sys.unit_elec())[0, 2:] <= cp.bmat(sys.unit_elec1())[0, 2:],
        # cp.bmat(sys.unit_elec())[0, 2:] >= cp.bmat(sys.unit_elec1())[0, 2:] * 0.3,
        # sys.CoalUnit.QSELEC >= 0.4 * sys.CoalUnit.QCAP * sys.CoalUnit.CAPACT * sys.CoalUnit.AF,
        # sys.CoalUnit.QSELEC <= cp.multiply(sys.CoalUnit.QCAP, [sys.CoalUnit.AF] * 2 + [sys.CoalUnit.AF * 0.9 ** i for i in range(len(sys.multiperiods) - 2)]) * sys.CoalUnit.CAPACT,
        # sys.CoalUnit.QSELEC >= cp.multiply(sys.CoalUnit.QCAP, [sys.CoalUnit.AF] * 2 + [sys.CoalUnit.AF * 0.7 ** i for i in range(len(sys.multiperiods) - 2)]) * sys.CoalUnit.CAPACT,
        
        cp.bmat(sys.unit_elec())[0, :2] == cp.bmat(sys.unit_elec1())[0, :2],
        cp.bmat(sys.unit_elec())[1:, :] == cp.bmat(sys.unit_elec1())[1:, :],
       
        # 机组退役 + 新建
        sys.CoalUnit.QCAP_ret == sys.CoalUnit.QCAP[0] * cp.cumsum(cp.hstack([0] + list(gbv.retire_coal.loc[:, Province]))) +\
            cp.cumsum(cp.hstack([0] * int(sys.CoalUnit.tl/5) + [sys.CoalUnit.QCAP_new[i] for i in range(int(len(sys.multiperiods)-sys.CoalUnit.tl/5))])) +\
            cp.hstack(list(cp.cumsum(sys.CoalUnit.QCAP_EarlyRet))[:list(gbv.retire_coal.loc[:, Province].cumsum().round(decimals=15)).index(1) + 1] + [0.] * (9-1-list(gbv.retire_coal.loc[:, Province].cumsum().round(decimals=15)).index(1))) +\
            cp.cumsum(sys.CoalUnit.QCAP_EarlyRet_new),
        sys.CoalUnit.QCAP_ret[0] <= [0], 
        # sys.CoalUnit.QCAP_EarlyRet[1:3] + sys.CoalUnit.QCAP_EarlyRet_new[1:3] <= sys.CoalUnit.QCAP[1:3] * cp.hstack([0.01, 0.1]),   # 提前退役约束
        sys.CoalUnit.QCAP_EarlyRet[list(gbv.retire_coal.loc[:, Province].cumsum().round(decimals=15)).index(1) + 1:] <= 0,
        sys.CoalUnit.QCAP_EarlyRet_new[1:] <= cp.sum(cp.bmat([([0.] * (i+1) +\
                                        [sys.CoalUnit.QCAP_new[i]] * int(sys.CoalUnit.tl/5) + [0.] * int(len(sys.multiperiods)-sys.CoalUnit.tl/5-i-1))[:len(sys.multiperiods)] for i in range(len(sys.multiperiods))]), axis=0)[1:],
        # sys.CoalUnit.QCAP_EarlyRet_new[1:5] <= sys.CoalUnit.QCAP_EarlyRet_new[2:6], 
        # sys.CoalUnit.QCAP_EarlyRet_new[1:3] <= 0,
        # sys.CoalUnit.QCAP_EarlyRet_new[1:4] <= sys.CoalUnit.QCAP_EarlyRet[1:4],
        # sys.CoalUnit.QCAP_ret == sys.CoalUnit.QCAP[0] * cp.cumsum(cp.hstack([0] + list(gbv.retire_coal.loc[:, Province]))),
        sys.OilUnit.QCAP_ret == sys.OilUnit.QCAP[0] * cp.hstack([0] + [min(1/(sys.OilUnit.tl/5)*t, 1) for t in range(1, len(sys.multiperiods))]) + cp.cumsum(cp.hstack([0] * int(sys.OilUnit.tl/5) + [sys.OilUnit.QCAP_new[i] for i in range(int(len(sys.multiperiods)-sys.OilUnit.tl/5))])),
        sys.GasUnit.QCAP_ret == sys.GasUnit.QCAP[0] * cp.hstack([0] + [min(1/(sys.GasUnit.tl/5)*t, 1) for t in range(1, len(sys.multiperiods))]) + cp.cumsum(cp.hstack([0] * int(sys.GasUnit.tl/5) + [sys.GasUnit.QCAP_new[i] for i in range(int(len(sys.multiperiods)-sys.GasUnit.tl/5))])),
        sys.NuclearUnit.QCAP_ret == sys.NuclearUnit.QCAP[0] * cp.hstack([0] + [min(1/(sys.NuclearUnit.tl/5)*t, 1) for t in range(1, len(sys.multiperiods))]) + cp.cumsum(cp.hstack([0] * int(sys.NuclearUnit.tl/5) + [sys.NuclearUnit.QCAP_new[i] for i in range(int(len(sys.multiperiods)-sys.NuclearUnit.tl/5))])),
        sys.CoalUnitCCS.QCAP_ret == sys.CoalUnitCCS.QCAP[0] * cp.hstack([0] + [min(1/(sys.CoalUnitCCS.tl/5)*t, 1) for t in range(1, len(sys.multiperiods))]) +
            cp.cumsum(cp.hstack([0] * int(sys.CoalUnitCCS.tl/5) + [sys.CoalUnitCCS.QCAP_rnv[i] for i in range(int(len(sys.multiperiods)-sys.CoalUnitCCS.tl/5))])) +\
            cp.cumsum(cp.hstack([0] * int(sys.CoalUnitCCS.tl/5) + [sys.CoalUnitCCS.QCAP_new[i] for i in range(int(len(sys.multiperiods)-sys.CoalUnitCCS.tl/5))])),
        sys.BiomassUnit.QCAP_ret == sys.BiomassUnit.QCAP[0] * cp.hstack([0] + [min(1/(sys.BiomassUnit.tl/5)*t, 1) for t in range(1, len(sys.multiperiods))]) + cp.cumsum(cp.hstack([0] * int(sys.BiomassUnit.tl/5) + [sys.BiomassUnit.QCAP_new[i] for i in range(int(len(sys.multiperiods)-sys.BiomassUnit.tl/5))])),
        sys.BECCSUnit.QCAP_ret == sys.BECCSUnit.QCAP[0] * cp.hstack([0] + [min(1/(sys.BECCSUnit.tl/5)*t, 1) for t in range(1, len(sys.multiperiods))]) + cp.cumsum(cp.hstack([0] * int(sys.BECCSUnit.tl/5) + [sys.BECCSUnit.QCAP_new[i] for i in range(int(len(sys.multiperiods)-sys.BECCSUnit.tl/5))])),
        sys.GasUnitCCS.QCAP_ret == sys.GasUnitCCS.QCAP[0] * cp.hstack([0] + [min(1/(sys.GasUnitCCS.tl/5)*t, 1) for t in range(1, len(sys.multiperiods))]) + cp.cumsum(cp.hstack([0] * int(sys.GasUnitCCS.tl/5) + [sys.GasUnitCCS.QCAP_new[i] for i in range(int(len(sys.multiperiods)-sys.GasUnitCCS.tl/5))])),
        sys.HydroUnit.QCAP_ret == sys.HydroUnit.QCAP[0] * cp.hstack([0] + [min(1/(sys.HydroUnit.tl/5)*t, 1) for t in range(1, len(sys.multiperiods))]) ,
        sys.OffWindUnit.QCAP_ret == sys.OffWindUnit.QCAP[0] * cp.hstack([0] + [min(1/(sys.OffWindUnit.tl/5)*t, 1) for t in range(1, len(sys.multiperiods))]) + cp.cumsum(cp.hstack([0] * int(sys.OffWindUnit.tl/5) + [sys.OffWindUnit.QCAP_new[i] for i in range(int(len(sys.multiperiods)-sys.OffWindUnit.tl/5))])),
        sys.OnWindUnit.QCAP_ret == sys.OnWindUnit.QCAP[0] * cp.hstack([0] + [min(1/(sys.OnWindUnit.tl/5)*t, 1) for t in range(1, len(sys.multiperiods))]) + cp.cumsum(cp.hstack([0] * int(sys.OnWindUnit.tl/5) + [sys.OnWindUnit.QCAP_new[i] for i in range(int(len(sys.multiperiods)-sys.OnWindUnit.tl/5))])),
        sys.PVUnit.QCAP_ret == sys.PVUnit.QCAP[0] * cp.hstack([0] + [min(1/(sys.PVUnit.tl/5)*t, 1) for t in range(1, len(sys.multiperiods))]) + cp.cumsum(cp.hstack([0] * int(sys.PVUnit.tl/5) + [sys.PVUnit.QCAP_new[i] for i in range(int(len(sys.multiperiods)-sys.PVUnit.tl/5))])),
        sys.CBECCSUnit20.QCAP_ret == cp.cumsum(cp.hstack([0] * int(sys.CBECCSUnit20.tl/5) + [sys.CBECCSUnit20.QCAP_rnv[i] for i in range(int(len(sys.multiperiods)-sys.CBECCSUnit20.tl/5))])),
        sys.CBECCSUnit40.QCAP_ret == cp.cumsum(cp.hstack([0] * int(sys.CBECCSUnit40.tl/5) + [sys.CBECCSUnit40.QCAP_rnv[i] for i in range(int(len(sys.multiperiods)-sys.CBECCSUnit40.tl/5))])),
        sys.CBECCSUnit60.QCAP_ret == cp.cumsum(cp.hstack([0] * int(sys.CBECCSUnit60.tl/5) + [sys.CBECCSUnit60.QCAP_rnv[i] for i in range(int(len(sys.multiperiods)-sys.CBECCSUnit60.tl/5))])),
        sys.CBECCSUnit80.QCAP_ret == cp.cumsum(cp.hstack([0] * int(sys.CBECCSUnit80.tl/5) + [sys.CBECCSUnit80.QCAP_rnv[i] for i in range(int(len(sys.multiperiods)-sys.CBECCSUnit80.tl/5))])),
        sys.CBECCSUnit100.QCAP_ret == cp.cumsum(cp.hstack([0] * int(sys.CBECCSUnit100.tl/5) + [sys.CBECCSUnit100.QCAP_rnv[i] for i in range(int(len(sys.multiperiods)-sys.CBECCSUnit100.tl/5))])),
        sys.CBECCSUnit20.QCAP_rnv + sys.CBECCSUnit40.QCAP_rnv + sys.CBECCSUnit60.QCAP_rnv + sys.CBECCSUnit80.QCAP_rnv + sys.CBECCSUnit100.QCAP_rnv + sys.CoalUnitCCS.QCAP_rnv >= 0.1 * (sys.CoalUnit.QCAP_EarlyRet_new + sys.CoalUnit.QCAP_EarlyRet),
        sys.CBECCSUnit20.QCAP_rnv + sys.CBECCSUnit40.QCAP_rnv + sys.CBECCSUnit60.QCAP_rnv + sys.CBECCSUnit80.QCAP_rnv + sys.CBECCSUnit100.QCAP_rnv + sys.CoalUnitCCS.QCAP_rnv <= (sys.CoalUnit.QCAP_EarlyRet_new + sys.CoalUnit.QCAP_EarlyRet),
        (sys.CoalUnit.QCAP_EarlyRet_new + sys.CoalUnit.QCAP_EarlyRet)[:3] <= 2,
        
        # sys.CoalUnit.QCAP_EarlyRet[:3] <= [0, 0, 15],

        sys.CoalUnit.QCAP == cp.cumsum(sys.CoalUnit.QCAP_new) + sys.CoalUnit.install_BY - sys.CoalUnit.QCAP_ret,
        # sys.CoalUnit.QCAP == cp.cumsum(sys.CoalUnit.QCAP_new) + sys.CoalUnit.install_BY - sys.CoalUnit.QCAP_ret - cp.cumsum(sys.CoalUnit.QCAP_EarlyRet),  cp.hstack(list(cp.cumsum(sys.CoalUnit.QCAP_EarlyRet))[:list(gbv.retire_coal.loc[:, Province].cumsum().round(decimals=15)).index(1) + 1] + [0.] * (9 -1- list(gbv.retire_coal.loc[:, Province].cumsum().round(decimals=15)).index(1)))
        sys.OilUnit.QCAP == cp.cumsum(sys.OilUnit.QCAP_new) - sys.OilUnit.QCAP_ret + sys.OilUnit.install_BY,
        sys.GasUnit.QCAP == cp.cumsum(sys.GasUnit.QCAP_new) - sys.GasUnit.QCAP_ret + sys.GasUnit.install_BY,
        sys.NuclearUnit.QCAP == cp.cumsum(sys.NuclearUnit.QCAP_new) - sys.NuclearUnit.QCAP_ret + sys.NuclearUnit.install_BY,
        sys.CoalUnitCCS.QCAP == cp.cumsum(sys.CoalUnitCCS.QCAP_new) - sys.CoalUnitCCS.QCAP_ret + sys.CoalUnitCCS.install_BY + cp.cumsum(sys.CoalUnitCCS.QCAP_rnv),
        sys.BiomassUnit.QCAP == cp.cumsum(sys.BiomassUnit.QCAP_new) - sys.BiomassUnit.QCAP_ret + sys.BiomassUnit.install_BY,
        sys.BECCSUnit.QCAP == cp.cumsum(sys.BECCSUnit.QCAP_new) - sys.BECCSUnit.QCAP_ret + sys.BECCSUnit.install_BY,
        sys.GasUnitCCS.QCAP == cp.cumsum(sys.GasUnitCCS.QCAP_new) - sys.GasUnitCCS.QCAP_ret + sys.GasUnitCCS.install_BY,
        sys.HydroUnit.QCAP == cp.cumsum(sys.HydroUnit.QCAP_new) - sys.HydroUnit.QCAP_ret + sys.HydroUnit.install_BY,
        sys.OffWindUnit.QCAP == cp.cumsum(sys.OffWindUnit.QCAP_new) - sys.OffWindUnit.QCAP_ret + sys.OffWindUnit.install_BY,
        sys.OnWindUnit.QCAP == cp.cumsum(sys.OnWindUnit.QCAP_new) - sys.OnWindUnit.QCAP_ret + sys.OnWindUnit.install_BY,
        sys.PVUnit.QCAP == cp.cumsum(sys.PVUnit.QCAP_new) - sys.PVUnit.QCAP_ret + sys.PVUnit.install_BY,
        sys.CBECCSUnit20.QCAP == cp.cumsum(sys.CBECCSUnit20.QCAP_rnv) - sys.CBECCSUnit20.QCAP_ret + sys.CBECCSUnit20.install_BY,
        sys.CBECCSUnit40.QCAP == cp.cumsum(sys.CBECCSUnit40.QCAP_rnv) - sys.CBECCSUnit40.QCAP_ret + sys.CBECCSUnit40.install_BY,
        sys.CBECCSUnit60.QCAP == cp.cumsum(sys.CBECCSUnit60.QCAP_rnv) - sys.CBECCSUnit60.QCAP_ret + sys.CBECCSUnit60.install_BY,
        sys.CBECCSUnit80.QCAP == cp.cumsum(sys.CBECCSUnit80.QCAP_rnv) - sys.CBECCSUnit80.QCAP_ret + sys.CBECCSUnit80.install_BY,
        sys.CBECCSUnit100.QCAP == cp.cumsum(sys.CBECCSUnit100.QCAP_rnv) - sys.CBECCSUnit100.QCAP_ret + sys.CBECCSUnit100.install_BY,

        # 提前退役
        # sys.CoalUnit.QCAP_EarlyRet[1:] <= sys.CoalUnit.QCAP[1:] * 0.4,
        # sys.CoalUnit.QSELEC[2:6] >= sys.CoalUnit.QSELEC[1:5] * 0.3,
        # sys.CoalUnit.QSELEC[3:] <= sys.CoalUnit.QSELEC[2:-1],
        # sys.CoalUnit.QSELEC[3:] <= sys.CoalUnit.QSELEC[2:-1],
        
        # 机组基年装机
        cp.hstack(sys.unit_byinstall()[0]) == cp.hstack(sys.unit_byinstall()[1]),
        
        # 上游产品
        sys.CoalUnit.Mineral.mining_fuel + sys.CoalUnit.Mineral.fuel_in - sys.CoalUnit.Mineral.fuel_out + sys.CoalUnit.Mineral.fuel_import == sys.CoalUnit.Mineral.fuel,
        sys.OilUnit.Mineral.mining_fuel + sys.OilUnit.Mineral.fuel_in - sys.OilUnit.Mineral.fuel_out + sys.OilUnit.Mineral.fuel_import == sys.OilUnit.Mineral.fuel,
        sys.GasUnit.Mineral.mining_fuel + sys.GasUnit.Mineral.fuel_in - sys.GasUnit.Mineral.fuel_out + sys.GasUnit.Mineral.fuel_import == sys.GasUnit.Mineral.fuel,
        sys.CoalUnit.Mineral.fuel >= sys.CoalUnit.QCAP * sys.CoalUnit.CAPACT * sys.CoalUnit.AF / sys.CoalUnit.EFF + sys.CoalUnitCCS.QCAP * sys.CoalUnitCCS.CAPACT * sys.CoalUnitCCS.AF / sys.CoalUnitCCS.EFF +\
            sys.CBECCSUnit20.QSELEC / sys.CBECCSUnit20.EFF * (1 - sys.CBECCSUnit20.ratio) +\
            sys.CBECCSUnit40.QSELEC / sys.CBECCSUnit40.EFF * (1 - sys.CBECCSUnit40.ratio) +\
            sys.CBECCSUnit60.QSELEC / sys.CBECCSUnit60.EFF * (1 - sys.CBECCSUnit60.ratio) +\
            sys.CBECCSUnit80.QSELEC / sys.CBECCSUnit80.EFF * (1 - sys.CBECCSUnit80.ratio) +\
            sys.CBECCSUnit100.QSELEC / sys.CBECCSUnit100.EFF * (1 - sys.CBECCSUnit100.ratio) +\
              sys.OtherCoal,
        sys.OilUnit.Mineral.fuel >= sys.OilUnit.QCAP * sys.OilUnit.CAPACT * sys.OilUnit.AF / sys.OilUnit.EFF + sys.OtherOil,
        sys.GasUnit.Mineral.fuel >= sys.GasUnit.QCAP * sys.GasUnit.CAPACT * sys.GasUnit.AF / sys.GasUnit.EFF + sys.GasUnitCCS.QCAP * sys.GasUnitCCS.CAPACT * sys.GasUnitCCS.AF / sys.GasUnitCCS.EFF + sys.OtherGas,
        
        # 资源潜力
        sys.PVUnit.QCAP <= gbv.potential.loc[Province, "pv"],
        sys.OnWindUnit.QCAP <= gbv.potential.loc[Province, "onwind"],
        sys.OffWindUnit.QCAP <= gbv.potential.loc[Province, "offwind"],
        sys.NuclearUnit.QCAP <= gbv.potential.loc[Province, "nuclear"],
        sys.HydroUnit.QCAP <= gbv.potential.loc[Province, "hydro"],
        sys.BiomassUnit.QSELEC / sys.BiomassUnit.EFF +\
                sys.CBECCSUnit20.QSELEC / sys.CBECCSUnit20.EFF * sys.CBECCSUnit20.ratio +\
                sys.CBECCSUnit40.QSELEC / sys.CBECCSUnit40.EFF * sys.CBECCSUnit40.ratio +\
                sys.CBECCSUnit60.QSELEC / sys.CBECCSUnit60.EFF * sys.CBECCSUnit60.ratio +\
                sys.CBECCSUnit80.QSELEC / sys.CBECCSUnit80.EFF * sys.CBECCSUnit80.ratio +\
                sys.CBECCSUnit100.QSELEC / sys.CBECCSUnit100.EFF * sys.CBECCSUnit100.ratio +\
              sys.BECCSUnit.QSELEC / sys.BECCSUnit.EFF <= gbv.biomass.loc[Province, :].values,
        
        # 先进装机
        sys.CoalUnit.QCAP_new[2:] <= sys.CoalUnit.QCAP[1:-1] * 0.15,
        # sys.CoalUnitCCS.QCAP_new[1:3] <= sys.CoalUnitCCS.QCAP[:2] * 0.05 + 2,
        sys.CoalUnitCCS.QCAP_new[3:] <= sys.CoalUnitCCS.QCAP[2:-1] * 0.3 + 5,
        sys.GasUnit.QCAP_new[1:] + sys.GasUnitCCS.QCAP_new[1:] <= (sys.GasUnit.QCAP[:-1] + sys.GasUnitCCS.QCAP[:-1]) * 0.1 + 1,
        sys.OilUnit.QCAP_new[1:] <= sys.OilUnit.QCAP[:-1] * 0.1,
        # sys.BiomassUnit.QCAP_new[1:] <= sys.BiomassUnit.QCAP[:-1] * 0.1 + 1,
        # sys.BECCSUnit.QCAP_new[1:] <= sys.BECCSUnit.QCAP[:-1] * 0.3 + 1,
        # sys.NuclearUnit.QCAP_new[1:] <= sys.NuclearUnit.QCAP[:-1] * 1.2 + 2,
        # sys.HydroUnit.QCAP_new[1:] <= sys.HydroUnit.QCAP[:-1] * 0.8 + 0.2,
        # sys.PVUnit.QCAP_new[1:] <= cp.multiply(sys.PVUnit.QCAP[:-1], gbv.pv_growth.loc[Province, 2025:]) + cp.hstack([15, 10, 10, 5, 5, 3, 0, 0]),
        # sys.OnWindUnit.QCAP_new[1:] + sys.OffWindUnit.QCAP_new[1:] <= cp.multiply((sys.OnWindUnit.QCAP[:-1] + sys.OffWindUnit.QCAP[:-1]), gbv.wind_growth.loc[Province, 2025:]) + cp.hstack([15, 0, 0, 0, 0, 0, 0, 0]),
        # sys.PVUnit.QCAP_new[1:] <= 130,
    
        # sys.CoalUnit.QCAP[2:-1] >= sys.CoalUnit.QCAP[3:],
        sys.BECCSUnit.QCAP[1:] >= sys.BECCSUnit.QCAP[:-1],
        sys.OnWindUnit.QCAP[2:] >= sys.OnWindUnit.QCAP[1:-1],
        sys.OffWindUnit.QCAP[2:] >= sys.OffWindUnit.QCAP[1:-1],
        sys.PVUnit.QCAP[2:] >= sys.PVUnit.QCAP[1:-1],
        sys.NuclearUnit.QCAP[2:] >= sys.NuclearUnit.QCAP[1:-1],
        sys.HydroUnit.QCAP[1:] >= sys.HydroUnit.QCAP[:-1],
        sys.CoalUnitCCS.QCAP[1:] >= sys.CoalUnitCCS.QCAP[:-1],

       
    ]
    return constraint


