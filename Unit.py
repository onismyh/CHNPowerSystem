import numpy as np
import cvxpy as cp
from Read_file import gbv
from constraint import Constraint


class Mineral:
    def __init__(self, type, price_mining=4.5, price_transIn=4.55, price_trans_Out=4.45, price_import=4.75, price_export=4.45):
        self.type = type
        self.price_mining = price_mining
        self.price_import = price_import

    def __str__(self) -> str:
        return self.type

    def fuel_cost(self):               # 开采成本
        return self.price_mining * self.mining_fuel
    def fuel_trans(self):              # 调入调出
        # return self.fuel_in[t] * self.price_transIn - self.fuel_out[t] * self.price_transOut
        return 0
    def fuel_trade(self):              # 进口出口
        return self.fuel_import * self.price_import #- self.fuel_export[t] * self.price_export


class PowerUnit:
    def __init__(self, type_, region, multiperiod, fuel, install_BY, mining=False, EarlyRetire=False, rnvbeccs=False, ratio=0.5):
        self.type = type_
        self.region = region
        self.multiperiod = multiperiod
        self.fuel = fuel
        self.install_BY = install_BY
        self.i = 0.05
        self.emission_factor = {"coal":0.715, "coalccs":0.715 * 0.05, "oil":0.548, "gas":0.409, "gasccs":0.409*0.05, "water":0, "wind":0, "solar":0, "UR":0, "biomass":0, "beccs":-0.349}
        try:
            self.emi_factor = self.emission_factor[self.fuel]
        except:
            pass
        self.CAPACT = 31.536
        self.AF = gbv.unit.loc[self.type, "AF"]
        if self.fuel in ["coal", "oil", "gas", "hydro", "wind", "solar", "UR"]:
            self.AF = gbv.AF.loc[self.region, self.fuel]
        self.EFF = gbv.unit.loc[self.type, "EFF"]
        if self.fuel in ["oil", "gas"]:
            self.EFF = gbv.EFF_coal.loc[self.region, "EFF_"+self.fuel]
        if self.fuel=="coal":
            self.EFF = gbv.EFF_coal.loc[self.region, ["EFF_"+self.fuel+"~%d" % i for i in self.multiperiod]].values
        
    
        self.fuel_cost = np.array([gbv.fuel_cost.loc[self.region, self.fuel]])
        self.tc = gbv.unit.loc[self.type, "建设期"]
        self.tl = gbv.unit.loc[self.type, "寿期"]
        self.init_invest = gbv.unit.loc[self.type, "初投资"]
        self.op_cost = gbv.unit.loc[self.type, "运维成本"]
        self.act_cost = gbv.unit.loc[self.type, "运行成本"]
        self.mining = mining
        self.EarlyRetire = EarlyRetire
        self.rnvbeccs = rnvbeccs
        if self.rnvbeccs:
            self.ratio = ratio           # 共燃生物质掺比
        
        if self.mining:
            params = {"coal":{"price_mining":gbv.coa_price[self.region], "price_transIn":gbv.coa_price[self.region]*1.1, "price_trans_Out":gbv.coa_price[self.region]*0.9, "price_import":gbv.coa_price[self.region]*1.11, "price_export":gbv.coa_price[self.region]*0.9},
                      "oil":{"price_mining":9.8+5, "price_transIn":9.75+5, "price_trans_Out":9.65+5, "price_import":6.75+5, "price_export":8.72+5},
                      "gas":{"price_mining":6.227+4, "price_transIn":6+4, "price_trans_Out":5.95+4, "price_import":5+4, "price_export":5.5+4}}
            self.Mineral = Mineral(type=self.fuel, **params[self.fuel])
    
    def QSELEC1(self):
        qs = self.QCAP * self.CAPACT * self.AF
        return qs
    
    def GECOST1(self):
        self.init_invest = gbv.tech_curve.loc[self.type, ["NCAP_COST~%d" % i for i in self.multiperiod]].values
        COSTCAP = self.init_invest / self.tc * ((self.i + 1) ** self.tc - 1) / self.i * self.i * (1 + self.i) ** self.tl / ((1 + self.i) ** self.tl - 1)
        if self.rnvbeccs:
            GECOST1 = cp.multiply(self.QCAP_rnv, COSTCAP)
            return GECOST1
        if self.fuel == "coalccs":
            GECOST1 = cp.multiply(self.QCAP_new, COSTCAP) + cp.multiply(self.QCAP_rnv, COSTCAP/2)
            return GECOST1
        GECOST1 = cp.multiply(self.QCAP_new, COSTCAP)
        return GECOST1
    
    def GECOST2(self):
        self.act_cost = gbv.tech_curve.loc[self.type, ["ACT_COST~%d" % i for i in self.multiperiod]].values
        return cp.multiply(self.QSELEC, self.act_cost)
    
    def GECOST3(self):
        self.op_cost = gbv.tech_curve.loc[self.type, ["NCAP_FOM~%d" % i for i in self.multiperiod]].values
        return cp.multiply(self.QCAP, self.op_cost)
    
    def GECOST4(self):                        # 燃料开采成本
        if self.mining:
            return self.Mineral.fuel_cost()
        if self.rnvbeccs:  
            return cp.bmat([(self.QSELEC / self.EFF)[i] * cp.hstack([self.ratio, 1 - self.ratio]) for i in range(9)]) @ cp.hstack(gbv.fuel_cost.loc[self.region, self.fuel].values)
            
        return self.QSELEC / self.EFF * self.fuel_cost
    
    def GECOST5(self):                         # 煤炭贸易成本
        if self.mining:
            return self.Mineral.fuel_trade() + self.Mineral.fuel_trans()
        return 0
    
    def GECOST(self):
        DiscountMat = np.array([1 / (1 + self.i) ** (5*t) for t in range(len(self.multiperiod))]).reshape((len(self.multiperiod), 1))
        return (self.GECOST1() + self.GECOST2() + self.GECOST3() + self.GECOST4() + self.GECOST5()).T @ DiscountMat
    
    def emission(self):
        if self.rnvbeccs:
            self.emi_factor = cp.hstack([self.emission_factor[fuel] for fuel in self.fuel])
            return cp.bmat([(self.QSELEC * 3.413 / self.EFF)[i] * cp.hstack([self.ratio, 1 - self.ratio]) for i in range(9)]) @ self.emi_factor * 44 / 12 
        return self.QSELEC * 3.413 / self.EFF * self.emi_factor * 44 / 12     # 万吨
    
    

class Power_System:
    def __init__(self, name, multiperiods, ins_base, trans_bool=False):
        self.name = name
        self.multiperiods = multiperiods
        self.install_base = ins_base
        self.i = 0.05
        self.trans_bool = trans_bool
        self.fuel_coal = "coal"
        self.fuel_coalccs = "coalccs"
        self.fuel_oil = "oil"
        self.fuel_gas = "gas"
        self.fuel_gasccs = "gasccs"
        self.fuel_nuc = "UR"
        self.fuel_biomass = "biomass"
        self.fuel_beccs = "beccs"
        self.fuel_sol = "solar"
        self.fuel_wind = "wind"
        self.fuel_hydro = "hydro"
        self.fuel_cbeccs = ["beccs", "coal"]
        self.DiscountMat = np.array([1 / (1 + self.i) ** (5*t) for t in range(len(self.multiperiods))]).reshape((len(self.multiperiods), 1))
        self.CoalUnit = PowerUnit(type_="EPLTCOAUSC",
                                  region=self.name,
                                  multiperiod=self.multiperiods,
                                  fuel=self.fuel_coal,
                                  install_BY=self.install_base["coal"],
                                  mining=True,
                                  EarlyRetire=False)
        self.OilUnit = PowerUnit(type_="EPLTOILST",
                               region=self.name,
                               multiperiod=self.multiperiods,
                               fuel=self.fuel_oil,
                               install_BY=self.install_base["oil"],
                               mining=True)
        self.GasUnit = PowerUnit(type_="EPLTNGANGCC",
                               region=self.name,
                               multiperiod=self.multiperiods,
                               fuel=self.fuel_gas,
                               install_BY=self.install_base["gas"],
                               mining=True)
        self.NuclearUnit = PowerUnit(type_="EPLTNUC",
                                 region=self.name,
                                 multiperiod=self.multiperiods,
                                 fuel=self.fuel_nuc,
                                 install_BY=self.install_base["nuclear"])
        self.BiomassUnit = PowerUnit(type_="EPLTBIOSLDC",
                                 region=self.name,
                                 multiperiod=self.multiperiods,
                                 fuel=self.fuel_biomass,
                                 install_BY=self.install_base["biomass"])
        self.CoalUnitCCS = PowerUnit(type_="EPLTCUSCCCS",
                                 region=self.name,
                                 multiperiod=self.multiperiods,
                                 fuel=self.fuel_coalccs,
                                 install_BY=self.install_base["coalccs"])
        self.GasUnitCCS = PowerUnit(type_="EPLTNGACCS",
                                 region=self.name,
                                 multiperiod=self.multiperiods,
                                 fuel=self.fuel_gasccs,
                                 install_BY=self.install_base["gasccs"])
        self.BECCSUnit = PowerUnit(type_="EPLTBSLDCCS",
                                    region=self.name,
                                    multiperiod=self.multiperiods,
                                    fuel=self.fuel_beccs,
                                    install_BY=self.install_base["beccs"])
        self.HydroUnit = PowerUnit(type_="EPLTHYDL",
                                   region=self.name,
                                   multiperiod=self.multiperiods,
                                   fuel=self.fuel_hydro,
                                   install_BY=self.install_base["hydro"])
        self.OffWindUnit = PowerUnit(type_="EPLTWINOFS",
                                     region=self.name,
                                     multiperiod=self.multiperiods,
                                     fuel=self.fuel_wind,
                                     install_BY=self.install_base["offwind"])
        self.OnWindUnit = PowerUnit(type_="EPLTWINONS",
                                    region=self.name,
                                    multiperiod=self.multiperiods,
                                    fuel=self.fuel_wind,
                                    install_BY=self.install_base["onwind"])
        self.PVUnit = PowerUnit(type_="EPLTSOLPV",
                                region=self.name,
                                multiperiod=self.multiperiods,
                                fuel=self.fuel_sol,
                                install_BY=self.install_base["pv"])
        self.CBECCSUnit20 = PowerUnit(type_="EPLTCBECCS20",
                                region=self.name,
                                multiperiod=self.multiperiods,
                                fuel=self.fuel_cbeccs,
                                install_BY=0,
                                rnvbeccs=True, ratio=0.2)
        self.CBECCSUnit40 = PowerUnit(type_="EPLTCBECCS40",
                                region=self.name,
                                multiperiod=self.multiperiods,
                                fuel=self.fuel_cbeccs,
                                install_BY=0,
                                rnvbeccs=True, ratio=0.4)
        self.CBECCSUnit60 = PowerUnit(type_="EPLTCBECCS60",
                                region=self.name,
                                multiperiod=self.multiperiods,
                                fuel=self.fuel_cbeccs,
                                install_BY=0,
                                rnvbeccs=True, ratio=0.6)
        self.CBECCSUnit80 = PowerUnit(type_="EPLTCBECCS80",
                                region=self.name,
                                multiperiod=self.multiperiods,
                                fuel=self.fuel_cbeccs,
                                install_BY=0,
                                rnvbeccs=True, ratio=0.8)
        self.CBECCSUnit100 = PowerUnit(type_="EPLTCBECCS100",
                                region=self.name,
                                multiperiod=self.multiperiods,
                                fuel=self.fuel_cbeccs,
                                install_BY=0,
                                rnvbeccs=True, ratio=1.0)
        
        

    def __str__(self) -> str:
        return "Province %s Power System" % self.name
    
    def var_act(self):
        self.CoalUnit.QCAP = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.CoalUnit.QCAP_ret = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.CoalUnit.QCAP_new = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.CoalUnit.QSELEC = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.CoalUnit.QCAP_EarlyRet = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.CoalUnit.QCAP_EarlyRet_new = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.CoalUnit.Mineral.mining_fuel = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.CoalUnit.Mineral.fuel_in = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.CoalUnit.Mineral.fuel_out = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.CoalUnit.Mineral.fuel_import = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.CoalUnit.Mineral.fuel = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.OtherCoal = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)

        self.OilUnit.QCAP = cp.Variable(len(self.OilUnit.multiperiod), pos=True)
        self.OilUnit.QCAP_ret = cp.Variable(len(self.OilUnit.multiperiod), pos=True)
        self.OilUnit.QCAP_new = cp.Variable(len(self.OilUnit.multiperiod), pos=True)
        self.OilUnit.QSELEC = cp.Variable(len(self.OilUnit.multiperiod), pos=True)
        self.OilUnit.Mineral.mining_fuel = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.OilUnit.Mineral.fuel_in = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.OilUnit.Mineral.fuel_out = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.OilUnit.Mineral.fuel_import = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.OilUnit.Mineral.fuel = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.OtherOil = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)

        self.GasUnit.QCAP = cp.Variable(len(self.GasUnit.multiperiod), pos=True)
        self.GasUnit.QCAP_ret = cp.Variable(len(self.GasUnit.multiperiod), pos=True)
        self.GasUnit.QCAP_new = cp.Variable(len(self.GasUnit.multiperiod), pos=True)
        self.GasUnit.QSELEC = cp.Variable(len(self.GasUnit.multiperiod), pos=True)
        self.GasUnit.Mineral.mining_fuel = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.GasUnit.Mineral.fuel_in = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.GasUnit.Mineral.fuel_out = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.GasUnit.Mineral.fuel_import = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.GasUnit.Mineral.fuel = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        self.OtherGas = cp.Variable(len(self.CoalUnit.multiperiod), pos=True)
        
        self.NuclearUnit.QCAP = cp.Variable(len(self.NuclearUnit.multiperiod), pos=True)
        self.NuclearUnit.QCAP_ret = cp.Variable(len(self.NuclearUnit.multiperiod), pos=True)
        self.NuclearUnit.QCAP_new = cp.Variable(len(self.NuclearUnit.multiperiod), pos=True)
        self.NuclearUnit.QSELEC = cp.Variable(len(self.NuclearUnit.multiperiod), pos=True)
        
        self.BiomassUnit.QCAP = cp.Variable(len(self.BiomassUnit.multiperiod), pos=True)
        self.BiomassUnit.QCAP_ret = cp.Variable(len(self.BiomassUnit.multiperiod), pos=True)
        self.BiomassUnit.QCAP_new = cp.Variable(len(self.BiomassUnit.multiperiod), pos=True)
        self.BiomassUnit.QSELEC = cp.Variable(len(self.BiomassUnit.multiperiod), pos=True)

        self.BECCSUnit.QCAP = cp.Variable(len(self.BECCSUnit.multiperiod), pos=True)
        self.BECCSUnit.QCAP_ret = cp.Variable(len(self.BECCSUnit.multiperiod), pos=True)
        self.BECCSUnit.QCAP_new = cp.Variable(len(self.BECCSUnit.multiperiod), pos=True)
        self.BECCSUnit.QSELEC = cp.Variable(len(self.BECCSUnit.multiperiod), pos=True)

        self.CBECCSUnit20.QCAP = cp.Variable(len(self.CBECCSUnit20.multiperiod), pos=True)
        self.CBECCSUnit20.QCAP_ret = cp.Variable(len(self.CBECCSUnit20.multiperiod), pos=True)
        self.CBECCSUnit20.QCAP_rnv = cp.Variable(len(self.CBECCSUnit20.multiperiod), pos=True)
        self.CBECCSUnit20.QSELEC = cp.Variable(len(self.CBECCSUnit20.multiperiod), pos=True)

        self.CBECCSUnit40.QCAP = cp.Variable(len(self.CBECCSUnit40.multiperiod), pos=True)
        self.CBECCSUnit40.QCAP_ret = cp.Variable(len(self.CBECCSUnit40.multiperiod), pos=True)
        self.CBECCSUnit40.QCAP_rnv = cp.Variable(len(self.CBECCSUnit40.multiperiod), pos=True)
        self.CBECCSUnit40.QSELEC = cp.Variable(len(self.CBECCSUnit40.multiperiod), pos=True)

        self.CBECCSUnit60.QCAP = cp.Variable(len(self.CBECCSUnit60.multiperiod), pos=True)
        self.CBECCSUnit60.QCAP_ret = cp.Variable(len(self.CBECCSUnit60.multiperiod), pos=True)
        self.CBECCSUnit60.QCAP_rnv = cp.Variable(len(self.CBECCSUnit60.multiperiod), pos=True)
        self.CBECCSUnit60.QSELEC = cp.Variable(len(self.CBECCSUnit60.multiperiod), pos=True)

        self.CBECCSUnit80.QCAP = cp.Variable(len(self.CBECCSUnit80.multiperiod), pos=True)
        self.CBECCSUnit80.QCAP_ret = cp.Variable(len(self.CBECCSUnit80.multiperiod), pos=True)
        self.CBECCSUnit80.QCAP_rnv = cp.Variable(len(self.CBECCSUnit80.multiperiod), pos=True)
        self.CBECCSUnit80.QSELEC = cp.Variable(len(self.CBECCSUnit80.multiperiod), pos=True)

        self.CBECCSUnit100.QCAP = cp.Variable(len(self.CBECCSUnit100.multiperiod), pos=True)
        self.CBECCSUnit100.QCAP_ret = cp.Variable(len(self.CBECCSUnit100.multiperiod), pos=True)
        self.CBECCSUnit100.QCAP_rnv = cp.Variable(len(self.CBECCSUnit100.multiperiod), pos=True)
        self.CBECCSUnit100.QSELEC = cp.Variable(len(self.CBECCSUnit100.multiperiod), pos=True)
        
        self.CoalUnitCCS.QCAP = cp.Variable(len(self.CoalUnitCCS.multiperiod), pos=True)
        self.CoalUnitCCS.QCAP_ret = cp.Variable(len(self.CoalUnitCCS.multiperiod), pos=True)
        self.CoalUnitCCS.QCAP_new = cp.Variable(len(self.CoalUnitCCS.multiperiod), pos=True)
        self.CoalUnitCCS.QCAP_rnv = cp.Variable(len(self.CoalUnitCCS.multiperiod), pos=True)
        self.CoalUnitCCS.QSELEC = cp.Variable(len(self.CoalUnitCCS.multiperiod), pos=True)
        
        self.GasUnitCCS.QCAP = cp.Variable(len(self.GasUnitCCS.multiperiod), pos=True)
        self.GasUnitCCS.QCAP_ret = cp.Variable(len(self.GasUnitCCS.multiperiod), pos=True)
        self.GasUnitCCS.QCAP_new = cp.Variable(len(self.GasUnitCCS.multiperiod), pos=True)
        self.GasUnitCCS.QSELEC = cp.Variable(len(self.GasUnitCCS.multiperiod), pos=True)

        self.HydroUnit.QCAP = cp.Variable(len(self.HydroUnit.multiperiod), pos=True)
        self.HydroUnit.QCAP_ret = cp.Variable(len(self.HydroUnit.multiperiod), pos=True)
        self.HydroUnit.QCAP_new = cp.Variable(len(self.HydroUnit.multiperiod), pos=True)
        self.HydroUnit.QSELEC = cp.Variable(len(self.HydroUnit.multiperiod), pos=True)

        self.OffWindUnit.QCAP = cp.Variable(len(self.OffWindUnit.multiperiod), pos=True)
        self.OffWindUnit.QCAP_ret = cp.Variable(len(self.OffWindUnit.multiperiod), pos=True)
        self.OffWindUnit.QCAP_new = cp.Variable(len(self.OffWindUnit.multiperiod), pos=True)
        self.OffWindUnit.QSELEC = cp.Variable(len(self.OffWindUnit.multiperiod), pos=True)
        
        self.OnWindUnit.QCAP = cp.Variable(len(self.OnWindUnit.multiperiod), pos=True)
        self.OnWindUnit.QCAP_ret = cp.Variable(len(self.OnWindUnit.multiperiod), pos=True)
        self.OnWindUnit.QCAP_new = cp.Variable(len(self.OnWindUnit.multiperiod), pos=True)
        self.OnWindUnit.QSELEC = cp.Variable(len(self.OnWindUnit.multiperiod), pos=True)

        self.PVUnit.QCAP = cp.Variable(len(self.PVUnit.multiperiod), pos=True)
        self.PVUnit.QCAP_ret = cp.Variable(len(self.PVUnit.multiperiod), pos=True)
        self.PVUnit.QCAP_new = cp.Variable(len(self.PVUnit.multiperiod), pos=True)
        self.PVUnit.QSELEC = cp.Variable(len(self.PVUnit.multiperiod), pos=True)
        
        self.inElec_HV = cp.Variable((len(self.multiperiods), len(gbv.Provinces) - 1), pos=True)
        self.inECAP_HV = cp.Variable((len(self.multiperiods), len(gbv.Provinces) - 1), pos=True)
        
        self.inElec_LV = cp.Variable((len(self.multiperiods), len(gbv.Provinces) - 1), pos=True)
        self.inECAP_LV = cp.Variable((len(self.multiperiods), len(gbv.Provinces) - 1), pos=True)
        
        self.inElec_UHV = cp.Variable((len(self.multiperiods), len(gbv.Provinces) - 1), pos=True)
        self.inECAP_UHV = cp.Variable((len(self.multiperiods), len(gbv.Provinces) - 1), pos=True)
        
        



    def unit_elec(self):
        return [self.CoalUnit.QSELEC, self.OilUnit.QSELEC, self.GasUnit.QSELEC, self.NuclearUnit.QSELEC, self.BiomassUnit.QSELEC, self.BECCSUnit.QSELEC,
                self.CoalUnitCCS.QSELEC, self.GasUnitCCS.QSELEC, self.HydroUnit.QSELEC, self.OffWindUnit.QSELEC, self.OnWindUnit.QSELEC, self.PVUnit.QSELEC, self.CBECCSUnit20.QSELEC,
                self.CBECCSUnit40.QSELEC, self.CBECCSUnit60.QSELEC, self.CBECCSUnit80.QSELEC, self.CBECCSUnit100.QSELEC,]
    def unit_elec1(self):
        return [self.CoalUnit.QSELEC1(), self.OilUnit.QSELEC1(), self.GasUnit.QSELEC1(), self.NuclearUnit.QSELEC1(), self.BiomassUnit.QSELEC1(), self.BECCSUnit.QSELEC1(),
                self.CoalUnitCCS.QSELEC1(), self.GasUnitCCS.QSELEC1(), self.HydroUnit.QSELEC1(), self.OffWindUnit.QSELEC1(), self.OnWindUnit.QSELEC1(), self.PVUnit.QSELEC1(), self.CBECCSUnit20.QSELEC1(),
                self.CBECCSUnit40.QSELEC1(), self.CBECCSUnit60.QSELEC1(), self.CBECCSUnit80.QSELEC1(), self.CBECCSUnit100.QSELEC1(),]
    
    def unit_byinstall(self):
        return ([self.CoalUnit.QCAP[0], self.OilUnit.QCAP[0], self.GasUnit.QCAP[0], self.NuclearUnit.QCAP[0], self.BiomassUnit.QCAP[0], self.BECCSUnit.QCAP[0], 
                  self.CoalUnitCCS.QCAP[0], self.GasUnitCCS.QCAP[0], self.HydroUnit.QCAP[0], self.OffWindUnit.QCAP[0], self.OnWindUnit.QCAP[0], self.PVUnit.QCAP[0], self.CBECCSUnit20.QCAP[0],
                  self.CBECCSUnit40.QCAP[0], self.CBECCSUnit60.QCAP[0], self.CBECCSUnit80.QCAP[0], self.CBECCSUnit100.QCAP[0]],
                [self.CoalUnit.install_BY, self.OilUnit.install_BY, self.GasUnit.install_BY, self.NuclearUnit.install_BY, self.BiomassUnit.install_BY, self.BECCSUnit.install_BY,
                 self.CoalUnitCCS.install_BY, self.GasUnitCCS.install_BY, self.HydroUnit.install_BY, self.OffWindUnit.install_BY, self.OnWindUnit.install_BY, self.PVUnit.install_BY, self.CBECCSUnit20.install_BY,
                 self.CBECCSUnit40.install_BY, self.CBECCSUnit60.install_BY, self.CBECCSUnit80.install_BY, self.CBECCSUnit100.install_BY])
    
    def emissionELC(self):
        return self.CoalUnit.emission() + self.CoalUnitCCS.emission() + self.OilUnit.emission() + self.GasUnit.emission() + self.GasUnitCCS.emission() + self.BECCSUnit.emission() + self.CBECCSUnit20.emission() +\
                self.CBECCSUnit40.emission() + self.CBECCSUnit60.emission() + self.CBECCSUnit80.emission() + self.CBECCSUnit100.emission() 
    
    # def emission(self):
    #     return self.emissionELC()  + self.other_sector_emission()
        
    def ToELEC(self):
        return self.CoalUnit.QSELEC + self.OilUnit.QSELEC + self.GasUnit.QSELEC + self.HydroUnit.QSELEC + self.OnWindUnit.QSELEC + self.OffWindUnit.QSELEC +\
        self.PVUnit.QSELEC + self.NuclearUnit.QSELEC + self.CoalUnitCCS.QSELEC + self.GasUnitCCS.QSELEC + self.BiomassUnit.QSELEC + self.BECCSUnit.QSELEC + self.CBECCSUnit20.QSELEC +\
        self.CBECCSUnit40.QSELEC + self.CBECCSUnit60.QSELEC + self.CBECCSUnit80.QSELEC + self.CBECCSUnit100.QSELEC
    
    def ToQCAP(self):
        return self.CoalUnit.QCAP + self.OilUnit.QCAP + self.GasUnit.QCAP + self.HydroUnit.QCAP + self.OnWindUnit.QCAP + self.OffWindUnit.QCAP + self.PVUnit.QCAP +\
              self.NuclearUnit.QCAP + self.CoalUnitCCS.QCAP + self.GasUnitCCS.QCAP + self.BiomassUnit.QCAP + self.BECCSUnit.QCAP + self.CBECCSUnit20.QCAP +\
                self.CBECCSUnit40.QCAP + self.CBECCSUnit60.QCAP + self.CBECCSUnit80.QCAP + self.CBECCSUnit100.QCAP
    
    def ToCOST(self):
        return self.CoalUnit.GECOST() + self.OilUnit.GECOST() + self.GasUnit.GECOST() + self.HydroUnit.GECOST() + self.OnWindUnit.GECOST() + self.OffWindUnit.GECOST() +\
              self.PVUnit.GECOST() + self.NuclearUnit.GECOST() + self.BiomassUnit.GECOST() + self.CoalUnitCCS.GECOST() + self.GasUnitCCS.GECOST() + self.BECCSUnit.GECOST() + self.CBECCSUnit20.GECOST() +\
              self.CBECCSUnit40.GECOST() + self.CBECCSUnit60.GECOST() + self.CBECCSUnit80.GECOST() + self.CBECCSUnit100.GECOST() +\
                (self.CoalUnit.QCAP_EarlyRet_new.T) @ self.DiscountMat * 120
    def CBECCSUnit_QCAP(self):
        return self.CBECCSUnit20.QCAP + self.CBECCSUnit40.QCAP + self.CBECCSUnit60.QCAP + self.CBECCSUnit80.QCAP + self.CBECCSUnit100.QCAP
           
    def CBECCSUnit_QSELEC(self):
        return self.CBECCSUnit20.QSELEC + self.CBECCSUnit40.QSELEC + self.CBECCSUnit60.QSELEC + self.CBECCSUnit80.QSELEC + self.CBECCSUnit100.QSELEC
    
    def CBECCSUnit_Neg_Emission(self):
        return self.CBECCSUnit20.QSELEC * 3.413 / self.CBECCSUnit20.EFF * self.CBECCSUnit20.ratio * self.CBECCSUnit20.emi_factor[0] * 44 / 12 +\
                self.CBECCSUnit40.QSELEC * 3.413 / self.CBECCSUnit40.EFF * self.CBECCSUnit40.ratio * self.CBECCSUnit40.emi_factor[0] * 44 / 12 +\
                self.CBECCSUnit60.QSELEC * 3.413 / self.CBECCSUnit60.EFF * self.CBECCSUnit60.ratio * self.CBECCSUnit60.emi_factor[0] * 44 / 12 +\
                self.CBECCSUnit80.QSELEC * 3.413 / self.CBECCSUnit80.EFF * self.CBECCSUnit80.ratio * self.CBECCSUnit80.emi_factor[0] * 44 / 12 +\
                self.CBECCSUnit100.QSELEC * 3.413 / self.CBECCSUnit100.EFF * self.CBECCSUnit100.ratio * self.CBECCSUnit100.emi_factor[0] * 44 / 12
    def CBECCSUnit_Rnv(self):
        return self.CBECCSUnit20.QCAP_rnv + self.CBECCSUnit40.QCAP_rnv + self.CBECCSUnit60.QCAP_rnv + self.CBECCSUnit80.QCAP_rnv + self.CBECCSUnit100.QCAP_rnv

    def Renew_QSELEC(self):
        return self.OnWindUnit.QSELEC + self.OffWindUnit.QSELEC + self.PVUnit.QSELEC + self.BiomassUnit.QSELEC + self.BECCSUnit.QSELEC + self.HydroUnit.QSELEC
    def Renew_NonHydro_QSELEC(self):
        return self.OnWindUnit.QSELEC + self.OffWindUnit.QSELEC + self.PVUnit.QSELEC + self.BiomassUnit.QSELEC + self.BECCSUnit.QSELEC

class ChinaPowerSystem:
    def __init__(self, multiperiods, name="China"):
        self.multiperiods = multiperiods
        # self.Provinces = Provinces
        self.name = name
        self.PSM = Power_System(
            name = self.name,
            multiperiods = self.multiperiods,
            ins_base = gbv.by_install.loc["China", :]
        )
        self.PSM.var_act()
        self.constraints = []
        self.constraints.extend(Constraint(self.PSM, "China", self.PSM))
        
       
        
        
    
    def TotalCost(self):
        return self.PSM.ToCOST()
    