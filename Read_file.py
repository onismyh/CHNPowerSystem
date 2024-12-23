import pandas as pd
from GlobalVar import globalVariables as gbv
def read_file():
    demand = pd.read_excel("../data/ssp_demand.xlsx", sheet_name="60CN", index_col=0)
    unit = pd.read_excel("../data/source data.xlsx", sheet_name="机组1", index_col=0)
    tech_curve = pd.read_excel("../data/source data.xlsx", sheet_name="机组技术曲线", index_col=0)
    fuel_cost = pd.read_excel("../data/source data.xlsx", sheet_name="燃料成本", index_col=0)
    by_install = pd.read_excel("../data/source data.xlsx", sheet_name="基年装机", index_col=0)
    # emission = pd.read_excel("../data/source data.xlsx", sheet_name="emission", index_col=0).fillna(10e8)
    EFF_coal = pd.read_excel("../data/source data.xlsx", sheet_name="EFF", index_col=0)
    AF = pd.read_excel("../data/source data.xlsx", sheet_name="AF", index_col=0)
    retire_coal = pd.read_excel("../data/source data.xlsx", sheet_name="煤电退役", index_col=0)
    potential = pd.read_excel("../data/source data.xlsx", sheet_name="资源潜力", index_col=0).fillna(10e8)
    biomass = pd.read_excel("../data/source data.xlsx", sheet_name="生物质", index_col=0)
    trans_act_cost = pd.read_excel("../data/source data.xlsx", sheet_name="输电成本",index_col=0)
    trans_ncap_cost = pd.read_excel("../data/source data.xlsx", sheet_name="NCAP_COST_trans",index_col=0)
    other_sec = pd.read_excel("../data/source data.xlsx", sheet_name="煤油气", index_col=0)
    pv_growth = pd.read_excel("../data/source data.xlsx", sheet_name="pv_growth", index_col=0)
    wind_growth = pd.read_excel("../data/source data.xlsx", sheet_name="wind_growth", index_col=0)
    trans_growth = pd.read_excel("../data/source data.xlsx", sheet_name="Trans_growth", index_col=0)
    
    return demand, unit, tech_curve, fuel_cost, by_install, potential, biomass, trans_act_cost, trans_ncap_cost, other_sec, EFF_coal, AF, retire_coal, pv_growth, wind_growth, trans_growth

def read_file_trans():
    lv_act = pd.read_excel("../data/ELC_trans/Trans_cost.xlsx", sheet_name="LV_ACT_COST", index_col=0).fillna(500)
    lv_ncap = pd.read_excel("../data/ELC_trans/Trans_cost.xlsx", sheet_name="LV_NCAP_COST", index_col=0).fillna(500)
    hv_act = pd.read_excel("../data/ELC_trans/Trans_cost.xlsx", sheet_name="HV_ACT_COST", index_col=0).fillna(500)
    hv_ncap = pd.read_excel("../data/ELC_trans/Trans_cost.xlsx", sheet_name="HV_NCAP_COST", index_col=0).fillna(500)
    uhv_act = pd.read_excel("../data/ELC_trans/Trans_cost.xlsx", sheet_name="UHV_ACT_COST", index_col=0).fillna(500)
    uhv_ncap = pd.read_excel("../data/ELC_trans/Trans_cost.xlsx", sheet_name="UHV_NCAP_COST", index_col=0).fillna(500)

    cons_trans = pd.read_excel("../data/ELC_trans/contraints.xlsx")
    cons_trans = cons_trans[cons_trans[2020]!=0]
    return lv_act, lv_ncap, hv_act, hv_ncap, uhv_act, uhv_ncap, cons_trans
    
def SSP():
    ssp1 = pd.read_excel("../data/SSP demand_7.xlsx", sheet_name="SSP1", index_col=0)
    ssp2 = pd.read_excel("../data/SSP demand_7.xlsx", sheet_name="SSP2", index_col=0)
    ssp3 = pd.read_excel("../data/SSP demand_7.xlsx", sheet_name="SSP3", index_col=0)
    ssp4 = pd.read_excel("../data/SSP demand_7.xlsx", sheet_name="SSP4", index_col=0)
    ssp5 = pd.read_excel("../data/SSP demand_7.xlsx", sheet_name="SSP5", index_col=0)
    # h2elec = pd.read_excel("../data/SSP demand_7.xlsx", sheet_name="Sheet1", index_col=0)
    ssp = pd.read_excel("../data/SSP demand_7.xlsx", sheet_name="SSP", index_col=0)
    return ssp1, ssp2, ssp3, ssp4, ssp5, ssp

gbv.demand, gbv.unit, gbv.tech_curve, gbv.fuel_cost, gbv.by_install, gbv.potential, gbv.biomass, gbv.trans_act_cost, gbv.trans_ncap_cost, \
    gbv.other_sec, gbv.EFF_coal, gbv.AF, gbv.retire_coal, gbv.pv_growth, gbv.wind_growth, gbv.trans_growth = read_file()
gbv.lv_act, gbv.lv_ncap, gbv.hv_act, gbv.hv_ncap, gbv.uhv_act, gbv.uhv_ncap, gbv.cons_trans = read_file_trans()
# gbv.ssp1, gbv.ssp2, gbv.ssp3, gbv.ssp4, gbv.ssp5, gbv.ssp = SSP()
# gbv.coa_price = {"BEIJ": 5.061825, "TIAN":4.937, "HEBE":4.614241, "SHNX":3.491946, "SHAD":6.077832, "NEMO":2.735464, "SHAN":5.862176, "JINU":5.520176,
#                   "ZHEJ":6.234614, "ANHU":5.969316, "FUJI":5.866581, "HUBE":5.929930, "HUNA":5.693200, "HENA":5.350011, "JINX":6.895393,
#              "SICH":5.977829, "CHON":6.131810, "LIAO":5.197401, "JILI":4.711492, "HEIL":4.650186, "SHAA":4.833489, "GANS":4.281152, "NINX":3.333743,
#                "QING":5.318813, "GUAD":5.810497, "GUAX":6.859814, "YUNN":5.492731, "GUIZ":5.240646, "HAIN":6.212842, "XING":2.198846}
gbv.Provinces = ['BEIJ', 'TIAN', 'HEBE', 'SHNX', 'NEMO', 'LIAO', 'JILI', 'HEIL', 'SHAN', 'JINU', 'ZHEJ', 'ANHU', 'FUJI', 'JINX', 'SHAD',
                 'HENA', 'HUBE', 'HUNA', 'GUAD', 'GUAX', 'HAIN', 'CHON', 'SICH', 'GUIZ', 'YUNN', 'SHAA', 'GANS', 'QING', 'NINX', 'XING']

gbv.coa_price = {"BEIJ": 5.061825, "TIAN":5.464, "HEBE":5.26, "SHNX":3.808, "SHAD":6.09, "NEMO":3.25, "SHAN":5.831, "JINU":6.044, "ZHEJ":6.484, "ANHU":6.46,
                  "FUJI":5.997, "HUBE":6.615, "HUNA":6.9, "HENA":5.655, "JINX":7.441, "SICH":6.382, "CHON":6.555, "LIAO":6.2, "JILI":6.488, "HEIL":6.208,
                    "SHAA":4.392, "GANS":5.53, "NINX":4.03, "QING":5.63, "GUAD":6.274, "GUAX":8.096, "YUNN":5.109, "GUIZ":5.764, "HAIN":5.806, "XING":3.016, "China":5}