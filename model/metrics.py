import numpy as np
from scipy.stats import truncnorm

####################
#                  #
#  USEFUL METHODS  #
#                  #
####################


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    """
    Returns sample from Normal distribution in range [low,upp]
    """
    ret = truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs()#random_state=rng)
    return ret

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_child_probability(x, *p):
    """
    Computes the probability to have a child at a given age, parameters a,b,c are tuned at initialization. 
    The curve form has been selected from bell shaped curves.
    """
    a,b,c = p
    mean = 32
    return (b * (a**3)) / ((x - mean)**2 + c*(a**2))

    


####################
#                  #
#  MODEL REPORTER  #
#                  #
####################
def get_emigrants_num(model):
    return model.emigrants_num

def get_immigrants_num(model):
    return model.immigrants_num

def get_f1_los(model):
    los = [dwell.household.satisfaction for dwell in model.dwellings_list if dwell.household and 1 in dwell.functions]
    if len(los) > 0:
        return np.mean(los)
    print("no households for this df")
    return None

def get_f2_los(model):
    los = [dwell.household.satisfaction for dwell in model.dwellings_list if dwell.household and 2 in dwell.functions]
    if len(los) > 0:
        return np.mean(los)
    print("no households for this df")
    return None

def get_f3_los(model):
    los = [dwell.household.satisfaction for dwell in model.dwellings_list if dwell.household and 3 in dwell.functions]
    if len(los) > 0:
        return np.mean(los)
    print("no households for this df")
    return None

def get_f4_los(model):
    los = [dwell.household.satisfaction for dwell in model.dwellings_list if dwell.household and 4 in dwell.functions]
    if len(los) > 0:
        return np.mean(los)
    print("no households for this df")
    return None

def get_f5_los(model):
    los = [dwell.household.satisfaction for dwell in model.dwellings_list if dwell.household and 5 in dwell.functions]
    if len(los) > 0:
        return np.mean(los)
    print("no households for this df")
    return None

def get_f6_los(model):
    los = [dwell.household.satisfaction for dwell in model.dwellings_list if dwell.household and 6 in dwell.functions]
    if len(los) > 0:
        return np.mean(los)
    print("no households for this df")
    return None

def get_f7_los(model):
    los = [dwell.household.satisfaction for dwell in model.dwellings_list if dwell.household and 7 in dwell.functions]
    if len(los) > 0:
        return np.mean(los)
    print("no households for this df")
    return None

def get_f8_los(model):
    los = [dwell.household.satisfaction for dwell in model.dwellings_list if dwell.household and 8 in dwell.functions]
    if len(los) > 0:
        return np.mean(los)
    print("no households for this df")
    return None

def get_f9_los(model):
    los = [dwell.household.satisfaction for dwell in model.dwellings_list if dwell.household and 9 in dwell.functions]
    if len(los) > 0:
        return np.mean(los)
    print("no households for this df")
    return None

def compute_average_los(model):
    household_los = [household.satisfaction for household in model.households_list]
    return np.mean(household_los)

def compute_average_age(model):
    tenants_age = [tenant.age for household in model.households_list for tenant in household.members]    
    return np.mean(tenants_age)

def compute_average_salary(model):
    household_salary = [household.get_household_salary() for household in model.households_list]
    return np.mean(household_salary)

def compute_number_dwellings(model):
    return len(model.dwellings_list)

def compute_available_dwellings(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty()])

def compute_number_dwellings_only_new(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_new()])

def compute_available_dwellings_only_new(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and dwell.is_new()])

def compute_available_dwellings_MOBI(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and dwell.get_owner() == "MOBI"])

def compute_available_dwellings_SCHL(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and dwell.get_owner() == "SCHL"])

def compute_available_dwellings_ABZ(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and dwell.get_owner() == "ABZ"])
    
def compute_average_relocations_waited_month(model):
    household_waited_month = [household.months_waited_since_mover for household in model.full_household_list]
    return np.mean(household_waited_month)  

def get_number_households(model):
    return len(model.households_list)

def get_sqm_per_tenant(model):
    tenants_sqm = [household.current_dwelling.size / household.get_household_size() for household in model.households_list]
    return np.mean(tenants_sqm)

def get_sqm_per_tenant_MOBI(model):
    tenants_sqm = [household.current_dwelling.size / household.get_household_size() for household in model.households_list if household.current_dwelling.get_owner() == "MOBI"]
    if len(tenants_sqm) > 0:
        return np.mean(tenants_sqm)
    return None

def get_sqm_per_tenant_ABZ(model):
    tenants_sqm = [household.current_dwelling.size / household.get_household_size() for household in model.households_list if household.current_dwelling.get_owner() == "ABZ"]
    if len(tenants_sqm) > 0:
        return np.mean(tenants_sqm)
    return None
    
def get_sqm_per_tenant_SCHL(model):
    tenants_sqm = [household.current_dwelling.size / household.get_household_size() for household in model.households_list if household.current_dwelling.get_owner() == "SCHL"]
    if len(tenants_sqm) > 0:
        return np.mean(tenants_sqm)
    return None

def get_sqm_per_tenant_1_room(model):
    tenants_sqm = [household.current_dwelling.size / household.get_household_size() for household in model.households_list if household.current_dwelling.rooms == 1]
    if len(tenants_sqm) > 0:
        return np.mean(tenants_sqm)
    return None

def get_sqm_per_tenant_2_room(model):
    tenants_sqm = [household.current_dwelling.size / household.get_household_size() for household in model.households_list if household.current_dwelling.rooms == 2]
    if len(tenants_sqm) > 0:
        return np.mean(tenants_sqm)
    return None

def get_sqm_per_tenant_3_room(model):
    tenants_sqm = [household.current_dwelling.size / household.get_household_size() for household in model.households_list if household.current_dwelling.rooms == 3]
    if len(tenants_sqm) > 0:
        return np.mean(tenants_sqm)
    return None

def get_sqm_per_tenant_4_room(model):
    tenants_sqm = [household.current_dwelling.size / household.get_household_size() for household in model.households_list if household.current_dwelling.rooms == 4]
    if len(tenants_sqm) > 0:
        return np.mean(tenants_sqm)
    return None

def get_sqm_per_tenant_5_room(model):
    tenants_sqm = [household.current_dwelling.size / household.get_household_size() for household in model.households_list if household.current_dwelling.rooms == 5]
    if len(tenants_sqm) > 0:
        return np.mean(tenants_sqm)
    return None

def get_sqm_per_tenant_6_room(model):
    tenants_sqm = [household.current_dwelling.size / household.get_household_size() for household in model.households_list if household.current_dwelling.rooms == 6]
    if len(tenants_sqm) > 0:
        return np.mean(tenants_sqm)
    return None

def get_sqm_per_tenant_7_room(model):
    tenants_sqm = [household.current_dwelling.size / household.get_household_size() for household in model.households_list if household.current_dwelling.rooms == 7]
    if len(tenants_sqm) > 0:
        return np.mean(tenants_sqm)
    return None

def get_MOBI_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.get_owner() == "MOBI"])

def get_ABZ_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.get_owner() == "ABZ"])

def get_SCHL_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.get_owner() == "SCHL"])

def get_f1_available_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and 1 in dwell.functions])

def get_f2_available_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and 2 in dwell.functions])

def get_f3_available_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and 3 in dwell.functions])

def get_f4_available_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and 4 in dwell.functions])

def get_f5_available_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and 5 in dwell.functions])

def get_f6_available_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and 6 in dwell.functions])

def get_f7_available_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and 7 in dwell.functions])

def get_f8_available_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and 8 in dwell.functions])

def get_f9_available_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and 9 in dwell.functions])

def get_1_room_available(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and dwell.rooms == 1])

def get_1_room_total(model):
    return len([dwell for dwell in model.dwellings_list if dwell.rooms == 1])

def get_2_room_available(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and dwell.rooms == 2])

def get_2_room_total(model):
    return len([dwell for dwell in model.dwellings_list if dwell.rooms == 2])

def get_3_room_available(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and dwell.rooms == 3])

def get_3_room_total(model):
    return len([dwell for dwell in model.dwellings_list if dwell.rooms == 3])

def get_4_room_available(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and dwell.rooms == 4])

def get_4_room_total(model):
    return len([dwell for dwell in model.dwellings_list if dwell.rooms == 4])

def get_5_room_available(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and dwell.rooms == 5])

def get_5_room_total(model):
    return len([dwell for dwell in model.dwellings_list if dwell.rooms == 5])

def get_6_room_available(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and dwell.rooms == 6])

def get_6_room_total(model):
    return len([dwell for dwell in model.dwellings_list if dwell.rooms == 6])

def get_7_room_available(model):
    return len([dwell for dwell in model.dwellings_list if dwell.is_empty() and dwell.rooms == 7])

def get_7_room_total(model):
    return len([dwell for dwell in model.dwellings_list if dwell.rooms == 7])

def get_average_children_num(model):
    ages = [h.get_household_size() - h.get_adults_num() for h in model.households_list]
    return np.mean(ages)

def get_household_functions_average(model):
    functions_num = [len(h.desired_functions) for h in model.households_list]
    return np.mean(functions_num)

def get_dwelling_functions_average(model):
    functions_num = [len(dwell.functions) for dwell in model.dwellings_list]
    return np.mean(functions_num)

def get_f1_total_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if 1 in dwell.functions])

def get_f2_total_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if 2 in dwell.functions])

def get_f3_total_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if 3 in dwell.functions])

def get_f4_total_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if 4 in dwell.functions])

def get_f5_total_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if 5 in dwell.functions])

def get_f6_total_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if 6 in dwell.functions])

def get_f7_total_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if 7 in dwell.functions])

def get_f8_total_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if 8 in dwell.functions])

def get_f9_total_dwellings_number(model):
    return len([dwell for dwell in model.dwellings_list if 9 in dwell.functions])

def get_num_dwellings_bright(model):
    return len([dwell for dwell in model.dwellings_list if "bright" in dwell.get_all_characteristics()])

def get_num_dwellings_with_balcony(model):
    return len([dwell for dwell in model.dwellings_list if "with balcony" in dwell.get_all_characteristics()])

def get_num_dwellings_with_green_spaces(model):
    return len([dwell for dwell in model.dwellings_list if "with green spaces" in dwell.get_all_characteristics()])

def get_num_dwellings_with_parking_place(model):
    return len([dwell for dwell in model.dwellings_list if "with parking place" in dwell.get_all_characteristics()])

def get_num_dwellings_work(model):
    return len([dwell for dwell in model.dwellings_list if "work" in dwell.get_all_characteristics()])

def get_num_dwellings_public_transports(model):
    return len([dwell for dwell in model.dwellings_list if "public transports" in dwell.get_all_characteristics()])

def get_num_dwellings_city_center(model):
    return len([dwell for dwell in model.dwellings_list if "city center" in dwell.get_all_characteristics()])

def get_num_dwellings_safe(model):
    return len([dwell for dwell in model.dwellings_list if "safe" in dwell.get_all_characteristics()])

def get_num_dwellings_sociocultural_mix(model):
    return len([dwell for dwell in model.dwellings_list if "sociocultural mix" in dwell.get_all_characteristics()])

def get_num_dwellings_accessible_by_car(model):
    return len([dwell for dwell in model.dwellings_list if "accessible by car" in dwell.get_all_characteristics()])


####################
####################
####################
####################


def get_num_hh_type1(model):
    return len([household for household in model.households_list if household.TYPE == 1])

def get_num_hh_type2(model):
    return len([household for household in model.households_list if household.TYPE == 2])

def get_num_hh_type3(model):
    return len([household for household in model.households_list if household.TYPE == 3])

def get_num_hh_type4(model):
    return len([household for household in model.households_list if household.TYPE == 4])

def get_num_hh_type5(model):
    return len([household for household in model.households_list if household.TYPE == 5])

def get_num_hh_type6(model):
    return len([household for household in model.households_list if household.TYPE == 6])

def get_num_hh_type7(model):
    return len([household for household in model.households_list if household.TYPE == 7])

def get_num_hh_type8(model):
    return len([household for household in model.households_list if household.TYPE == 8])

def get_num_hh_type9(model):
    return len([household for household in model.households_list if household.TYPE == 9])

def get_num_hh_type10(model):
    return len([household for household in model.households_list if household.TYPE == 10])

def get_num_hh_type11(model):
    return len([household for household in model.households_list if household.TYPE == 11])

def get_num_hh_type12(model):
    return len([household for household in model.households_list if household.TYPE == 12])

def get_num_hh_type13(model):
    return len([household for household in model.households_list if household.TYPE == 13])

def get_num_hh_room3_MOBI(model):
    tenants_room3 = [household.get_household_size() for household in model.households_list if household.current_dwelling.get_owner() == "MOBI" and household.current_dwelling.rooms == 3]
    if len(tenants_room3) > 0:
        return np.mean(tenants_room3)
    return None

def get_num_hh_room3_SCHL(model):
    tenants_room3 = [household.get_household_size() for household in model.households_list if household.current_dwelling.get_owner() == "SCHL" and household.current_dwelling.rooms == 3]
    if len(tenants_room3) > 0:
        return np.mean(tenants_room3)
    return None

def get_num_hh_room3_ABZ(model):
    tenants_room3 = [household.get_household_size() for household in model.households_list if household.current_dwelling.get_owner() == "ABZ" and household.current_dwelling.rooms == 3]
    if len(tenants_room3) > 0:
        return np.mean(tenants_room3)
    return None

def get_num_hh_1_ABZ(model):
    return len([household for household in model.households_list if household.current_dwelling.get_owner() == "ABZ" and household.get_household_size() == 1])

def get_num_hh_1_SCHL(model):
    return len([household for household in model.households_list if household.current_dwelling.get_owner() == "SCHL" and household.get_household_size() == 1])

def get_num_hh_1_MOBI(model):
    return len([household for household in model.households_list if household.current_dwelling.get_owner() == "MOBI" and household.get_household_size() == 1])

def get_num_flatshare_type1(model):
    return len([household for household in model.households_list if household.TYPE == 1 and household.get_household_size() > 1])

def get_num_flatshare_type5(model):
    return len([household for household in model.households_list if household.TYPE == 5 and household.get_household_size() > 1])

def get_num_flatshare_type13(model):
    return len([household for household in model.households_list if household.TYPE == 13 and household.get_household_size() > 1])



####################
#                  #
#  AGENT REPORTER  #
#                  #
####################

def get_hh_id(hh):
    return hh.id

def get_hh_adults_age(hh):
    return hh.get_adults_average_age()

def get_hh_size(hh):
    return hh.get_household_size()

def get_hh_num_adults(hh):
    return hh.get_adults_num()

def get_hh_num_minors(hh):
    return hh.get_minors_num()

def get_hh_type(hh):
    return hh.TYPE

def get_hh_los(hh):
    return hh.satisfaction

def get_hh_num_desired_functions(hh):
    return len(hh.desired_functions)

def get_hh_trigger(hh):
    if hh.trigger:
        return hh.get_reason()
    return None 

def get_dwelling_id(hh):
    if hh.current_dwelling:
        return hh.current_dwelling.id
    return None

def get_mover(hh):
    return int(hh.mover)

def get_desired_functions(hh):
    return hh.desired_functions

def get_postcode(hh):
    return hh.current_dwelling.building.postcode

def get_owner(hh):
    return hh.current_dwelling.get_owner()   

