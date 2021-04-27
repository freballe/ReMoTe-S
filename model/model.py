import math
import csv
import numpy as np
import parameters
import pandas as pd
from scipy.optimize import curve_fit
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from dwelling import DwellingAgent 
from household import HouseholdAgent
from trigger import Trigger
from building import BuildingAgent
from metrics import * 

class HouseholdDecisionModel(Model):  
    def __init__(self, seed=42, scenario="baseline", sensitivity_analysis=False, char_probs=-1, household_shrink_prob=parameters.HOUSEHOLD_MOVES_AFTER_CHILD_LEAVES_PROBABILITY, dwellings_visited=parameters.NUM_DWELLING_VISITED, flat_size=parameters.MAX_HOUSEHOLD_SIZE, construction_rate=parameters.BUILDINGS_CONSTRUCTION_RATE, immigration_rate=parameters.IMMIGRATION_RATE, mean_rooms=parameters.MEAN_ROOMS):
        print("Starting model..")
        print("Loading data..")
        self.scenario = scenario
        if scenario == "C":
            parameters.MAX_HOUSEHOLD_SIZE = 2
        elif scenario == "D":
            parameters.MAX_HOUSEHOLD_SIZE = 4
        elif scenario == "Z":
            parameters.MEAN_ROOMS = 2.5

        if sensitivity_analysis:    # sensitivity analysis ON , if so it can overwrite some scenario setted parameters!!
            parameters.HOUSEHOLD_MOVES_AFTER_CHILD_LEAVES_PROBABILITY = household_shrink_prob
            parameters.NUM_DWELLING_VISITED = dwellings_visited
            parameters.MAX_HOUSEHOLD_SIZE = flat_size
            parameters.BUILDINGS_CONSTRUCTION_RATE = construction_rate
            parameters.IMMIGRATION_RATE = immigration_rate
            parameters.MEAN_ROOMS = mean_rooms
            
            if char_probs != -1:
                parameters.DWELLING_FUNCTIONS_PROBABILITIES["with green spaces"] = [char_probs] * 9
                parameters.DWELLING_FUNCTIONS_PROBABILITIES["public transports"] = [char_probs] * 9
                parameters.DWELLING_FUNCTIONS_PROBABILITIES["sociocultural mix"] = [char_probs] * 9

        # load CSVs
        self.ABZ_TYPE_DFC_df = pd.read_csv("../data/ABZ_TYPE_DF.csv", decimal=",", names=["TYPE","1","2","3","4","5","6","7","8","9","10","11","12","13"])
        self.SCHL_TYPE_DFC_df = pd.read_csv("../data/SCHL_TYPE_DF.csv", decimal=",", names=["TYPE","1","2","3","4","5","6","7","8","9","10","11","12","13"])
        self.MOBI_TYPE_DFC_df = pd.read_csv("../data/MOBI_TYPE_DF.csv", decimal=",", names=["TYPE","1","2","3","4","5","6","7","8","9","10","11","12","13"])
        self.ALL_OWNERS_TYPE_DFC_df = pd.read_csv("../data/ALL_OWNERS_TYPE_DFC.csv", decimal=",", names=["TYPE","1","2","3","4","5","6","7","8","9","10","11","12","13"])

        self.step_num = 0

        self.reason_dict_all = { # store how many times people moved for each trigger
                    "increase salary": 0,
                    "new building": 0,
                    "expire contract": 0,
                    "demolition": 0,
                    "renovation/transformation": 0,
                    "interpersonal problems": 0,
                    "rent too high": 0,
                    "underoccupancy": 0,
                    "growing old": 0,
                    "family": 0,
                    "change job location": 0,
                    "need for a change": 0,
                    "create couple": 0,
                    "new child": 0,
                    "separate/divorce": 0,
                    "children leaving": 0,
                    "leaving the flatshare": 0,
        } 

        self.reason_dict_move = { # store how many times people moved for each trigger
                    "increase salary": 0,
                    "new building": 0,
                    "expire contract": 0,
                    "demolition": 0,
                    "renovation/transformation": 0,
                    "interpersonal problems": 0,
                    "rent too high": 0,
                    "underoccupancy": 0,
                    "growing old": 0,
                    "family": 0,
                    "change job location": 0,
                    "need for a change": 0,
                    "create couple": 0,
                    "new child": 0,
                    "separate/divorce": 0,
                    "children leaving": 0,
                    "leaving the flatshare": 0,
        } 

        self.buildings_list = []
        self.dwellings_list = []
        self.households_list = []
        self.full_household_list = []                           # contains all the households that ever existed in the model
        self.schedule = BaseScheduler(self)
        self.birth_curve_coeff = self.init_birth_curve_coeff()  # coefficient of the curve that approximate births probability given age
        self.household_id = 0
        self.tenant_id = 0
        self.dwelling_id = 0
        self.building_id = 0

        self.immigrants_num = 0
        self.emigrants_num = 0
        print("Creating buildings..")
        self.set_buildings()
        print("Creating dwellings..")
        self.set_dwellings()
        print("DWELLINGS RENT MEAN", np.mean([dwell.rent_price for dwell in self.dwellings_list]) / 12)
        print("DWELLINGS NUM", len(self.dwellings_list))
        
        print("Assigning households to dwellings..")
        self.set_households()

        print("Assigned all households")
        self.create_vacant_dwellings()

        # vars needed for fuction calls that happen once a year
        self.create_building_month = np.random.randint(12)
        self.demolish_building_month = np.random.randint(12)
        self.renovate_dwelling_month = np.random.randint(12)
        self.check_cooperatives_month = np.random.randint(12)

        ## create data collector
        self.datacollector = DataCollector(
            model_reporters = { "los": compute_average_los, 
                                "number of months waited before relocation": compute_average_relocations_waited_month, 
                                "average age": compute_average_age,
                                "average salary": compute_average_salary,
                                "number of dwellings": compute_number_dwellings,
                                "available dwellings": compute_available_dwellings,
                                "number of dwellings new": compute_number_dwellings_only_new,
                                "available dwellings new": compute_available_dwellings_only_new,
                                "MOBI available dwellings": compute_available_dwellings_MOBI,
                                "SCHL available dwellings": compute_available_dwellings_SCHL, 
                                "ABZ available dwellings": compute_available_dwellings_ABZ,
                                "MOBI all dwellings": get_MOBI_dwellings_number,
                                "SCHL all dwellings": get_SCHL_dwellings_number,
                                "ABZ all dwellings": get_ABZ_dwellings_number,
                                "number of households": get_number_households, 
                                "overall sqm per tenant": get_sqm_per_tenant,
                                "MOBI sqm per tenant": get_sqm_per_tenant_MOBI, 
                                "SCHL sqm per tenant": get_sqm_per_tenant_SCHL, 
                                "ABZ sqm per tenant": get_sqm_per_tenant_ABZ,
                                "room 1 sqm per tenant": get_sqm_per_tenant_1_room,
                                "room 2 sqm per tenant": get_sqm_per_tenant_2_room,
                                "room 3 sqm per tenant": get_sqm_per_tenant_3_room,
                                "room 4 sqm per tenant": get_sqm_per_tenant_4_room,
                                "room 5 sqm per tenant": get_sqm_per_tenant_5_room,
                                "room 6 sqm per tenant": get_sqm_per_tenant_6_room,
                                "room 7 sqm per tenant": get_sqm_per_tenant_7_room,
                                "F1 available dwellings": get_f1_available_dwellings_number,
                                "F2 available dwellings": get_f2_available_dwellings_number,
                                "F3 available dwellings": get_f3_available_dwellings_number,
                                "F4 available dwellings": get_f4_available_dwellings_number,
                                "F5 available dwellings": get_f5_available_dwellings_number,
                                "F6 available dwellings": get_f6_available_dwellings_number,
                                "F7 available dwellings": get_f7_available_dwellings_number,
                                "F8 available dwellings": get_f8_available_dwellings_number,
                                "F9 available dwellings": get_f9_available_dwellings_number,
                                "room 1 available": get_1_room_available,
                                "room 2 available": get_2_room_available,
                                "room 3 available": get_3_room_available,
                                "room 4 available": get_4_room_available,
                                "room 5 available": get_5_room_available,
                                "room 6 available": get_6_room_available,
                                "room 7 available": get_7_room_available,
                                "room 1 total": get_1_room_total,
                                "room 2 total": get_2_room_total,
                                "room 3 total": get_3_room_total,
                                "room 4 total": get_4_room_total,
                                "room 5 total": get_5_room_total,
                                "room 6 total": get_6_room_total,
                                "room 7 total": get_7_room_total,
                                "household functions number average": get_household_functions_average,
                                "dwelling functions number average": get_dwelling_functions_average,
                                "F1 total dwellings": get_f1_total_dwellings_number,
                                "F2 total dwellings": get_f2_total_dwellings_number,
                                "F3 total dwellings": get_f3_total_dwellings_number,
                                "F4 total dwellings": get_f4_total_dwellings_number,
                                "F5 total dwellings": get_f5_total_dwellings_number,
                                "F6 total dwellings": get_f6_total_dwellings_number,
                                "F7 total dwellings": get_f7_total_dwellings_number,
                                "F8 total dwellings": get_f8_total_dwellings_number,
                                "F9 total dwellings": get_f9_total_dwellings_number,
                                "F1 los": get_f1_los,
                                "F2 los": get_f2_los,
                                "F3 los": get_f3_los,
                                "F4 los": get_f4_los,
                                "F5 los": get_f5_los,
                                "F6 los": get_f6_los,
                                "F7 los": get_f7_los,
                                "F8 los": get_f8_los,
                                "F9 los": get_f9_los,
                                "num dwellings bright": get_num_dwellings_bright,
                                "num dwellings with balcony": get_num_dwellings_with_balcony,
                                "num dwellings with green spaces": get_num_dwellings_with_green_spaces,
                                "num dwellings with parking place": get_num_dwellings_with_parking_place,
                                "num dwellings work": get_num_dwellings_work,
                                "num dwellings public transports": get_num_dwellings_public_transports,
                                "num dwellings city center": get_num_dwellings_city_center,
                                "num dwellings safe": get_num_dwellings_safe,
                                "num dwellings sociocultural mix": get_num_dwellings_sociocultural_mix,
                                "num dwellings accessible by car": get_num_dwellings_accessible_by_car,
                                "num emigrants": get_emigrants_num,
                                "num immigrants": get_immigrants_num,
                                "type1": get_num_hh_type1,
                                "type2": get_num_hh_type2,
                                "type3": get_num_hh_type3,
                                "type4": get_num_hh_type4,
                                "type5": get_num_hh_type5,
                                "type6": get_num_hh_type6,
                                "type7": get_num_hh_type7,
                                "type8": get_num_hh_type8,
                                "type9": get_num_hh_type9,
                                "type10": get_num_hh_type10,
                                "type11": get_num_hh_type11,
                                "type12": get_num_hh_type12,
                                "type13": get_num_hh_type13,
                                "num tenants mobi": get_num_hh_room3_MOBI,
                                "num tenants schl": get_num_hh_room3_SCHL,
                                "num tenants abz": get_num_hh_room3_ABZ,
                                "1person household abz": get_num_hh_1_ABZ,
                                "1person household schl": get_num_hh_1_SCHL,
                                "1person household mobi": get_num_hh_1_MOBI,
                                "flatshare type 1": get_num_flatshare_type1,
                                "flatshare type 5": get_num_flatshare_type5,
                                "flatshare type 13": get_num_flatshare_type13})
                                
        self.datacollector_hh = DataCollector(
            agent_reporters={
                             "id":get_hh_id,
                             "hhs":get_hh_size,
                             "num adults":get_hh_num_adults,
                             "num minors":get_hh_num_minors,
                             "average adults age":get_hh_adults_age,
                             "type": get_hh_type,
                             "los": get_hh_los,
                             "num desired functions": get_hh_num_desired_functions,
                             "trigger": get_hh_trigger,
                             "dwelling id": get_dwelling_id,
                             "MOV": get_mover,
                             "housing functions": get_desired_functions,
                             "postcode": get_postcode,
                             "owner": get_owner,
                             })                            

    def step(self):
        if self.step_num > 0:
            self.call_once_a_year_functions(month=self.step_num % 12)
        
        self.demolish_buildings_check()
        self.renovate_dwellings_check()
        self.send_environment_triggers()
        
        # alternate migrations / household already in activations
        if np.random.binomial(1, 0.5, 1)[0] == 1:
            self.handle_migrations()
            self.schedule.step()
        else:
            self.schedule.step()
            self.handle_migrations()

        self.datacollector.collect(self) 
        self.datacollector_hh.collect(self)        
        self.step_num += 1
        self.immigrants_num = 0
        self.emigrants_num = 0

    def call_once_a_year_functions(self, month):
        """
        Calls all those functions that need to be called once a year in a random way, e.g not all functions are executed the first month of the year
        """
        if month == 0: # at the beginning of the year define at which month the functions are going to be executed
            self.increase_buildings_age()
            self.create_building_month = np.random.randint(12)
            self.demolish_building_month = np.random.randint(12)
            self.renovate_dwelling_month = np.random.randint(12)
            self.check_cooperatives_month = np.random.randint(12)
            
            self.fire_tenants()

        if month == self.create_building_month:
            self.create_buildings()
        if month == self.demolish_building_month:
            self.demolish_buildings_notify()
        if month == self.renovate_dwelling_month:
            self.renovate_dwellings_notify()            
        if month == self.check_cooperatives_month:
            self.check_cooperatives_compliance_rules()
        
    def fire_tenants(self):
        """
        Notify 3.2 % of tenants that they will be fired. This will really take place when the household they are in calls the function "change_salary"
        """
        tenants_list = [tenant for household in self.households_list for tenant in household.members if tenant.member_type == "adult" and tenant.to_fire == False and tenant.is_unemployed == False]
        num_tenants_to_fire = round(len(tenants_list) * parameters.JOB_LOSS_PROBABILITY)
        sample_tenants_list = np.random.choice(tenants_list, size=num_tenants_to_fire if num_tenants_to_fire < len(tenants_list) else len(tenants_list), replace=False)
        for tenant in sample_tenants_list:
            tenant.to_fire = True
        

    def create_vacant_dwellings(self):
        """
        Create some empty dwellings necessary to have the good vacancy rates for each postcode. 
        Vacancy rates:
         0.4% dwellings with postcode 10--
         0.1% dwellings with postcode 80--
         2.7% for all other postcodes 
        """
        dwellings_1000 = [dwell for dwell in self.dwellings_list if dwell.building.postcode == 1000]    # dwellings with postcode == 1000
        dwellings_8000 = [dwell for dwell in self.dwellings_list if dwell.building.postcode == 8000]
        dwellings_others = [dwell for dwell in self.dwellings_list if dwell.building.postcode != 1000 and dwell.building.postcode != 8000]
        num_dwell_1000_to_add = round(len(dwellings_1000) * parameters.VACANCY_RATE_POSTCODE_1000)
        num_dwell_8000_to_add = round(len(dwellings_8000) * parameters.VACANCY_RATE_POSTCODE_8000)
        num_dwell_others_to_add = round(len(dwellings_others) * parameters.VACANCY_RATE_POSTCODE_OTHERS)
        
        num_dwell_1000_to_add = num_dwell_1000_to_add if num_dwell_1000_to_add > 0 else 1
        num_dwell_8000_to_add = num_dwell_8000_to_add if num_dwell_8000_to_add > 0 else 1
        num_dwell_others_to_add = num_dwell_others_to_add if num_dwell_others_to_add > 0 else 1

        buildings_1000 = [building for building in self.buildings_list if building.postcode == 1000]
        buildings_8000 = [building for building in self.buildings_list if building.postcode == 8000]
        buildings_others = [building for building in self.buildings_list if building.postcode != 1000 and building.postcode != 8000]
        
        for i in range(num_dwell_1000_to_add):
            building = np.random.choice(buildings_1000,1)[0]
            dwelling_agent = DwellingAgent(unique_id=self.get_next_dwelling_id(), model=self, building=building, init=True)
            building.dwellings_list.append(dwelling_agent)      # add it to building internal list of dwellings it contains
            self.dwellings_list.append(dwelling_agent)          # add it to the list of all dwellings present in the model, maybe in the future this redundancy could be removed.

        for i in range(num_dwell_8000_to_add):
            building = np.random.choice(buildings_8000,1)[0]
            dwelling_agent = DwellingAgent(unique_id=self.get_next_dwelling_id(), model=self, building=building, init=True)
            building.dwellings_list.append(dwelling_agent)      # add it to building internal list of dwellings it contains
            self.dwellings_list.append(dwelling_agent)          # add it to the list of all dwellings present in the model, maybe in the future this redundancy could be removed.

        for i in range(num_dwell_others_to_add):
            building = np.random.choice(buildings_others,1)[0]
            dwelling_agent = DwellingAgent(unique_id=self.get_next_dwelling_id(), model=self, building=building, init=True)
            building.dwellings_list.append(dwelling_agent)      # add it to building internal list of dwellings it contains
            self.dwellings_list.append(dwelling_agent)          # add it to the list of all dwellings present in the model, maybe in the future this redundancy could be removed.
        
    def renovate_dwellings_notify(self):
        """
        Renovate 0.9 % of dwellings. Chosen dwellings must be in a building that won't be demolished.
        """
        renovation_rate = parameters.DWELLINGS_RENOVATION_RATE
        num_dwelling_to_renovate = math.ceil(renovation_rate * len(self.dwellings_list))
        dwellings_not_in_demolition = [dwelling for dwelling in self.dwellings_list if not dwelling.building.to_demolish]
        dwellings_to_renovate = np.random.choice(dwellings_not_in_demolition, size=num_dwelling_to_renovate if num_dwelling_to_renovate <= len(dwellings_not_in_demolition) else len(dwellings_not_in_demolition), replace=False)
        for dwelling in dwellings_to_renovate:
            dwelling.to_renovate = True
            dwelling.renovation_notification_counter = parameters.RENOVATION_NOTIFICATION_TIME
            if dwelling.household:
                dwelling.household.insert_new_trigger(Trigger(category="problem solving", origin="environment", reason="renovation/transformation"))

    def renovate_dwellings_check(self):
        """
        Checks if renovations are concluded. If not decrement relocation counter of dwellings that are renovating. 
        Also checks if dwellings that need to be renovated are empty or if 12 months passed from the renovation notification.
        """
        # checks if dwellings with to_renovate == True can be renovated (either because they are empty or because 12 months passed by from notification)
        dwellings_to_renovate = [dwelling for dwelling in self.dwellings_list if dwelling.to_renovate]
        for dwelling in dwellings_to_renovate:
            if dwelling.renovation_notification_counter == 0:    # 12 months passed by from notification
                if dwelling.household:
                    dwelling.household.remove_household()
                dwelling.to_renovate = False
                dwelling.is_renovating = True
                dwelling.renovation_counter = parameters.RENOVATION_TIME
            else:
                if not dwelling.household:
                    dwelling.to_renovate = False
                    dwelling.is_renovating = True
                    dwelling.renovation_counter = parameters.RENOVATION_TIME
                else:                               # decrement counter
                    dwelling.renovation_notification_counter -= 1
        
        # checks if renovations are concluded
        renovating_dwellings = [dwelling for dwelling in self.dwellings_list if dwelling.is_renovating]
        for dwelling in renovating_dwellings:
            if dwelling.renovation_counter == 0:
                dwelling.is_renovating = False
                dwelling.rent_price += dwelling.rooms  *  np.random.choice(range(parameters.MIN_RENT_INCREASE, parameters.MAX_RENT_INCREASE),1)[0]

            else:
                dwelling.renovation_counter -= 1
        
    def add_reason_to_dict_all(self, reason):
        if self.step_num >= parameters.STARTING_MONTH_REASON_COLLECTION:
            if reason in self.reason_dict_all:
                self.reason_dict_all[reason] += 1
            else:   # not in dict
                self.reason_dict_all[reason] = 0

    def add_reason_to_dict_move(self, reason):
        if self.step_num >= parameters.STARTING_MONTH_REASON_COLLECTION:
            if reason in self.reason_dict_move:
                self.reason_dict_move[reason] += 1
            else:   # not in dict
                self.reason_dict_move[reason] = 0

    def dump_data_to_csv(self, model_name, hh_name, reas_name_all, reas_data_move, iter):
        data = self.datacollector.get_model_vars_dataframe()
        data.to_csv (model_name, index = True, header=True)
        data = self.datacollector_hh.get_agent_vars_dataframe()
        data = data.loc[(data["id"] == parameters.HOUSEHOLD1_ID_TO_FOLLOW) | (data["id"] == parameters.HOUSEHOLD2_ID_TO_FOLLOW)]
        data.to_csv (hh_name, index = True, header=True)
        
        if iter == 0:
            with open(reas_name_all, 'w') as f:
                f.write("model_id,")
                for key in self.reason_dict_all.keys():
                    f.write("%s,"%(key))
                f.write("\n")
        # dump reason dict to csv
        with open(reas_name_all, 'a') as f:
            f.write("%s,"%(iter))
            for key in self.reason_dict_all.keys():
                f.write("%s,"%(self.reason_dict_all[key]))
            f.write("\n")

        if iter == 0:
            with open(reas_data_move, 'w') as f:
                f.write("model_id,")
                for key in self.reason_dict_move.keys():
                    f.write("%s,"%(key))
                f.write("\n")
        # dump reason dict to csv
        with open(reas_data_move, 'a') as f:
            f.write("%s,"%(iter))
            for key in self.reason_dict_move.keys():
                f.write("%s,"%(self.reason_dict_move[key]))
            f.write("\n")

    def init_birth_curve_coeff(self):
        """
        Initialize coefficients of the curve that approximate probability of having a child at a given age.
        """
        # x and y are datapoints coordinate from the curve we want to approximate (in this case the birth curve)
        x = [15,21,24,26,29,32,35,36,38,40,41,49]
        y = [0.0, 0.02, 0.04, 0.06, 0.1, 0.115, 0.1, 0.08, 0.06, 0.04, 0.02, 0]
        # p0 is the initial guess for the fitting coefficients (a,b,c), it doesn't really matters as long as it is not 0!
        p0 = [0.1,0.1,0.1]
        coeff,_ = curve_fit(get_child_probability, xdata=x, ydata=y , p0=p0, bounds=(0, 101))
        return coeff

    def send_environment_triggers(self):
        """
        Send triggers to households of tenants. (Every year)
        """            
        for household in self.households_list:
            if np.random.binomial(1, parameters.CHANGE_JOB_LOCATION_PROBABILITY, 1)[0] == 1:       # change job location 
                household.insert_new_trigger(Trigger(category="radical change", origin="environment", reason="change job location"))
            
            if np.random.binomial(1, parameters.NEED_FOR_CHANGE_PROBABILITY, 1)[0] == 1:           # need for a change
                household.insert_new_trigger(Trigger(category="radical change", origin="environment", reason="need for a change"))
            
            if np.random.binomial(1, parameters.INTERPERSONAL_PROBLEMS_PROBABILITY, 1)[0] == 1:    # interpersonal problems
                household.insert_new_trigger(Trigger(category="problem solving", origin="environment", reason="interpersonal problems"))

            if household.time > 12 and household.current_dwelling.get_owner() == "MOBI" and np.random.binomial(1, parameters.EXPIRE_RENTAL_CONTRACT_PROBABILITY, 1)[0] == 1:    # expire rental contract
                household.insert_new_trigger(Trigger(category="problem solving", origin="environment", reason="expire contract"))

    def increase_buildings_age(self):
        """
        Increase age attribute of all the buildings
        """
        for building in self.buildings_list:
            building.age += 1

    def create_buildings(self):
        num_buildings_to_create = math.ceil(parameters.BUILDINGS_CONSTRUCTION_RATE * len(self.buildings_list))

        # create buildings
        for i in range(num_buildings_to_create):
            building_agent = BuildingAgent(unique_id=self.get_next_building_id(), model=self, dwellings_num=np.random.choice(range(2,121),1)[0], owner=np.random.choice(["ABZ","SCHL","MOBI"],1)[0])
            self.buildings_list.append(building_agent)
                       
            # create dwellings
            for i in range(building_agent.dwellings_num):
                dwelling_agent = DwellingAgent(unique_id=self.get_next_dwelling_id(), model=self, building=building_agent)
                building_agent.dwellings_list.append(dwelling_agent)      # add it to building internal list of dwellings it contains
                self.dwellings_list.append(dwelling_agent)          # add it to the list of all dwellings present in the model, maybe in the future this redundancy could be removed.

            households_in_postcode_of_building = [household for household in self.households_list if household.current_dwelling.building.postcode == building_agent.postcode and household.satisfaction <= 4] # if postcode is the same then send trigger that there is a new building
            for household in households_in_postcode_of_building:
                household.insert_new_trigger(Trigger(category="opportunity", origin="environment", reason="new building", new_building=building_agent))
            
    def demolish_buildings_notify(self):
        """
        Notifies buildings occupants if their building is going to be demolished
        """
        num_buildings_to_demolish = math.ceil(parameters.BUILDINGS_DEMOLITION_RATE * len(self.buildings_list))
        
        # keep only old buildings (age > 30)
        copy = [building for building in self.buildings_list if building.age > 30 and not building.to_demolish] # second condition to avoid double notification of demolition

        for building in copy:
            if num_buildings_to_demolish > 0:
                building.notify_demolition()
                num_buildings_to_demolish -= 1
            else:
                break
            
    def demolish_buildings_check(self):
        """
        Checks if buildings with to_demolish = True need to be demolished. This can happen if 12 months passed by the notification or because there are no more households in the building
        """
        buildings_to_demolish = [building for building in self.buildings_list if building.to_demolish]
        for building in buildings_to_demolish:
            if building.demolition_counter == 0:    # 12 months passed by from notification
                building.remove_building()
            else:
                num_households_in_building = len([dwelling.household for dwelling in building.dwellings_list if dwelling.household])
                if num_households_in_building == 0:     # no morehouseholds in the building
                    building.remove_building()
                else:                               # decrement counter
                    building.demolition_counter -= 1
            
    def set_buildings(self):
        """
        Init buildings.
        """
        dwellings_num_list = self.get_dwellings_num_per_building()
        owners_list = self.get_owners_list()

        for i in range(parameters.NUM_BUILDINGS):
            building_agent = BuildingAgent(unique_id=self.get_next_building_id(), model=self, dwellings_num=dwellings_num_list[i], owner=owners_list[i], init=True)
            self.buildings_list.append(building_agent)
           
    def set_dwellings(self):
        """
        Create and distribute dwellings over available buildings.
        """        
        for building in self.buildings_list:
            for i in range(building.dwellings_num):
                dwelling_agent = DwellingAgent(unique_id=self.get_next_dwelling_id(), model=self, building=building, init=True)
                building.dwellings_list.append(dwelling_agent)      # add it to building internal list of dwellings it contains
                self.dwellings_list.append(dwelling_agent)          # add it to the list of all dwellings present in the model, maybe in the future this redundancy could be removed.
  
    def set_households(self):
        """
        Create and distribute households over available dwellings. Used only at init!
        """
        p = [0, 0.05, 0.085, 0.032, 0.006, 0.103, 0.077, 0.069, 0.19, 0.054, 0.058, 0.044, 0.111, 0.121]  # probability of being 1, for each TYPE (first element is 0 (could have been anything) as need indexes to start at 1)

        for i in range(parameters.NUM_HOUSEHOLDS):
            id_household = self.get_next_household_id()
            TYPE = np.random.choice(np.arange(1, len(p)), p=p[1:], size=1)[0]

            household_agent = HouseholdAgent(unique_id=id_household, model=self, TYPE=TYPE, init=True)
            iters = 0
            available_dwellings = household_agent.get_dwelling()            # get free dwellings
            while len(available_dwellings) == 0:                        # while there is not a free building
                TYPE = np.random.choice(np.arange(1, len(p)), p=p[1:], size=1)[0]
                household_agent = HouseholdAgent(unique_id=id_household, model=self, TYPE=TYPE, init=True)
                if iters > parameters.MAX_ITERS_FOR_ROOMS_CONDITIONS:
                    available_dwellings = household_agent.get_dwelling(upper_room_condition=False)       # get free dwellings (only lower room condition)
                else:
                    available_dwellings = household_agent.get_dwelling(upper_room_condition=True)        # get free dwellings (both room conditions)
                iters += 1

            self.add_household_to_model(dwelling=available_dwellings[0], household=household_agent, init=True)


    def get_dwellings_num_per_building(self):
        """
        Returns a list of random integers that sums up to a fixed number (in this case the sum must be parameters.NUM_HOUSEHOLDS because we don't want any vacant dwelling. Vacant dwellings will be created in a second phase).
        At index i of the list there is the number of dwellings to create for building i
        """
        n = parameters.NUM_BUILDINGS
        dwellings_arr = np.random.multinomial(parameters.NUM_HOUSEHOLDS, np.ones(n)/n, size=1)[0]
        return dwellings_arr
   
    def get_owners_list(self):
        """
        Return a list with owner for each building. At index i -> owner for building i.
        Distributed such that: 33.5% ABZ(1), 39.5% SCHL(2), 27% Mobi(3)
        """
        num_buildings = parameters.NUM_BUILDINGS
        abz = round(num_buildings * 0.335)
        schl = round(num_buildings * 0.395)
        mobi = round(num_buildings * 0.27)
        owners = ["ABZ"]*abz + ["SCHL"]*schl + ["MOBI"]*mobi
        np.random.shuffle(owners)
        return owners

    def get_buildings_num(self):
        return len(self.buildings_list)

    def get_next_tenant_id(self):
        """
        Generator of sequential IDs for tenants
        """
        tmp = self.tenant_id
        self.tenant_id += 1
        return tmp

    def get_next_household_id(self):
        """
        Generator of sequential IDs for households
        """
        tmp = self.household_id
        self.household_id += 1
        return tmp

    def get_next_dwelling_id(self):
        """
        Generator of sequential IDs for dwellings
        """
        tmp = self.dwelling_id
        self.dwelling_id += 1
        return tmp

    def get_next_building_id(self):
        """
        Generator of sequential IDs for buildings
        """
        tmp = self.building_id
        self.building_id += 1
        return tmp

    def handle_migrations(self):
        total_num_households = len(self.households_list)
        immigrating_num = round(total_num_households * parameters.IMMIGRATION_RATE)
        i = 0

        while i < immigrating_num:            
            id_household = self.get_next_household_id()
            household_agent = HouseholdAgent(unique_id=id_household, model=self)
            
            available_dwellings = household_agent.get_dwelling() # get free dwellings
            
            if len(available_dwellings) > 0:
                self.immigrants_num += 1
                self.add_household_to_model(dwelling=available_dwellings[0], household=household_agent)
            
            i += 1


    def add_household_to_model(self, dwelling, household, init=False):
        """
        Adds household to model.
        """
        self.schedule.add(household)
        if household not in self.households_list:
            self.households_list.append(household)
        if household not in self.full_household_list:
            self.full_household_list.append(household)
        household.relocate_to_new_dwelling(dwelling)
        if init:
            household.init_time()

    def check_cooperatives_compliance_rules(self):
        """
        Check if the compliance rules of cooperatives are respected.
        """
        abz_big_dwellings = [dwell for dwell in self.dwellings_list if dwell.building.owners_type == "ABZ" and not dwell.is_empty() and dwell.rooms >= 4]
        schl_big_dwellings = [dwell for dwell in self.dwellings_list if dwell.building.owners_type == "SCHL" and not dwell.is_empty() and dwell.rooms >= 4]
        
        for dwelling in abz_big_dwellings:
            if dwelling.household.get_household_size() < dwelling.rooms - 2:
                dwelling.household.insert_new_trigger(Trigger(category="problem solving", origin="environment", reason="underoccupancy"))
        for dwelling in schl_big_dwellings:
            if dwelling.household.get_household_size() < dwelling.rooms - 2:
                dwelling.household.insert_new_trigger(Trigger(category="problem solving", origin="environment", reason="underoccupancy"))


    def assign_household_to_tenant(self, tenant, leaving_flatshare=False):
        """
        Assign tenant to a household, the household can be already existing or it can create a new household in which it is alone.
        The tenants look in "NUM_ATTEMPTS_TO_FIND_PARTNER" households randomly and, if none of them is right for it(meaning it doesn't create a couple or find a flatshare), it creates one.
        """
        found_existing_household = False
        tenant.remove_tenant_from_household()
        
        # Look for a random group to join, if the age of the adults is within a +/- 10 y.o. then join them 
        households_potentially_joinable = [household for household in self.households_list if household.TYPE in set([1,4,5,7,10,11,13])]
        np.random.shuffle(households_potentially_joinable)    # shuffle list otherwise always the same at the beginning
        potentially_joinable_num = len(households_potentially_joinable)
        i = 0
        while i < parameters.NUM_ATTEMPTS_TO_FIND_PARTNER and i < potentially_joinable_num and not found_existing_household:
            household_agent = households_potentially_joinable[i]
            adult = household_agent.get_adult()                   # returns the only adult in the household (because TYPE is in set([1,4,5,7,10,11,13]))
            if tenant.age-10 <= adult.age <= tenant.age+10 and tenant.id != adult.id: # if the two adults have similar age (i.e with at max 10 years old of difference)
                if (household_agent.TYPE in set([1,5,13]) and household_agent.get_household_size() < parameters.MAX_HOUSEHOLD_SIZE and household_agent.get_minors_num() == 0) or \
                    (household_agent.TYPE not in set([1,5,13]) and household_agent.get_adults_num() < 2):
                    if household_agent.current_dwelling and household_agent.get_household_size() <= household_agent.current_dwelling.rooms: # even if someone else join it's fine
                        if leaving_flatshare:
                            self.add_reason_to_dict_move("leaving the flatshare")
                                                
                        # in a flatshare you are not forcing the whole flatmates to move just because you join (in a couple it can happens)
                        # only half of the times in a flatshare that has less than 2 members the 'create couple' trigger is sent
                        if (household_agent.TYPE not in set([1,5,13])) or \
                            (household_agent.TYPE in set([1,5,13]) and household_agent.get_adults_num() < 2 and np.random.binomial(1, 0.5, 1)[0] == 1):
                            household_agent.insert_new_trigger(Trigger(category="radical change", origin="internal", reason="create couple"))
                        
                        household_agent.add_member(tenant)
                        found_existing_household = True
            i += 1

        if not found_existing_household:    # create a new household alone, if it can't find anything it simply leaves the model
            id_household = self.get_next_household_id()
            
            household_agent = HouseholdAgent(unique_id=id_household, model=self, tenant=tenant)
            available_dwellings = household_agent.get_dwelling()                # get free dwellings
            
            if len(available_dwellings) > 0:
                new_dwelling = available_dwellings[0]
                if leaving_flatshare:
                    self.add_reason_to_dict_move("leaving the flatshare")
                self.add_household_to_model(dwelling=new_dwelling, household=household_agent)