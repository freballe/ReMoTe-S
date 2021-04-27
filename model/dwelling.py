from mesa import Agent
import numpy as np
from metrics import get_truncated_normal
import parameters

class DwellingAgent(Agent):
    def __init__(self, unique_id, model, building, init=False):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.new_constructed = not init  # indicate if it has been constructed at init or not
        self.rooms = round(get_truncated_normal(parameters.MEAN_ROOMS, sd=1.0, low=1, upp=10))
        self.size = self.get_size_from_number_rooms(self.rooms)     
        self.rent_price = self.size * np.random.normal(loc=15.9, scale=5.27) * 12       # RP = DS*rentsqm; rentsqm (average 15.9; sd 5.27)
        self.characteristics = self.get_characteristics(init) # 1= bright; 2= with balcony; 3=with green spaces; 4= with parking place; 5= with low rent
        self.household = None       # occupying household
        self.building = building
        self.is_renovating = False  # True when dwelling is in renovation (duration 3 months)
        self.to_renovate = False    # True when the owner has decided to renovate the dwelling, but it needs to leave some time (12 months) to the household to leave the house
        self.renovation_counter = parameters.RENOVATION_TIME                            # counts the months left till the end of renovation
        self.renovation_notification_counter = parameters.RENOVATION_NOTIFICATION_TIME  # counts the months left to the household to leave the house
        self.functions = self.get_functions(init)
        
    def is_new(self):
        return self.new_constructed

    def is_empty(self):
        if self.household:
            return False
        return True
    
    def get_owner(self):
        return self.building.owners_type

    def get_postcode(self):
        return self.building.postcode

    def get_size_from_number_rooms(self, rooms):
        if rooms == 1:
            return np.random.normal(loc=parameters.MEAN_DS_1_ROOM, scale=parameters.STDDEV_DS_1_ROOM, size=1)[0]
        elif rooms == 2:
            return np.random.normal(loc=parameters.MEAN_DS_2_ROOM, scale=parameters.STDDEV_DS_2_ROOM, size=1)[0]
        elif rooms == 3:
            return np.random.normal(loc=parameters.MEAN_DS_3_ROOM, scale=parameters.STDDEV_DS_3_ROOM, size=1)[0]
        elif rooms == 4:
            return np.random.normal(loc=parameters.MEAN_DS_4_ROOM, scale=parameters.STDDEV_DS_4_ROOM, size=1)[0]
        elif rooms == 5:
            return np.random.normal(loc=parameters.MEAN_DS_5_ROOM, scale=parameters.STDDEV_DS_5_ROOM, size=1)[0]
        elif rooms == 6:
            return np.random.normal(loc=parameters.MEAN_DS_6_ROOM, scale=parameters.STDDEV_DS_6_ROOM, size=1)[0]
        else:
            return np.random.normal(loc=parameters.MEAN_DS_7_ROOM, scale=parameters.STDDEV_DS_7_ROOM, size=1)[0]

    def get_characteristics(self, init):
        # default case
        characteristics_number = np.random.choice(range(1,5),1)[0]
        characteristics = ["bright", "with balcony", "with green spaces", "with parking place"]
        
        if self.model.scenario == "A3":
            if not init:    # new dwellings
                characteristics_number = np.random.choice(range(1,2),1)[0]
                characteristics = ["with green spaces"]
        elif self.model.scenario == "B7":
            if not init:    # new dwellings
                characteristics_number = np.random.choice(range(1,4),1)[0]
                characteristics = ["bright", "with balcony", "with parking place"]
        elif self.model.scenario == "B3":
            if not init:    # new dwellings
                characteristics_number = np.random.choice(range(1,2),1)[0]
                characteristics = ["with parking place"]
        elif self.model.scenario == "A7":
            if not init:    # new dwellings
                characteristics_number = np.random.choice(range(1,4),1)[0]
                characteristics = ["bright", "with balcony", "with green spaces"]
        elif self.model.scenario == "A1":
            if not init:    # new dwellings
                characteristics_number = np.random.choice(range(1,5),1)[0]
                characteristics = ["bright", "with balcony", "with green spaces", "with parking place"]
        elif self.model.scenario == "B1":
            if not init:    # new dwellings
                characteristics_number = np.random.choice(range(1,5),1)[0]
                characteristics = ["bright", "with balcony", "with green spaces", "with parking place"]

        return set(np.random.choice(characteristics, size=characteristics_number if characteristics_number <= len(characteristics) else len(characteristics), replace=False))


    def get_all_characteristics(self):
        """
        Return the set of all charactersitic of a dwelling, i.e. both the ones internal to the dwelling and those of the building in which the dwelling is.
        """
        return self.characteristics | self.building.places_of_interest | self.building.neighborhood 
                
    def get_functions(self, init=False):
        """
        Return dwelling functions following specific probabilities. 
        If a dwelling or its building has some caracteristics then we sample, for each function, 
        from the binomial with probability that depends on the dictionary below and if at least one of the samples 
        returns 1 then the dwelling has that function.
        """
        functions_to_return = set()     
        functions_dict = parameters.DWELLING_FUNCTIONS_PROBABILITIES
        functions_num = int(round(get_truncated_normal(mean=5.0, sd=1.7, low=1, upp=9)))

        while len(functions_to_return) == 0:    # dwellings must have at least one function
            for function in range(1,10):
                binomial_sum = 0
                for characteristic, probabilities in functions_dict.items():
                    if characteristic in self.characteristics or \
                    characteristic in self.building.places_of_interest or \
                    characteristic in self.building.neighborhood:
                        binomial_sum += np.random.binomial(1, probabilities[function-1], 1)[0]  # this is 1 when the binomial with probability related to the i-th dwelling function returns 1
                
                if binomial_sum > 0:    # at least one binomial returned 1
                    functions_to_return.add(function)     # add e.g "bright" to the dwelling chars

            if len(functions_to_return) > functions_num:
                functions_to_return = set(np.random.choice(list(functions_to_return), size=functions_num if functions_num<=len(functions_to_return) else len(functions_to_return), replace=False))
                
        return functions_to_return

    def remove_dwelling(self):
        """
        Removes dwelling from model.
        """
        if self.household:
            self.household.remove_household()
        self.model.dwellings_list.remove(self)