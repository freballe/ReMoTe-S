import numpy as np
from mesa import Agent
from trigger import Trigger

class BuildingAgent(Agent):
    def __init__(self, unique_id, model, dwellings_num, owner, init=False):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.dwellings_num = dwellings_num
        self.dwellings_list = [] # dwellings in the building
        self.owners_type = owner # each building is owned by only one company(1 = ABZ 2 = SCHL 3 = MOBI)

        # define postcode
        if self.owners_type == "ABZ":
            self.postcode = 8000
        elif self.owners_type == "SCHL":
            self.postcode = 1000
        else:
            self.postcode = np.random.choice(range(1000,10000,1000),1)[0]
        
        self.neighborhood = self.get_neighborhood(init) # safe; charming; lively and popular; with schools of good reputation; socio-cultural mixity
        self.places_of_interest = self.get_places_of_interest(init) # work; public transports; family; cultural activities
        self.age = np.random.choice(range(0,40),1)[0] # years from construction     

        self.to_demolish = False # if True, demolish building when 12 months have passed or when all households have left 
        self.demolition_counter = 12 # when demolition counter goes to zero, demolish building


    def get_dwellings_num(self):
        return self.dwellings_num

    
    def get_neighborhood(self, init):    
        neighborhood_chars_number = np.random.choice(range(0,4),1)[0] # number of neighborhood characteristics the building has
        neighborhood_chars = ["safe", "sociocultural mix", "accessible by car"] # all possible chars

        if self.model.scenario == "A3":
            if not init:
                neighborhood_chars_number = np.random.choice(range(0,2),1)[0] # number of neighborhood characteristics the building has
                neighborhood_chars = ["sociocultural mix"]
        elif self.model.scenario == "B7":
            if not init:
                neighborhood_chars_number = np.random.choice(range(0,3),1)[0] # number of neighborhood characteristics the building has
                neighborhood_chars = ["safe", "accessible by car"]
        if self.model.scenario == "A7":
            if not init:
                neighborhood_chars_number = np.random.choice(range(0,3),1)[0] # number of neighborhood characteristics the building has
                neighborhood_chars = ["safe", "sociocultural mix"]
        elif self.model.scenario == "B3":
            if not init:
                neighborhood_chars_number = np.random.choice(range(0,2),1)[0] # number of neighborhood characteristics the building has
                neighborhood_chars = ["accessible by car"]
        if self.model.scenario == "A1":
            if not init:
                neighborhood_chars_number = np.random.choice(range(0,3),1)[0] # number of neighborhood characteristics the building has
                neighborhood_chars = ["safe", "sociocultural mix"]
        elif self.model.scenario == "B1":
            if not init:
                neighborhood_chars_number = np.random.choice(range(0,4),1)[0] # number of neighborhood characteristics the building has
                neighborhood_chars = ["safe", "sociocultural mix", "accessible by car"]
        
        chars = np.random.choice(neighborhood_chars, size=neighborhood_chars_number if neighborhood_chars_number<=len(neighborhood_chars) else len(neighborhood_chars), replace=False)
        return set(chars) # sample subset


    def get_places_of_interest(self, init):        
        POI_number = np.random.choice(range(0,4),1)[0] if len(self.neighborhood) > 0 else np.random.choice(range(1,4),1)[0] # number of places of interest(POI) the building is close to, at least one caracteristic(between those of neighborhood and POI) for the buildings
        POI = ["work", "public transports", "city center"] # all possible POI
        
        if self.model.scenario == "A3":
            if not init:
                POI_number = np.random.choice(range(0,2),1)[0] if len(self.neighborhood) > 0 else 1 # number of places of interest(POI) the building is close to, at least one caracteristic(between those of neighborhood and POI) for the buildings
                POI = ["public transports"]  
        elif self.model.scenario == "B7":
            if not init:
                POI_number = np.random.choice(range(0,3),1)[0] if len(self.neighborhood) > 0 else np.random.choice(range(1,3),1)[0] # number of places of interest(POI) the building is close to, at least one caracteristic(between those of neighborhood and POI) for the buildings
                POI = ["work", "city center"] 
        if self.model.scenario == "A7":
            if not init:
                POI_number = np.random.choice(range(0,3),1)[0] if len(self.neighborhood) > 0 else np.random.choice(range(1,3),1)[0] # number of places of interest(POI) the building is close to, at least one caracteristic(between those of neighborhood and POI) for the buildings
                POI = ["public transports", "city center"]  
        elif self.model.scenario == "B3":
            if not init:
                POI_number = np.random.choice(range(0,2),1)[0] if len(self.neighborhood) > 0 else 1 # number of places of interest(POI) the building is close to, at least one caracteristic(between those of neighborhood and POI) for the buildings
                POI = ["work"] 
        if self.model.scenario == "A1":
            if not init:
                POI_number = np.random.choice(range(0,4),1)[0] if len(self.neighborhood) > 0 else np.random.choice(range(1,4),1)[0] # number of places of interest(POI) the building is close to, at least one caracteristic(between those of neighborhood and POI) for the buildings
                POI = ["work", "public transports", "city center"]  
        elif self.model.scenario == "B1":
            if not init:
                POI_number = np.random.choice(range(0,3),1)[0] if len(self.neighborhood) > 0 else np.random.choice(range(1,3),1)[0] # number of places of interest(POI) the building is close to, at least one caracteristic(between those of neighborhood and POI) for the buildings
                POI = ["work", "city center"] 
        
        chars = np.random.choice(POI, size=POI_number if POI_number <= len(POI) else len(POI), replace=False)
        return set(chars)


    def is_coop(self):
        if self.owners_type == "ABZ" or self.owners_type == "SCHL":
            return True
        return False

    def remove_building(self):
        """
        Removes building from model.
        """
        for dwelling in self.dwellings_list:
            dwelling.remove_dwelling()
        self.model.buildings_list.remove(self)


    def notify_demolition(self):
        """
        Notify all occupying households that the building is going to be demolished. (From that moment they have 12 months)
        """
        self.to_demolish = True

        # send trigger to all households
        for dwelling in self.dwellings_list:
            if dwelling.household:
                dwelling.household.insert_new_trigger(Trigger(category="problem solving", origin="environment", reason="demolition"))