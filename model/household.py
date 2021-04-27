from mesa import Agent
import numpy as np
from tenant import TenantAgent
from trigger import Trigger
from metrics import get_truncated_normal, get_child_probability
from numpy.random import RandomState
import parameters

class HouseholdAgent(Agent):
    def __init__(self, unique_id, model, TYPE=None, tenant=None, init=False):
        super().__init__(unique_id, model)
        self.id = unique_id        
        self.previous_TYPE = 0
        self.mover = False                                            # mover is 1 if the tenant is looking for a new dwelling, but not found one yet
        self.trigger = None                                     
        self.months_waited_since_mover = 0                            # the number of months that the household has waited for a new dwelling since it decided to move
        self.time = 0                                               # rent time
        self.TYPE = None
        if TYPE:    # if TYPE is not given as parameter, it will be defined in create_members (using the other household fields to infer it)
            self.TYPE = TYPE

        self.members = []                                           # init in create_members
        self.current_dwelling = None
        self.create_members(tenant=tenant, init=init, TYPE=TYPE)    # set of members(adults and minors) of the household
        self.desired_functions = self.sample_desired_functions()    # set of integers, each integer corresponds to a function
        self.satisfaction = parameters.INIT_LOS                     # initialized when relocating

        # vars needed for function calls that happen once a year
        self.separate_adult_month = np.random.randint(12)
        self.change_salary_month = np.random.randint(12)
        self.get_new_child_month = np.random.randint(12)
        self.elderlies_leave_model_month = np.random.randint(12)


    ###################################################
    #                    HELPERS                      #
    ###################################################

    def get_reason(self):
        return self.trigger.reason

    def is_moving(self):
        return self.mover

    def get_household_size(self):
        return len(self.members)

    def get_adults_num(self):
        return len([member for member in self.members if member.member_type == "adult"])
    
    def get_minors_num(self):
        return len([member for member in self.members if member.member_type == "minor"])
    
    def get_adult(self):
        """
        Return a random adult from the household
        """
        adults = [member for member in self.members if member.member_type == "adult"]
        avg = np.random.choice(adults, 1)[0]
        return avg

    def init_time(self):
        """
        Used only at init in add_household_to_model (model.py)
        """
        time = round(get_truncated_normal(mean=(8.8*12), sd=(9.76*12), low=0, upp=65*12))

        while time >= self.get_adults_average_age():
            time = round(get_truncated_normal(mean=(8.8*12), sd=(9.76*12), low=0, upp=65*12))
        return time
    
    def increment_waiting_counter(self):
        self.months_waited_since_mover += 1       # increment months waited from first decision to move

    def get_adults_average_age(self):
        """
        The age of a household is the average age of all the adults member in that household.
        """
        age_list = []
        for member in self.members:
            if member.member_type == "adult":
                age_list.append(member.age)
        return round(np.mean(age_list))

    def get_household_salary(self):
        """
        Return the sum of the salaries of members of the household
        """
        salaries = [t.salary for t in self.members]
        return sum(salaries)


    def step(self):
        """
        Called at each step of the model 
        """
        if self.mover:
            self.increment_waiting_counter()

        if self.months_waited_since_mover >= parameters.MAX_MONTH_TO_WAIT_SINCE_MOVER:
            if self.trigger and ((self.trigger == "underoccupancy" and self.months_waited_since_mover > 24) or self.trigger != "underoccupancy"): # 24 months of time when underoccupancy
                self.model.emigrants_num += 1
                self.remove_household()
        else:
            self.check_TYPE_triggers()
            self.satisfaction -= parameters.SATISFACTION_MONTHLY_DECREASE
            self.satisfaction = max(self.satisfaction, parameters.MIN_LOS)
            members_copy = self.members.copy()
            for member in members_copy:            
                member.step()
            
            if len(self.members) > 0:
                if self.get_adults_num() <= 0:
                    self.remove_household()
                else:
                    if (self.model.schedule.steps % 12) == 0:
                        self.time += 1
                    self.call_once_a_year_functions(month=self.model.schedule.steps % 12)
                    self.check_internal_triggers()
                    self.handle_trigger()

                    if self.mover and self.months_waited_since_mover > 0:
                        self.selection_of_the_dwelling()
            else:
                self.remove_household()
        
    def call_once_a_year_functions(self, month):
        """
        Calls all those functions that need to be called once a year in a random way, e.g not all functions are executed the first month of the year
        """
        if month == 0: # at the beginning of the year define at which month the functions are going to be executed
            self.separate_adult_month = np.random.randint(12)
            self.change_salary_month = np.random.randint(12)
            self.get_new_child_month = np.random.randint(12)
            self.elderlies_leave_model_month = np.random.randint(12)
        
        if month == self.elderlies_leave_model_month:
            self.elderlies_leave_model()
        if month == self.separate_adult_month:
            self.separate_adult()
        if month == self.change_salary_month:
            self.change_salary()
        if month == self.get_new_child_month:
            self.get_new_child()

    def elderlies_leave_model(self):
        """
        With a given probability, elders with TYPE 12 and 13 leave the model
        """
        if self.TYPE in set([12,13]) and np.random.binomial(1, parameters.ELDERLIES_LEAVE_MODEL_PROBABILITY, 1)[0] == 1:
            self.remove_household()

    def change_salary(self):
        """
        Increase or decrease salary for each of the members of the household
        """
        for member in self.members:
            if member.to_fire and not member.is_unemployed: # if it is already unemployed leave it alone!
                member.decrease_salary()
            else:
                member.salary += member.salary * parameters.YEARLY_SALARY_INCREASE # increase salary by parameters.YEARLY_SALARY_INCREASE percent
                if self.satisfaction <= 4 and np.random.binomial(1, parameters.MOVE_AFTER_SALARY_INCREASE, 1)[0] == 1:
                    self.insert_new_trigger(Trigger(category="opportunity", origin="environment", reason="increase salary"))

    def separate_adult(self):
        """
        Separate one adult from the household if binomial of divorce or flatshare leaving probability returns 1
        """
        if self.get_adults_num() == 2 and self.get_TYPE() not in set([1,5,13]):     # TYPE 1, 5 and 13 are the flatshare TYPE and therefore shouldn't divorce
            if np.random.binomial(1, parameters.DIVORCE_PROBABILITY, 1)[0] == 1:
                adult = self.get_adult()
                adult.divorced = True

                self.insert_new_trigger(Trigger(category="radical change", origin="internal", reason="separate/divorce"))
                adult.become_indipendent()
        elif self.get_TYPE() in set([1,5,13]) and self.get_adults_num() > 1:
            if np.random.binomial(1, parameters.LEAVE_FLATSHARE_PROBABILITY, 1)[0] == 1:
                adult = self.get_adult()
                self.model.add_reason_to_dict_all("leaving the flatshare")
                adult.become_indipendent(leaving_flatshare=True)


    def handle_trigger(self):
        """
        Dispatch the different kind of triggers that could occur and modify household attributes accordingly
        """
        trigger = self.trigger
        
        if trigger and not self.mover:
            if self.get_reason() != "new building":
                self.model.add_reason_to_dict_all(self.get_reason()) # needed to keep track of the reason count
            if self.get_reason() == "interpersonal problems" and \
                self.get_adults_num() > 1 and \
                np.random.choice([True, False],1)[0]: # if True it means this is an internal interpersonal problem i.e between the members of the household and therefore the number of adults need to be > 1 to make sense
                
                adult = self.get_adult()
                adult.become_indipendent()
                self.trigger = None # the adult is gone so delete the trigger
                self.mover = False
            else: # for all other cases make the household change dwelling
                self.mover = True
                if not trigger.reason in set(["salary increase", "expire contract", "new building", "renovation/transformation", "demolition", "interpersonal problems", "rent too high", "underoccupancy", "family"]):
                    self.desired_functions = self.sample_desired_functions()


    def selection_of_the_dwelling(self):
        """
        # Look for a new dwelling, if there is one available then move to it, else stay one more month in the old one
        """
        if self.get_reason() in set(["salary increase", "expire contract", "interpersonal problems", "rent too high", "underoccupancy", "need for a change"]):
            available_dwellings = self.get_dwelling()
            if len(available_dwellings) > 0:            # there is a free building
                self.model.add_reason_to_dict_move(self.get_reason())  # needed to keep track of the reason count
                self.relocate_to_new_dwelling(available_dwellings[0])
            
        elif self.get_reason() in set(["family", "change job location"]):
            # search dwelling with different postcode
            available_dwellings = self.get_dwelling(postcode=self.current_dwelling.building.postcode)
            if len(available_dwellings) > 0:            # there is a free building
                self.model.add_reason_to_dict_move(self.get_reason())  # needed to keep track of the reason count
                self.relocate_to_new_dwelling(available_dwellings[0])
            
        elif self.get_reason() == "new building":
            # apply only to that building, otherwise stay where you are
            dwellings_available_in_new_building = [dwell for dwell in self.trigger.new_building.dwellings_list if dwell.is_empty() and self.check_conditions(dwell)]
            self.model.add_reason_to_dict_all(self.get_reason())  # needed to keep track of the reason count
            if len(dwellings_available_in_new_building) > 0:
                self.model.add_reason_to_dict_move(self.get_reason())  # needed to keep track of the reason count
                self.relocate_to_new_dwelling(dwellings_available_in_new_building[0])
            else:   # just remove the trigger and don't move anymore
                self.mover = False
                self.trigger = None
                self.months_waited_since_mover = 0
        else:
            if self.current_dwelling.get_owner() == "MOBI":
                available_dwellings = self.get_dwelling()
                if len(available_dwellings) > 0:            # there is a free building
                    self.model.add_reason_to_dict_move(self.get_reason())  # needed to keep track of the reason count
                    self.relocate_to_new_dwelling(available_dwellings[0])
            else:   # ABZ, SCHL
                # the cooperative assign dwelling to household, and not like for MOBI where the household itself search a dwelling
                cooperatives_available_dwellings = [dwell for dwell in self.model.dwellings_list if dwell.is_empty() and self.check_conditions(dwell)]
                if len(cooperatives_available_dwellings) > 0:
                    self.model.add_reason_to_dict_move(self.get_reason())  # needed to keep track of the reason count
                    self.relocate_to_new_dwelling(cooperatives_available_dwellings[0])

    def check_internal_triggers(self):
        """
        Checks if there are some triggers that are created from the current status of the household.
        """
        if self.get_household_salary() < round(0.33 * self.current_dwelling.rent_price):
            self.insert_new_trigger(Trigger(category="problem solving", origin="internal", reason="rent too high"))
             
        if np.random.binomial(1, parameters.MOVE_FOR_FAMILY_PROBABILITY, 1)[0] == 1:   # move for family
            self.insert_new_trigger(Trigger(category="problem solving", origin="internal", reason="family"))

    def relocate_to_new_dwelling(self, new_dwelling):
        """
        Change all the relocation parameters in both household and building/dwelling
        """
        if self.current_dwelling:           # means that the household is moving from another dwelling
            self.current_dwelling.household = None
        
        self.months_waited_since_mover = 0
        self.current_dwelling = new_dwelling
        self.current_dwelling.household = self
        self.time = 0
        self.mover = False
        self.trigger = None
        self.satisfaction = self.get_satisfaction(dwelling=self.current_dwelling)

    def get_TYPE(self):
        previous_TYPE = self.TYPE

        new_TYPE = 0
        age = self.get_adults_average_age()
        adults_num = self.get_adults_num()
        household_size = self.get_household_size()
        children_num = household_size - adults_num

        if adults_num == 1: # 1,4,5,7,10,11,13
            if children_num == 0: # 1,5,7,11,13
                if age < 36:    # 1
                    new_TYPE = 1
                elif 35 < age < 65: # 5,7,11
                    if previous_TYPE:
                        if previous_TYPE in set([1,5]):
                            new_TYPE = 5
                        elif previous_TYPE in set([2,6,7]):
                            new_TYPE = 7
                        else: # previous_TYPE in set([3,4,8,9,10,11,12,13]):
                            new_TYPE = 11
                        
                    else:   # immigrants
                        new_TYPE = np.random.choice([5,7,11],1)[0]
                else:  # 13
                    new_TYPE = 13                                
            else: # 4,10,13
                if age < 36:
                    new_TYPE = 4
                elif 35 < age < 65:
                    new_TYPE = 10
                else:
                    new_TYPE = 13
        elif adults_num == 2: # 1,2,3,5,6,8,9,12,13
            if children_num == 0: # 1,2,5,6,9,12,13
                if age < 36:    # 1,2
                    if previous_TYPE:
                        if previous_TYPE == 1: # 1
                            new_TYPE = 1                            
                        else: # 2
                            new_TYPE = 2
                    else:   # immigrants
                        new_TYPE = np.random.choice([1,2],1)[0]
                elif 35 < age < 65: # 5,6,9
                    if previous_TYPE:
                        if previous_TYPE in set([1,5]): # 5
                            new_TYPE = 5
                        elif previous_TYPE in set([2,6,7]): # 6
                            new_TYPE = 6
                        else: # previous_TYPE in set([3,4,8,9,10,11,12,13]): # 9
                            new_TYPE = 9
                    else:   # immigrants
                        new_TYPE = np.random.choice([5,6,9],1)[0]
                else: # 12,13
                    if previous_TYPE:
                        if previous_TYPE in set([6,7,8,9,10,11,12,13]): # 12 
                            new_TYPE = 12 
                        elif previous_TYPE in set([1,5,13]): # 13
                            new_TYPE = 13
                        else:   # special case
                            new_TYPE = 12 
                    else:   # immigrants
                        new_TYPE = np.random.choice([12,13],1)[0]
            else: # 3,8,12
                if age < 36:    # 3
                    new_TYPE = 3
                elif 35 < age < 65: # 8
                    new_TYPE = 8
                else:
                    new_TYPE = 12               
        else:   # 1,5,13
            if children_num == 0:
                if age < 36:    # 1
                    new_TYPE = 1
                elif 35 < age < 65: # 5
                    new_TYPE = 5
                else:  # 13
                    new_TYPE = 13
            else:   # special case
                new_TYPE = 13
       
        self.TYPE = new_TYPE
        return new_TYPE

    def check_TYPE_triggers(self):
        """
        Checks if the TYPE changed from the previous one. If so launch a trigger. This function is called every month
        """
        previous_TYPE = self.TYPE
        new_TYPE = self.get_TYPE()

        if new_TYPE != previous_TYPE:
            res = self.get_TYPE_change_trigger_reason(new_TYPE, previous_TYPE)
            if res:
                reas, category = res 
                self.insert_new_trigger(Trigger(category=category, origin="internal", reason=reas))

    def get_TYPE_change_trigger_reason(self, new_TYPE, previous_TYPE):
        """
        Return the reason of TYPE changing. 
        """
        if (previous_TYPE in set([5,6,7,8,9,10,11]) and new_TYPE in set([12,13])):
            return "growing old", "problem solving"
        
    def get_new_child(self, age=0, init=None, imposed=False):
        """
        Add a child to a household with some probability distribution.
        Get probability "p" of having a child from function that depends on age, and if binomial distribution returns 1 (with probability p) then add a child.
        If imposed == True we need to return the child (this is needed at init when TYPE is predefined)
        """
        if imposed and self.get_minors_num() < parameters.MAX_CHILDREN_NUM:
            new_minor = TenantAgent(self.model.get_next_tenant_id(), self.model, self, age, "minor")
            self.add_member(new_minor)
            self.insert_new_trigger(Trigger(category="radical change", origin="internal", reason="new child"))

        if self.get_TYPE() in set([2,3,6,8]) or (self.get_TYPE() in set([4,10]) and init):  # the second condition of the or accepts that TYPE 4 and 10 have childs but only at init
            p = get_child_probability(self.get_adults_average_age() - age, *self.model.birth_curve_coeff)        # this is done to calculate the probability that the child born "age" years ago in the life of its parent(this is done only in initialization. after kids have age 0)

            if np.random.binomial(1, p, 1)[0] == 1 and self.get_minors_num() < parameters.MAX_CHILDREN_NUM:
                new_minor = TenantAgent(self.model.get_next_tenant_id(), self.model, self, age, "minor")
                self.add_member(new_minor)
                self.insert_new_trigger(Trigger(category="radical change", origin="internal", reason="new child"))

    def get_age_from_range(self, age_range):
        """
        Returns age from range given as parameter. The age is sampled from the distribution of the range (such distributions are in parameters file (name -> AGE_RANGE_1_PROB,..))
        """
        if age_range == 1:
            return np.random.choice(np.arange(0,18), p=parameters.AGE_RANGE_1_PROB, size=1)[0]
        elif age_range == 2:
            return np.random.choice(np.arange(18,36), p=parameters.AGE_RANGE_2_PROB, size=1)[0]
        elif age_range == 3:
            return np.random.choice(np.arange(36,65), p=parameters.AGE_RANGE_3_PROB, size=1)[0]
        else: # age_range == 4
            return np.random.choice(np.arange(65,100), p=parameters.AGE_RANGE_4_PROB, size=1)[0]   


    def create_members(self, tenant=None, init=False, TYPE=None):
        """
        Returns a set of adult and minor tenants, their number and age depends on their TYPE.
        tenant -> case when we want to create an household but we already have a tenant to put into it (children, separated ...)
        """
        if tenant:  # this is the case when a child moves when it becomes indipendent OR when an adult separate (e.g divorce)
            self.add_member(tenant)
            if tenant.age < 36:
                self.TYPE = 1
            elif tenant.age < 65:
                self.TYPE = np.random.choice([5,7,11],1)[0]
            else:
                self.TYPE = 13
        else:
            if TYPE:    # this is the case only at init where we want a precise distribution of TYPE (the TYPE distrib is defined at init by the model)
                adults_num = 0
                adult_age_range = []
                
                # set adults
                if TYPE == 1:
                    adults_num = int(round(get_truncated_normal(mean=1.49, sd=1, low=1, upp=6)))
                    adult_age_range = 2
                elif TYPE in set([2,3]):
                    adults_num = 2
                    adult_age_range = 2
                elif TYPE == 4:
                    adults_num = 1
                    adult_age_range = 2
                elif TYPE == 5:
                    adults_num = int(round(get_truncated_normal(mean=1.17, sd=0.57, low=1, upp=4)))
                    adult_age_range = 3
                elif TYPE in set([6,8,9]):
                    adults_num = 2
                    adult_age_range = 3
                elif TYPE in set([7,10,11]):
                    adults_num = 1
                    adult_age_range = 3
                elif TYPE == 12:
                    adults_num = 2
                    adult_age_range = 4
                elif TYPE == 13:
                    adults_num = int(round(get_truncated_normal(mean=1.12, sd=0.48, low=1, upp=5)))
                    adult_age_range = 4
                else:
                    print("Something wrong, TYPE out of bounds, it should be in range 1-13.")

                for i in range(adults_num):
                    adult_age = self.get_age_from_range(adult_age_range)
                    new_adult = TenantAgent(self.model.get_next_tenant_id(), self.model, self, adult_age, "adult")
                    self.add_member(new_adult)
                
                if TYPE in set([3,4,8,10]):
                    minors_num = int(round(get_truncated_normal(mean=1.67, sd=0.76, low=1, upp=5)))
                else:
                    minors_num = 0
                
                # set minors
                for i in range(minors_num):
                    minor_age = self.get_age_from_range(age_range=1)
                    self.get_new_child(age=minor_age, init=init, imposed=True)

            else: # this is the case when creating immigrant households (i.e HHS and age are sampled from distributions)         
                p = [0.16,0.29,0.18,0.23,0.14]
                members_num = np.random.choice(np.arange(1, len(p)+1), p=p, size=1)[0]
                if members_num == 5:    # because it is 5 or more
                    members_num += np.random.choice(range(0, parameters.MAX_HOUSEHOLD_SIZE),1)[0] # max size of household is 10
                
                members_num = min(members_num, parameters.MAX_HOUSEHOLD_SIZE)

                if members_num == 1:
                    adults_num = 1
                    minors_num = 0
                else:
                    adults_num = np.random.choice(range(1, members_num+1),1)[0] 
                    if adults_num <= 2: # only couples or people alone can have children
                        minors_num = members_num - adults_num
                    else:
                        adults_num = np.random.choice(range(1, 3),1)[0]
                        minors_num = members_num - adults_num

                # reduce children num if too high
                if minors_num >= parameters.MAX_CHILDREN_NUM: 
                    minors_num = parameters.MAX_CHILDREN_NUM - 1

                min_adult_age = 100

                # add adults
                for i in range(adults_num):
                    adult_age = np.random.choice(np.arange(parameters.CHILD_INDIPENDENT_AGE, 100), p=parameters.AGE_PROB_FOR_ADULT_IMMIGRANTS, size=1)[0]
                    if adult_age < min_adult_age:
                        min_adult_age = adult_age

                    new_adult = TenantAgent(self.model.get_next_tenant_id(), self.model, self, adult_age, "adult")
                    self.add_member(new_adult)
                
                # add minors
                for i in range(minors_num):
                    adult_age_at_birth = np.random.choice(np.arange(16, min_adult_age), size=1)[0]
                    minor_age = min_adult_age - adult_age_at_birth

                    if minor_age >= parameters.CHILD_INDIPENDENT_AGE:
                        minor_age = parameters.CHILD_INDIPENDENT_AGE - 1

                    self.get_new_child(age=minor_age, init=init, imposed=True)
                
                #update TYPE
                self.TYPE = self.get_TYPE()

    def get_satisfaction(self, dwelling=None):
        """
        Returns household's current satisfaction. 
        Satisfaction is defined as the size of the difference between the desired functions of the household and the functions of the current dwelling 
        """
        A = self.desired_functions
        B = dwelling.functions
        matched_functions_percentage = len(A & B) / len(A)
        
        los = matched_functions_percentage * parameters.MAX_LOS * parameters.SLOPE_LOS_REGRESSION + parameters.Y_INTERCEPT_LOS_REGRESSION
        los = min(los, parameters.MAX_LOS)
        los = max(los, parameters.MIN_LOS)
        return round(los)
        
    def sample_desired_functions(self):
        """
        Return the set of household's desired functions. Each desired function appears with a probability p that depends on the owner of the current dwelling and the current TYPE/type.
        """
        desired_functions = set()
        TYPE = self.get_TYPE()
        owner = self.current_dwelling.get_owner() if self.current_dwelling else "ALL_OWNERS"
        if owner == "ABZ":
            df = self.model.ABZ_TYPE_DFC_df
        elif owner == "SCHL":
            df = self.model.SCHL_TYPE_DFC_df
        elif owner == "MOBI":
            df = self.model.MOBI_TYPE_DFC_df
        else:   # all owners
            df = self.model.ALL_OWNERS_TYPE_DFC_df

        p_list = df[str(TYPE)].values    # list of probabilities for defined owner and TYPE
        while len(desired_functions) == 0:
            for i, p in enumerate(p_list):
                if np.random.binomial(1, p, 1)[0] == 1:
                    desired_functions.add(i+1)
        return desired_functions

    def get_dwelling(self, postcode=None, upper_room_condition=True):
        """
        Returns a list of available dwelling that satisfies household necessities. 
        The list is sorted on similarity between household desired functions and dwelling functions.
        If parameter "postcode" is defined, return only dwellings that have a postcode different from the one given as parameter.
        Parameter "upper_room_condition" -> if True consider both conditions on rooms number (HHS-1 <= NR <= HHS+2). If False consider only lower condition (HHS-1 <= NR)
        """
        dwellings = {} # dictionary of the available dwellings that satisfy the household necessities (key is dwelling, value is the size of the intersection between that dwelling functions and household ones)
        available_dwellings = [dwell for dwell in self.model.dwellings_list if dwell.is_empty()]
        if available_dwellings:
            sample_empty_dwellings = np.random.choice(available_dwellings, size=parameters.NUM_DWELLING_VISITED if parameters.NUM_DWELLING_VISITED <= len(available_dwellings) else len(available_dwellings), replace=False)
            for dwelling in sample_empty_dwellings:
                if self.check_conditions(dwelling=dwelling, postcode=postcode, upper_room_condition=upper_room_condition):
                    if postcode == None or (postcode != None and postcode != dwelling.building.postcode):  
                        new_satisfaction = self.get_satisfaction(dwelling=dwelling)
                        dwellings[dwelling] = new_satisfaction    # potential satisfaction is used to sort the available dwellings

            dwellings = [k for k, v in sorted(dwellings.items(), key=lambda item: item[1], reverse=True)]   # return a list of sorted (by similarity score) dwellings
        else:
            dwellings = []
        return dwellings

    def check_conditions(self, dwelling, postcode=None, upper_room_condition=True):
        """
        Returns True if the dwelling respect all the conditions imposed by the household (e.g the price is minor than 1/3 of the salary, etc.)
        """
        if  dwelling.is_empty() and \
            self.get_household_salary() >= round(0.33 * dwelling.rent_price) and \
            self.check_tenancy_type(dwelling) and \
            self.check_room_requirements(dwelling=dwelling, upper_room_condition=upper_room_condition) and \
            not dwelling.building.to_demolish and \
            not dwelling.is_renovating and \
            not dwelling.to_renovate and \
            self.check_MOBI_postcode(dwelling, postcode) and \
            self.check_that_dwelling_different(dwelling=dwelling):
            
            new_satisfaction = self.get_satisfaction(dwelling=dwelling)
            if new_satisfaction >= self.satisfaction:
                return True
        return False

    def check_MOBI_postcode(self, dwelling, postcode):
        """
        Checks that, if the current owner is MOBI, the new dwelling has the same postcode.
        """
        if postcode == None:
            if not self.current_dwelling:
                return True
            if self.current_dwelling.get_owner() != "MOBI":
                return True
            if self.current_dwelling.get_postcode() == dwelling.get_postcode(): # MOBI postcode
                return True
            return False
        return True

    def check_that_dwelling_different(self, dwelling):
        """
        Checks that the dwelling in which we would like to go is not the same that we are leaving.
        """
        if self.current_dwelling:
            if dwelling.id != self.current_dwelling.id:
                return True
            else:
                return False
        else:
            return True  # any dwelling is good, no current dwelling

    def check_room_requirements(self, dwelling, upper_room_condition=True):
        if dwelling.rooms < self.get_household_size()-1:
            return False
        
        if not upper_room_condition:
            return True
        
        if dwelling.get_owner() == "MOBI":
            if dwelling.rooms > self.get_household_size()+2:
                return False
        else:
            if dwelling.rooms > self.get_household_size()+1:
                return False

        return True

    def check_tenancy_type(self, dwelling):
        """
        Checks if tenancy type of household is the same of the one of the dwelling (e.g they are both "cooperatives")
        """
        if not self.current_dwelling:
            return True
        if self.current_dwelling.get_owner() == "MOBI" or \
            self.current_dwelling.get_owner() == dwelling.get_owner():
            return True
        return False 

    def remove_household(self):
        """
        Removes the household from the model. If the reason is emigration, increment emigration counter
        """
        if self.current_dwelling:                   # means that the tenants is moving from another dwelling and this is not initialization phase
            self.current_dwelling.household = None
        try:
            self.model.households_list.remove(self)
        except:
            pass  # do nothing!
        try:
            self.model.schedule.remove(self)
        except:
            pass  # do nothing!
            
    def insert_new_trigger(self, new_trigger):
        """
        Insert operation for trigger data structure. For now it doesn't overwrite the previous trigger
        """
        if self.trigger == None:
            self.trigger = new_trigger

    def add_member(self, tenant):
        if tenant not in self.members:
            self.members.append(tenant)       