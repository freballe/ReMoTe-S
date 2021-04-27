from mesa import Agent
import numpy as np
from trigger import Trigger
import parameters

class TenantAgent(Agent):
    """
    Tenant initialization
    """
    def __init__(self, unique_id, model, household, age, member_type):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.household = household
        self.age = age
        self.death_month = self.get_death_month()                   
        self.member_type = member_type
        if self.member_type == "adult":
            self.salary = self.get_salary()
        else:   # minor
            self.salary = 0

        self.to_fire = False # when true, the tenant will be fired as soon as its household 
        self.is_unemployed = False
        self.unemployed_struct = {}    # used to store unemployment data
        self.divorced = False   # True if the tenant divorced
        self.widowed = False    # True if tenant lost its partner


    def step(self):
        """
        Increment age of the tenant and removes it from the household if it reached its death age.
        """
        if (self.model.schedule.steps % 12) == 0 and self.model.schedule.steps > 0:
            self.age += 1

        if self.is_unemployed:
            self.decrease_salary_struct["counter"] -= 1
            if self.decrease_salary_struct["counter"] == 0:
                self.is_unemployed = False
                self.salary = self.decrease_salary_struct["old salary"]

        if (self.age*12 + self.model.step_num % 12) >= self.death_month:    # if current age month greater equal to death month , remove tenant           
            if self.household.get_adults_num() > 0 and self.household.get_TYPE() in set([2,3,6,8,9,12]) and self.member_type == "adult": # the one dying is part of a couple
                for member in self.household.members:
                    if member != self:  # the other adult is becoming a widow, not the one dying
                        member.widowed = True
            
            self.remove_tenant_from_household()
        else:
            if self.member_type == "minor" and self.age >= parameters.CHILD_INDIPENDENT_AGE:
                self.become_indipendent()

    def decrease_salary(self):
        """
        Populate the dictionary that stores info about the old salary and when to restore it 
        (the old salary is restored in a range of [1,24] months after the tenant lose is job)
        """
        self.is_unemployed = True
        self.to_fire = False
        self.decrease_salary_struct = {}
        self.decrease_salary_struct["old salary"] = self.salary
        self.decrease_salary_struct["counter"] = np.random.choice(range(1,25),1)[0]   # this counter is decreased every month. When it goes to zero the salary is restored back to its previous value
        if self.age < 25 or \
           self.salary < 45564 or \
           (self.member_type == "adult" and self.household.get_adults_num() == 1 and self.household.get_minors_num() > 0):  # single adult with children
            range_num = int(self.salary * 0.2)
            if range_num > 0:
                self.salary -= np.random.choice(range(range_num),1)[0] # decrease by 20%
        else:
            range_num = int(self.salary * 0.3)
            if range_num > 0:
                self.salary -= np.random.choice(range(range_num),1)[0] # decrease by 30%

    def get_salary(self):
        """
        Return salary of a tenant. The salary is sampled from probability distributions, 
        each one depending on the age range of the tenant
        """
        household_age = self.age
        if household_age in range(parameters.CHILD_INDIPENDENT_AGE):
            mean = parameters.MEAN_SALARY_YOUNG
            std = parameters.STDDEV_SALARY_YOUNG
        elif household_age in range(parameters.CHILD_INDIPENDENT_AGE,65):
            mean = parameters.MEAN_SALARY_MIDDLE
            std = parameters.STDDEV_SALARY_MIDDLE
        else: # +65
            mean = parameters.MEAN_SALARY_OLD
            std = parameters.STDDEV_SALARY_OLD

        return np.abs(np.random.normal(loc=mean,scale=std,size=1)[0])

    def become_indipendent(self, leaving_flatshare=False):
        """
        Make the tenant indipendendent by creating another household on its own. 
        """
        if self.member_type == "minor":
            self.member_type = "adult"
            if self.salary == 0:    # this is true when they are minors
                self.salary = self.get_salary()
            if self.household.get_minors_num() == 0 and np.random.binomial(1, parameters.HOUSEHOLD_MOVES_AFTER_CHILD_LEAVES_PROBABILITY, 1)[0] == 1: 
                self.household.insert_new_trigger(Trigger(category="radical change", origin="internal", reason="children leaving"))
        self.model.assign_household_to_tenant(self, leaving_flatshare)

    def get_death_month(self):
        """
        Returns the age at which the tenant will die.
        """
        a = 7                           # power law parameter  
        samples = 1
        p = np.random.power(a, samples) # probability
        death_age = int(p[0] * 100 + 1)
        while death_age <= self.age:        #  avoid that the tenant dies at first step
            p = np.random.power(a, samples) # probability
            death_age = int(p[0] * 100 + 1)
        return (death_age * 12) - np.random.randint(12) # get a random month to die

    def remove_tenant_from_household(self):
        """
        Removes the tenant from the household. 
        """
        if self.household:
            if self in self.household.members:
                self.household.members.remove(self)
            if len(self.household.members) <= 0 or self.household.get_adults_num() == 0:  # if no more members in this household, remove it from the model
                self.household.remove_household()     
