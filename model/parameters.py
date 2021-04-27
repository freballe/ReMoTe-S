import numpy as np

HOUSEHOLD_MOVES_AFTER_CHILD_LEAVES_PROBABILITY = 0.127 # 12.7%, prob that parents change dwelling after children leave
MAX_HOUSEHOLD_SIZE = 10
NUM_DWELLING_VISITED = 25                      # number of dwellings that are visited (in one month) when looking for a new accomodation

HOUSEHOLD1_ID_TO_FOLLOW = 13
HOUSEHOLD2_ID_TO_FOLLOW = 20

NUM_HOUSEHOLDS = 1000               # households number
NUM_BUILDINGS = 30

MAX_MONTH_TO_WAIT_SINCE_MOVER = 12    # max number of months to wait after mover before leaving the model
RENOVATION_TIME = 3                 # duration of renovation
RENOVATION_NOTIFICATION_TIME = 12   # time household have from renovation notification to leave

# vacancy rates
VACANCY_RATE_POSTCODE_1000 = 0.004       #  0.4%
VACANCY_RATE_POSTCODE_8000 = 0.001       #  0.1%
VACANCY_RATE_POSTCODE_OTHERS = 0.027     #  2.7%

# immigration rate
IMMIGRATION_RATE = 0.0275   # per month

# household
MAX_CHILDREN_NUM = 8
YEARLY_SALARY_INCREASE = 0.009
MEAN_SALARY_YOUNG = 43462
MEAN_SALARY_MIDDLE = 53141
MEAN_SALARY_OLD = 44899
STDDEV_SALARY_YOUNG = 23154
STDDEV_SALARY_MIDDLE = 27946
STDDEV_SALARY_OLD = 25155

# tenant
CHILD_INDIPENDENT_AGE = 19          # age at which a child become indipendent
# probability to have a certain age when joining the model as an immigrant
AGE_PROB_FOR_ADULT_IMMIGRANTS = np.array([1.33,1.30,1.33,1.38,1.42,1.44,1.49,1.55,1.64,1.69,1.73,1.73,1.77,1.74,1.76,1.76,1.77,1.75,1.78,1.77,1.77,1.71,1.69,1.69,1.67,1.66,1.69,1.70,1.74,1.79,1.81,1.86,1.89,1.90,1.93,1.92,1.95,1.88,1.80,1.73,1.67,1.60,1.53,1.48,1.43,1.36,1.30,1.25,1.23,1.18,1.21,1.18,1.18,1.16,1.14,1.08,1.05,1.00,0.94,0.86,0.77,0.73,0.68,0.63,0.60,0.57,0.52,0.46,0.42,0.37,0.33,0.27,0.22,0.18,0.14,0.11,0.09,0.06,0.04,0.03,0.04])
AGE_PROB_FOR_MINOR_IMMIGRANTS = np.array([5.21,5.40,5.40,5.48,5.44,5.42,5.31,5.35,5.30,5.38,5.29,5.27,5.16,5.11,5.10,5.09,5.02,5.10,5.18])
AGE_PROB_FOR_ADULT_IMMIGRANTS /= AGE_PROB_FOR_ADULT_IMMIGRANTS.sum()  # normalize (numpy random choice issue need this)
AGE_PROB_FOR_MINOR_IMMIGRANTS /= AGE_PROB_FOR_MINOR_IMMIGRANTS.sum()  # normalize (numpy random choice issue need this)
# probabilities to have a certain age given the range
AGE_RANGE_1_PROB = [0.0548,0.0569,0.0569,0.0578,0.0573,0.0572,0.0560,0.0565,0.0559,0.0567,0.0558,0.0556,0.0545,0.0539,0.0537,0.0537,0.0529,0.0539] # range 1 -> [0,17] (all range extremes are included)
AGE_RANGE_2_PROB = [0.0432,0.0463,0.0464,0.0474,0.0491,0.0507,0.0515,0.0531,0.0554,0.0587,0.0603,0.0618,0.0618,0.0633,0.0621,0.0629,0.0629,0.0631] # range 2 -> [18,35]
AGE_RANGE_3_PROB = [0.0350,0.0355,0.0352,0.0352,0.0342,0.0338,0.0336,0.0333,0.0331,0.0337,0.0339,0.0348,0.0358,0.0361,0.0370,0.0376,0.0379,0.0384,0.0384,0.0388,0.0374,0.0359,0.0344,0.0333,0.0320,0.0305,0.0296,0.0285,0.0271] # range 3 -> [36,64]
AGE_RANGE_4_PROB = [0.0650,0.0610,0.0601,0.0565,0.0595,0.0553,0.0485,0.0474,0.0468,0.0443,0.0430,0.0409,0.0384,0.0352,0.0315,0.0300,0.0281,0.0260,0.0247,0.0232,0.0212,0.0190,0.0172,0.0151,0.0134,0.0111,0.0092,0.0073,0.0059,0.0046,0.0035,0.0025,0.0017,0.0012,0.0017] # range 4 -> [65,99]

# buildings
BUILDINGS_CONSTRUCTION_RATE = 0.0069 # annual building construction rate 0.69%
BUILDINGS_DEMOLITION_RATE = 0.01

#dwellings
MEAN_ROOMS = 3.5
DWELLINGS_RENOVATION_RATE = 0.009   # annual dwellings renovation rate
# follows -> mean and stddev of dwelling size (DS). The distribution depends on the number of rooms in the dwelling
MEAN_DS_1_ROOM = 34.0
MEAN_DS_2_ROOM = 52.8
MEAN_DS_3_ROOM = 73.1
MEAN_DS_4_ROOM = 96.0
MEAN_DS_5_ROOM = 123.2
MEAN_DS_6_ROOM = 143.7
MEAN_DS_7_ROOM = 156.4
STDDEV_DS_1_ROOM = 16.9
STDDEV_DS_2_ROOM = 16.8
STDDEV_DS_3_ROOM = 19.3
STDDEV_DS_4_ROOM = 24.7
STDDEV_DS_5_ROOM = 31.3
STDDEV_DS_6_ROOM = 32.0
STDDEV_DS_7_ROOM = 29.5

# rent
MAX_RENT_INCREASE = 3405           # max rent increase after a dwelling renovation. Annual rent increase, for EACH ROOM. Example if the renovated dwelling has 4 rooms and we sampled 3000 from the range then the total annual salary increase is 3000*4=12000, thus 1000 more per month
MIN_RENT_INCREASE = 2536           # min rent increase after a dwelling renovation

# triggers probabilities
DIVORCE_PROBABILITY = 0.02                      # per year
LEAVE_FLATSHARE_PROBABILITY = 0.186             # per year
MOVE_AFTER_SALARY_INCREASE = 0.015              # per year

MOVE_FOR_FAMILY_PROBABILITY = 0.011 / 12        # per month   (1.1% per year, assume indipendent events)
CHANGE_JOB_LOCATION_PROBABILITY = 0.064 / 12    # per month   (6.4% per year, assume indipendent events)
NEED_FOR_CHANGE_PROBABILITY = 0.031 / 12        # per month   (3.1% per year, assume indipendent events)
INTERPERSONAL_PROBLEMS_PROBABILITY = 0.016 / 12 # per month   (1.6% per year, assume indipendent events)
EXPIRE_RENTAL_CONTRACT_PROBABILITY = 0.023 / 12 # per month   (2.3% per year, assume indipendent events)

JOB_LOSS_PROBABILITY = 0.032                    # per year. Percentage of tenants that lose their job each year (3.2%)
ELDERLIES_LEAVE_MODEL_PROBABILITY = 0.0

# attempts
NUM_ATTEMPTS_TO_FIND_PARTNER = 50               # number of households that a  divorced/leaving child/flatshare leaving  visits to find a joinable household (e.g a partner to create a couple). If after all the attempts it is not able to find anything it create an household alone
MAX_ITERS_FOR_ROOMS_CONDITIONS = 10             # (at init ONLY) number of times a household look for dwellings with both the conditions on rooms (HHS-1 <= NR <= HHS+2)

# satisfaction
SATISFACTION_MONTHLY_DECREASE = 0.005
SLOPE_LOS_REGRESSION = 4/5    # slope of the line used for los regression
Y_INTERCEPT_LOS_REGRESSION = 1  # y intercept of the line used for los regression (note that the min los is 1)
MIN_LOS = 1
MAX_LOS = 5
INIT_LOS = 3    # los when each hh is initialized

# reas collection
STARTING_MONTH_REASON_COLLECTION = 24 # starting month at which to start collecting reasons data


DWELLING_FUNCTIONS_PROBABILITIES = {}  # dict with the probabilities that for each dwelling function we have the characteristics e.g "bright" is 24% in when we have DF1 in the dwelling
DWELLING_FUNCTIONS_PROBABILITIES["bright"] =               [0.24,0.24,0.27,0.21,0.21,0.24,0.26,0.24,0.25]
DWELLING_FUNCTIONS_PROBABILITIES["with balcony"] =         [0.38,0.39,0.34,0.33,0.36,0.38,0.38,0.39,0.39]
DWELLING_FUNCTIONS_PROBABILITIES["with green spaces"] =    [0.26,0.29,0.28,0.29,0.30,0.26,0.30,0.29,0.30]
DWELLING_FUNCTIONS_PROBABILITIES["with parking place"] =   [0.16,0.19,0.17,0.20,0.20,0.19,0.17,0.17,0.13]
DWELLING_FUNCTIONS_PROBABILITIES["work"] =                 [0.33,0.33,0.27,0.27,0.32,0.31,0.29,0.32,0.28]
DWELLING_FUNCTIONS_PROBABILITIES["public transports"] =    [0.49,0.53,0.52,0.37,0.54,0.55,0.52,0.52,0.53]
DWELLING_FUNCTIONS_PROBABILITIES["city center"] =          [0.31,0.33,0.35,0.38,0.33,0.33,0.35,0.33,0.35]
DWELLING_FUNCTIONS_PROBABILITIES["safe"] =                 [0.39,0.31,0.31,0.33,0.31,0.31,0.31,0.32,0.32]
DWELLING_FUNCTIONS_PROBABILITIES["sociocultural mix"] =    [0.24,0.23,0.21,0.16,0.22,0.25,0.22,0.25,0.23]
DWELLING_FUNCTIONS_PROBABILITIES["accessible by car"] =    [0.17,0.24,0.24,0.20,0.24,0.26,0.22,0.23,0.18]