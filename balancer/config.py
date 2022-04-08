"""Modify the variables below to define the execution of the balancer.

Read the description (below each variable as comments) for each variable carefuly.
"""


## INPUT DEFINITION
###############################################################################
PATH_TO_CSV = 'survey.csv'
# Type string
# Path to the csv file containing the information gathered for each participant

COLUMN_MAPPER = {
    'Name': 'name',
    'If your team is less than 4 people would you like to have other students join your team to get to a total of 4?': 'team_open',
    'If you already have team members, please list the full name of all (max 4) team members below. (Write N/A if you are not part of a team)': 'teams',
    'What year at the university are you?': 'year',
    'How comfortable are you with python code?': 'python'
}
# Type dict
# map current column names to new names
# !!!!!!NOTE!!!!!!!! hereafter refer to the new column names in your variables

PARTICIPANT_COLUMN = 'name'
# Type string
# Column id in the input csv file uniquely identifying the partiticpant

TEAM_CHOICE_COLUMN = 'teams'
# Type string
# Column id in the input csv file with desired teamates
# contents should be iterable eg. string containing commas

TEAM_OPEN_COLUMN = 'team_open'
# Type string
# Column id in the input csv indicating if their team is open for new participants
# Contents should be boolean, eg. "yes", "False", "1" etc.

BALANCE_VARIABLES = [
#     ('gender', 'at_least_or_none', ('female', 2)),
    ('year', 'weighted', 1.0),
    ('python', 'callable', 'square')
]
# Type list of tuple, each of len 3
# Each tuple contains in order:
#     - Column id in the input csv of the variable
#     - The type of variable, options below
#     - A parameter to use associate with that type of variable
# Variable types:
#     - 'weighted', simply weigh the value by a float. Parameter is float.
#     - 'callable', apply a callable that returns a float to the value. Parameter is callable or string name of callable in `numpy`
#     - 'at_least_or_none', indicates that the the team should contains at least X or none of Y in the parameter (Y, X)

CUSTOM_VALUE_MAPPING = {
    'year': {'Senior': 4, 'Junior': 3, 'Sophomore': 2, 'Freshman':1}
}
# Dict of dict
# for a column name, provide a dictionary mapping current contents to desired contents

## ALGORITHM DEFINITION
###############################################################################
LOSS = "square"
# Type callable or name of callable in numpy that returns a float
# The function that is applied to the stdev of team scores to be minimized

TEAM_SCORING = 'mean'
# str, "mean" or "sum"
# how to treat participant scores to determine the team score
# mean is agnostic to team size, so a team of any size can score any value
# sum will produce lower scores for teams with fewer members

MAXIMUM_PARTICIPANTS = 4
# Type int
# the maximum number of participants per team

