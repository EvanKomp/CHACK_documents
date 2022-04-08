"""Execute the balancing protocol."""
import sys
import logging
from functools import partial
from typing import Union, List

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 1000)

import config

logging.basicConfig(filename='balance.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser(description='Process flags.')
#####################################################################
# Function to load the raw csv and get it ready for balancing
#####################################################################
def load_clean_csv():
    """Load input csv file and clean and prepare it as pandas.

    Returns
    -------
    DataFrame : data with only relevant, possibly renamed columns
    """
    df = pd.read_csv(config.PATH_TO_CSV)
    
    logger.info(f'Raw survey loaded with columns {df.columns}')
    
    df = df.rename(columns=config.COLUMN_MAPPER)

    df = df[[
        config.PARTICIPANT_COLUMN,
        config.TEAM_CHOICE_COLUMN,
        config.TEAM_OPEN_COLUMN,
        *(var[0] for var in config.BALANCE_VARIABLES)
    ]]
    logger.info(f'Survey columns renamed to {df.columns}')
    
    # now we have to parse specific values
    # start with names
    df[config.PARTICIPANT_COLUMN] = df[config.PARTICIPANT_COLUMN].apply(
        lambda x: str(x).lower())
    
    # drop duplicate responses
    initial_names = df[config.PARTICIPANT_COLUMN].values
    initial_index = np.array(df.index)
    df.drop_duplicates(config.PARTICIPANT_COLUMN, inplace=True, keep='last')
    dropped_indexes = ~np.isin(initial_index, np.array(df.index))
    df.reset_index(drop=True, inplace=True)
    logger.info(f'Dropped duplicate names {initial_names[dropped_indexes]}')
    
    # now do desired teamates
    #### DEFINE A FUNCTION TO PARSE TEAMATE INPUTS
    def get_teamates(x):
        x = str(x)
        if x == 'nan' or x == 'None':
            return []
        else:
            teamate_list = x.split(',')
            new_list = []
            for teamate in teamate_list:
                teamate = teamate.strip()
                teamate = teamate.lower()
                if teamate not in df[config.PARTICIPANT_COLUMN].values:
                    teamate_nums = input(
                        f'INPUT REQUIRED: "{teamate}" not a recognized participant. Please indicate\
 the participant(s) being specified as the number(s) seperated by spaces from this list. Write "none" to skip:\n\
{df[config.PARTICIPANT_COLUMN]}'
                    )
                    if teamate_nums == 'none':
                        continue
                    teamate_nums = teamate_nums.split(' ')
                    teamate_nums = [int(num) for num in teamate_nums]
                    
                    teamate = df[config.PARTICIPANT_COLUMN][teamate_nums].values
                else:
                    teamate = [teamate]
                new_list.extend(teamate)
            return new_list
    ########################################
    # clean up the teams, this will probably require user input
    df[config.TEAM_CHOICE_COLUMN] = df[config.TEAM_CHOICE_COLUMN].apply(lambda x:
        get_teamates(x)
    )
    # add self if necessary and sort
    for i, team in enumerate(df[config.TEAM_CHOICE_COLUMN]):
        participant = df.loc[i, config.PARTICIPANT_COLUMN]
        if participant not in team and len(team) > 0:
            team.append(participant)
        team.sort()
    logger.info(f'Cleaned and sorted preferred teams.')
    
    # now team preferences
    new_preference = []
    for i, preference in enumerate(df[config.TEAM_OPEN_COLUMN]):
        if not len(df.loc[i, config.TEAM_CHOICE_COLUMN]) > 0:
            new_preference.append(None)
        else:
            if 'no' in preference.lower():
                new_preference.append(False)
            elif 'yes' in preference.lower():
                new_preference.append(True)
            else:
                user_says = input(f'Could not interperet {preference}. Indicate "True" or "False" for whether the team is open')
                new_preference.append(bool(user_says))
    df[config.TEAM_OPEN_COLUMN] = new_preference
    logger.info("Parsed individual statements of teams are open or not")
    
    # now change variables that the user supplied
    for key, mapper in config.CUSTOM_VALUE_MAPPING.items():
        def get_from_mapper(x):
            if x in mapper:
                return mapper[x]
            else:
                return input(f'"{x}" not in the specified mapper {mapper}. Input what should be recorded.')
            
        if key not in df.columns:
            raise TypeError(
                f'Specified column name {key} not in the data columns {df.columns}'
            )
        else:
            df[key] = df[key].apply(lambda x: get_from_mapper(x))
            logger.info(f'Converted values in column {key} according to {mapper}')
    return df

#####################################################################
# Class to represent on participant
#####################################################################

class Participant:
    """Indicates one participant.
    
    Parameters
    ----------
    identifier : any
        Unique identifier for the participant.
        Will be converted to string and lowercased
    variables : dict
        Dictionary mapping name of variables used for scoring to this participants value.
    team : Team, default None
        team to assign this participant to.
        
    Attributes
    ----------
    pid : str, the name of the participant
    team : Team, the teams this part is assigned to
    tid : ind, the team number this part is assigned to
    score : float, the overall score for this participant
    scores : dict, maps the score contribution for each variable
    report : dict, summary of this participants position
    """
    def __init__(self, identifier, variables: dict, team = None):
        # standardize the identifier input
        self._team = None
        if team is not None:
            self.team = team
        
        self.pid = str(identifier).lower()
        self.variables = variables
        self.scores = None
        return
    
    @property
    def tid(self):
        """The participant's team's id"""
        if self.team is None:
            return None
        else:
            return self.team.tid
        
    @property
    def team(self):
        return self._team
    
    @team.setter
    def team(self, new_team):
        if new_team is None:
            self._team = None
        else:
            assert isinstance(new_team, Team)
            if self._team is not None and self._team.tid != new_team.tid:
                replace_team = input(
                    f'Participant ({self.pid}) is being reassigned from team {self._team.tid} to {new_team.tid}. Continue? (y/n)'
                ) 
                if replace_team == 'y':
                    self._team = new_team
                else:
                    pass
            else:
                self._team = new_team
        return
    
    @property
    def report(self):
        participant_report = {'participant': self.pid}
        for scoring_var, scoring_value in self.scores.items():
            participant_report['scoring_'+scoring_var] = scoring_value
        participant_report['participant_score'] = self.score
        participant_report['team_placed'] = self.tid
        return participant_report

#####################################################################
# Class to represent one team
#####################################################################
    
class Team:
    """Indicates one team.
    
    Parameters
    ----------
    identifier : any
        Unique identifier for the team.
        Will be converted to int
    participants : iterable of Participants, default None
        The participants to initialize this team with.
        Default is empty team
    open_team : bool, default True
        Indicates if the team is open to new participants
        
    Attributes
    ----------
    tid : int, the team number
    participants : list of Participant, parts in the team
    space_left : int, number of open spots in the team
    pids : list of str, names of participants in team
    """
    def __init__(self, identifier, participants: list = None, open_team: bool = True):
        self.tid = int(identifier)
        if participants is None:
            self._participants = []
        else:
            assert all([
                isinstance(participant, Participant) for participant in participants
            ]), "`participants` should be a list of Participants"
            self.participants = participants
        self.open_team = open_team
        return
    
    @property
    def space_left(self):
        """Indicates how many spots are left in this team."""
        if not self.open_team:
            return 0
        else:
            return int(config.MAXIMUM_PARTICIPANTS - self.num_participants)
        
    @property
    def num_participants(self):
        """The number of participants in this team"""
        return len(self.participants)
    
    def add(self, participants: Union[Participant, List[Participant]]):
        """Add one or more participants to the team.
        
        Parameters
        ----------
        participants : list of Participants
        """
        if type(participants) != list:
            participants = [participants]
        
        if len(participants) > self.space_left:
            raise ValueError(
                f'Cannot add {len(new_participants)} participants to team with {self.space_left} spots remaining'
            )
        else:
            self.participants = self.participants + participants
        return
    
    def remove(self, participants: Union[str, List[str]]):
        """Remove one or more participants by name.
        
        Parameters
        ----------
        participants : list of str
            Names of participants to remove.
        """
        if type(participants) != list:
            participants = [participants]
        # check we are note trying to remove someone that is not there
        for part_to_remove in participants:
            if part_to_remove not in self.pids:
                raise ValueError(f'cannot remove {part_to_remove} from team of {self.pids}, not in team')
        
        # now actually remove
        new_participants = []
        for part in self.participants:
            if part.pid not in participants:
                new_participants.append(part)
            else:
                part.team = None
        self.participants = new_participants
        return

    @property
    def participants(self):
        return self._participants
    
    @participants.setter
    def participants(self, new_participants):
        assert all([isinstance(part, Participant) for part in new_participants]),\
            "Cannot set participants to non Participant objects"
        if len(new_participants) > int(config.MAXIMUM_PARTICIPANTS):
            raise ValueError(
                f'Cannot have team of {len(new_participants)} participants.'
            )
        participants_to_save = []
        for part in new_participants:
            part.team = self
            if part.team is self:
                participants_to_save.append(part)
            else:
                logger.info(
                    f'Participant {part.pid} could not be placed in this team {self.tid}, already assigned to team {part.tid}'
                )
                pass
        self._participants = participants_to_save
        return
    
    @property
    def pids(self):
        return [part.pid for part in self.participants]

#####################################################################
# Class to track which participants and teams are where
#####################################################################

class Registrar:
    """
    Initializes and stores participants and teams.
    
    Parameters
    ----------
    survey_df : DataFrame
        Dataframe formated by load_clean_csv according to the config.
        Contains input information on teams, participants, and participant variables
        
    Attributes
    ----------
    participants : dict of {part name, part object}
    teams : dict of {team number, team object}
    teamless_participants : list of Participants, parts without a team
    open_teams : list of Teams, teams with at least one open space
    tids : list of int, team ids
    """
    def __init__(self, survey_df):
        self.df = survey_df
        
        # start by initializing participants
        self.balance_variables = [balance_var[0] for balance_var in config.BALANCE_VARIABLES]
        participants = []
        for i, row in self.df.iterrows():
            pid = row[config.PARTICIPANT_COLUMN]
            participant = Participant(pid, variables = dict(row[self.balance_variables]))
            preferred_team = row[config.TEAM_CHOICE_COLUMN]
            team_open = row[config.TEAM_OPEN_COLUMN]
            participants.append({
                'pid': pid,
                'participant': participant,
                'team_choice': preferred_team,
                'team_open': team_open,
            })
        self.participants = pd.DataFrame(participants)
        logger.info(f'Initialized all participants in {self.participants["pid"].values}')
        
        # use this information to initialize teams
        teams = []
        tid = 0
        for i, row in self.participants.iterrows():
            pid, part, team_choice, team_open = row
            if part.team is not None:
                logger.info(f'{pid} already placed on team {part.tid}')
                continue
            elif len(team_choice) == 0:
                logger.info(f'{pid} has no team preference, moving on.')
                continue
            else:
                if len(team_choice) == 0 and team_choice[0] == pid:
                    logger.info(f'{pid} has no team preference, moving on.')
                    continue
                else:
                    pass
                
                # this is the big stuff, we need to try to create and assign a team for
                # this person and claimed teamates
                # first check that the listed teamates match those listed by the other people
                claimed_teamates_series = self.participants['team_choice'][self.participants['pid'].isin(team_choice)]
                if not all(
                    [claimed_teamates == team_choice for claimed_teamates in claimed_teamates_series.values]
                ) or len(team_choice) > config.MAXIMUM_PARTICIPANTS:
                    teamates = input(
                        f'TEAMATE MISMATCH OR TOO BIG when trying to place participant ({pid}). They claimed teamates : \n {team_choice}\n\
Those participants claimed:\n {claimed_teamates_series}\n. Please indicate the numbers seperated by spaces of the participants\
 from the below list who should be teamed up with ({pid}). Report "none" to leave this participant teamless. {self.participants["pid"]}'
                    )
                    if teamates == 'none':
                        continue
                    else:
                        teamates = self.participants['pid'][[int(teamate_num) for teamate_num in teamates.split()]]
                else:
                    teamates = self.participants['pid'].loc[claimed_teamates_series.index]
                    
                
                # now check team open preference
                claimed_open_preference = self.participants.loc[teamates.index]['team_open']
                if all([open_pref == team_open for open_pref in claimed_open_preference]):
                    this_team_open = team_open
                    if this_team_open is None:
                        this_team_open = False
                else:
                    this_team_open = input(f"Teamates {teamates.values} did not all claim the same team-open preference:\n{claimed_open_preference}.\
 Input 'True' or 'False' for whether the team should be open to new teamates.")
                    if this_team_open == "True":
                        this_team_open = True
                    else:
                        this_team_open = False
                
                # create a team!
                team = Team(tid, participants = self.participants['participant'].loc[teamates.index], open_team = this_team_open)
                logger.info(f"Created team {tid} with participants {teamates.values}, open preference = {this_team_open}")
                tid +=1
                teams.append(team)
        
        # teams made, clean up participants and save
        self.participants = dict(zip(
            self.participants['pid'].values,
            self.participants['participant'].values
        ))
        self.teams = {}
        for team in teams:
            self.teams[team.tid] = team
        
        self.team_scores = None
        self.overall_score = None
        return
    
    @property
    def teamless_participants(self):
        """Return only the participants that do not have teams."""
        teamless = [part for part in self.participants.values() if part.team is None]
        return teamless
    
    @property
    def open_teams(self):
        """Return only the teams that are open."""
        teams_with_space = [team for team in self.teams.values() if team.space_left > 0]
        return teams_with_space
        
    @property
    def tids(self):
        return list(self.teams.keys())
    
    
    def add_team(self, team: Team):
        """Add a new team to the registrar.
        
        Team with id -1 will be assigned a new id.
        
        Parameters
        ----------
        team : Team, team to add.
        """
        if team.tid == -1:
            team.tid = min(set(range(max(self.tids)+2)) - set(self.tids))
        elif team.tid in self.tids:
            raise ValueError(f'Cannot add team to registrar. tid {team.tid} already taken')
        self.teams[team.tid] = team
        return
    
    def generate_report(self):
        """Report the balanced teams.
        
        Returns reports in the form of dataframes, and saves to csv
        
        Returns
        -------
        DataFrame : report describing each participant
        DataFrame : report describing each team
        """
        if self.overall_score is None:
            raise ValueError('Cannot generate report, teams have not been balanced')
        # first let's create a participant - team pairing report
        participant_reports = [participant.report for participant in self.participants.values()]
        participants_report = pd.DataFrame(participant_reports)
        
        # now a report on teams
        teams_report = pd.DataFrame({
            'team': self.tids,
            'participants': [team.pids for team in self.teams.values()],
            'team_score': self.team_scores
        })
        
        # write to file
        participants_report.to_csv('participants_report.csv')
        team_report_file = open('teams_report.csv', 'w')
        team_report_file.write('Overall loss of team placement: '+ str(self.overall_score) + '\n')
        teams_report.to_csv(team_report_file)
        team_report_file.close()
        
        return participants_report, teams_report
        

#####################################################################
# Functions to define protocols when conditions are not met.
# should take the balancer that will execute the protocol as self, and
# the offending participant and team that broke the condition  as first arguments.
# kwargs are assigned by scorer based on config parameters
#####################################################################
        
def _at_least_or_none_protocol(balancer, participant, team, var_name, comparison_value):
    if team.space_left > 0:
        # add another with that condition
        potential_participants = []
        for part in balancer.registrar.teamless_participants:
            if part.variables[var_name] == comparison_value:
                potential_participants.append(part)

        scores_when_added = {}
        for part in potential_participants:
            scores_when_added[part] = balancer._attempt_add(part, team)

        part_to_add = min(scores_when_added, key=scores_when_added.get)
        team.add(part_to_add)
        logger.info(f'Placed participant {part_to_add.pid} in team {team.tid}\
 to satisfy condition {var_name} failure that occurred when {participant.pid} was added to {team.tid}')

    else:
        # we have to move this participant elsewhere
        # first check there is seomwehere to put them
        team.remove(participant.pid)
        potential_teams = []
        for potential_team in balancer.registrar.open_teams:
            if potential_team is team:
                continue
            participants_are = [p.variables[var_name] == comparison_value for p in team.participants]
            if np.sum(participants_are) > 0:
                potential_teams.append(potential_team)
        if len(potential_teams) == 0:
            logger.info(f'Participant {participant.pid} on team {team.tid} fails\
condition {var_name}, but no other option remains')
            team.add(participant)
        else:
            balancer._place_participant(participant, potential_teams)
    return

#####################################################################
# Class to evaluate teams and individuals
#####################################################################

class Scorer:
    """Score an individual or a team.
    
    Creates scoring functions based on the config.
    """
    
    def __init__(self):
        if config.TEAM_SCORING == 'sum':
            self.team_scoring_function = np.sum
        elif config.TEAM_SCORING == 'mean':
            self.team_scoring_function = np.mean
        else:
            raise TypeError(
                f'unrecognized team scoring method {config.TEAM_SCORING}, check config'
            )
        
        ## now create the functions used to score individuals for each variable
        participant_scoring_functions = {}
        team_conditionals = {}
        team_conditionals_protocols={}
        for var_name, var_type, parameters in config.BALANCE_VARIABLES:
            if var_type == 'weighted':
                assert type(parameters) == float, "parameter for 'weighted' variable type must be float, check config"
                def func(var_value, parameters):
                    return var_value * parameters
                func = partial(func, parameters= parameters)
            elif var_type == 'callable':
                if callable(parameters):
                    func = parameters
                else:
                    try:
                        func = getattr(np, parameters)
                    except:
                        raise TypeError(
                            f'could not interperent {parameters} as a callable, check config'
                        )
            elif var_type == 'at_least_or_none':
                assert type(parameters) == tuple,\
                    "parameter for 'at_least_or_none' variable type must be tuple of (value that equals true, number required), check config"
                def team_condition(team, var_name, comparison_value, count):
                    # get participants are booleans
                    participants_are = [p.variables[var_name] == comparison_value for p in team.participants]
                    return (not any(participants_are)) or (np.sum(participants_are) > 1)
                
                team_condition = partial(
                    team_condition,
                    var_name=var_name,
                    comparison_value=parameters[0],
                    count=parameters[1]
                )
                team_conditionals[var_name] = team_condition
                
#                 now we need to define what will happen if the condition is False
#                 In this case we have to add another participant, or pick a different team
                
                team_conditionals_protocols[var_name] = partial(
                    _at_least_or_none_protocol,
                    var_name=var_name,
                    comparison_value=parameters[0]
                )
                continue
            else:
                raise TypeError(
                    f'unrecognized variable type {var_type}, check config'
                )
            participant_scoring_functions[var_name] = func

        self.participant_scoring_functions = participant_scoring_functions
        self.team_conditionals = team_conditionals
        self.team_conditionals_protocols = team_conditionals_protocols
        return
    
    def score_participant(self, participant):
        """Score a single participant.
        
        Parameters
        ----------
        participant : Participant to score
        
        Returns
        -------
        float : summed score over variables
        dict : breakdown of scoring variables for participant
        """
        if participant.scores is None:
            scores = {var_name: func(participant.variables[var_name])\
                      for var_name, func in self.participant_scoring_functions.items()}
            score = np.sum(list(scores.values()))
            participant.scores = scores
            participant.score = score
        else:
            scores = participant.scores
            score = participant.score
        return score, scores
    
    def score_team(self, team):
        """Score a team.
        
        Either sums or averages the participant scores, check config.
        
        Parameters
        ----------
        team : Team to score
        
        Returns
        -------
        float : team score
        dict : breakdown of participant scores
        """
        if len(team.participants) == 0:
            return None, None
        scores = {participant.pid: self.score_participant(participant)[0]\
                  for participant in team.participants}
        score = self.team_scoring_function(list(scores.values()))
        return score, scores
    
    def score_teams(self, teams):
        """Score multiple teams.
        
        Parameters
        ----------
        teams : list of Teams
            teams to score
            
        Returns
        -------
        dict : team id team score mapping
        """
        scores = {team.tid: self.score_team(team)[0] for team in teams}
        return scores
    
    def check_team_conditionals(self, team):
        """Check if a team satisfies any conditions specified in config.
        
        Parameters
        ----------
        team : Team to check
        
        Returns
        -------
        boo :, whether all conditions are passed
        dict : any condition names that failed mapped to the protocol to be executed
            to fix the problem
        """
        conditions = {}
        for conditional_name, conditional_function in self.team_conditionals.items():
            conditions[conditional_name] = conditional_function(team)
        all_pass = all([passes for passes in conditions.values()])
        if all_pass:
            return True, None
        else:
            protocols = {}
            for condition_name, condition in conditions.items():
                if not condition:
                    protocols[condition_name] = self.team_conditionals_protocols[condition_name]
                    
            return False, protocols
                
#####################################################################
# Class to sort the participants in a registrar based on the scorers
# responses
#####################################################################

class Balancer:
    """Balances all participants in a registrar into teams.
    
    Parameters
    ----------
    registrar : Registrar containing all participants to sort
    scorer : Scorer that evaluates teams and individuals
    """
    def __init__(self, registrar: Registrar, scorer: Scorer):
        self.registrar = registrar
        self.scorer = scorer
        
        # add empty teams to the registrar to fill the new participants into
        current_space = 0
        for open_team in self.registrar.open_teams:
            current_space += open_team.space_left
        new_teams_needed = round(
            (len(self.registrar.teamless_participants) - current_space)/config.MAXIMUM_PARTICIPANTS
        )
        logger.info(f'Initializing {new_teams_needed} empty teams for a total\
    of {new_teams_needed+len(self.registrar.teams)}')
        for i in range(new_teams_needed):
            team = Team(-1)
            self.registrar.add_team(team)
            logger.info(f'Created new team {team.tid}')
        return
    
    def balance(self):
        """Sort the registrar."""
        while len(self.registrar.teamless_participants) > 0:
            self._balance()
            
        # get the final team scores
        overall_score, teams_scores = self._evaluate_many_teams(
            list(self.registrar.teams.values())
        )
        self.registrar.team_scores = teams_scores
        self.registrar.overall_score = overall_score
        return
    
    def _balance(self):
        """Attempt to balance.
        
        returns true when complete. Might break, in which case it could be executed again.
        """
        teams_to_score = list(self.registrar.teams.values())
        for participant in self.registrar.teamless_participants:
            handled_conditionals = self._place_participant(participant, self.registrar.open_teams)
            if handled_conditionals:
                break
            
        return
    
    def _attempt_add(self, participant, team):
        """Places a part on a team, scores the whole registrar, and then removes the part.
        
        Returns the score if the participant were on that team.
        """
        team.add(participant)
        score = self._evaluate_many_teams(list(self.registrar.teams.values()))[0]
        team.remove(participant.pid)
        return score
    
    def _place_participant(self, participant, potential_teams):
        """Place a part on a team to minimize the loss."""
        total_scores_when_added = {}
        for potential_team in potential_teams:
            total_scores_when_added[potential_team] = self._attempt_add(participant, potential_team)
        # now add the participant to that team for real
        team = min(total_scores_when_added, key=total_scores_when_added.get)
        team.add(participant)
        logger.info(f'Placed participant {participant.pid} in team {team.tid}')
        
        # check conditionals
        team_conditionals, protocols = self.scorer.check_team_conditionals(team)
        handled_conditionals = False
        if not team_conditionals:
            broken_conditions = list(protocols.keys())
            logger.info(f'Placement of participant {participant.pid} in team {team.tid}\
 caused the following condition_failures: {broken_conditions}')
            
            # execute a protocol and reevaluate
            protocols[broken_conditions[0]](self, participant, team)
            # team_conditionals, protocols = self.scorer.check_team_conditionals(team)
            handled_conditionals = True
        return handled_conditionals
        
    def _evaluate_many_teams(self, teams):
        """Have the scorer evaluate a bunch of teams and combine it."""
        scores = self.scorer.score_teams(teams)
        # now drop any NaNs, eg an empty team
        scores = np.array(list(scores.values())).astype(float)
        scores = scores[~np.isnan(scores)]
        return np.std(scores), scores
    
    
#####################################################################
# Execute the package!!
#####################################################################
parser.add_argument('--cleaned', dest='cleaned', action='store_const',
                     const=True, default=False,
                     help='Whether the data has already been cleaned.')

args = parser.parse_args()

if args.cleaned:
    df = pd.read_csv('cleaned_survey.csv', index_col=0)
    df[config.TEAM_CHOICE_COLUMN] = df[config.TEAM_CHOICE_COLUMN].apply(lambda x: eval(x))
else:
    df = load_clean_csv()
    df.to_csv('cleaned_survey.csv')
    
registrar = Registrar(df)

scorer = Scorer()

balancer = Balancer(registrar, scorer)

balancer.balance()

registrar.generate_report()