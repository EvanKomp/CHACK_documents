# chack_team_balancer
Sorts unassigned team members for the most balanced possible teams. The algorithm works by assigning a weight or function to each of a number of variables and assigning a score for each participant. A team score is then determined as the mean score of participants. The algorithm loops through participants, determines which team (or a new team) would minimize the standard deviation of team scores when the participant is placed there. A function is applied to the standard deviation, such as the square, and considered when minimizing. This allows for tuning of the algorithm. Finally, conditions can be applied to the teams such as a max size, or enforce a certain number of participants of a certain gender. See the config file.

Example, if two variables are considered "A" with weight 2.0, and "B" with function "square"

A participant with `A = 2` and `B = 3` will have a score of `2 * 2.0 + 3**2 = 13.0`

A team's score is some aggregation of participant scores such as sum or mean.

## contents
```
environment.yml           # conda environment specification file
config.py                 # python file allowijg user to specify how the balancer will run
baance.py                 # python script accessible by command line to create teams
```

## Use
- Install and activate the conda environment at `environment.yml`
- Download a csv style spreadsheet containing the names of participants, their desired teamates, and any parameters to be used for sorting
- Modify user parameters in `config.py`. Descriptions for the variables are provided in that file as comments
- Execute `python balance.py`
- Assigned teams are output as `teams_report.csv`. Each row is a team, and is of form (team num, \*team members, team score)
- Info on each participant, how they scored, what team they are on is output as `participants_report.csv`
