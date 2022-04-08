# chack_team_balancer
Sorts unassigned team members for the most balanced possible teams. Thw algorithm works by assigning a weight to each of a number of variables and assigning a score for each participant. A team score is then determined as the mean score of participants. The algorithm loops through participants, determines which team (or a new team) would minimize the standard deviation of team scores when the participant is placed there. A function is applied to the standard deviation, such as the square, and considered when minimizing. This allows for tuning of the algorithm. Finally, conditions can be applied to the teams such as a max size, or enforce a certain number of participants of a certain gender. See the config file.

## Use
- Install and activate the conda environment at `environment.yml`
- Download a csv style spreadsheet containing the names of participants, their desired teamates, and any parameters to be used for sorting
- Modify user parameters in `config.py`. Descriptions for the variables are provided in that file as comments
- Execute `python balance.py`
- Assigned teams are output as `teams_report.csv`. Each row is a team, and is of form (team num, \*team members, team score)
- Info on each participant, how they scored, what team they are on is output as `participants_report.csv`
