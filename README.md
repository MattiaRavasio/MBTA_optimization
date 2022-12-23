# MBTA_optimization

This is a repository for the Optimization methods group project. We attempted scheduling the MBTA trains using open source data about the average flow of passengers per station. In order to propose a model suited for a real-world deployment we evaluated a robust framwrok to take into account passenger demand uncertanity, and how to operate under a limited amount of fleet.  

The code is mainly wirtten in Julia, using JuMP and Gurobi, with some pre-processing done in python. 

## The repository

- **Documents**: presentation and report for the project
- **processed_data**: ready-to-use data for the JuMP model
- **tables**: A few tables with some results
- **Optimization.ipynb**: some experimentions and plottings 
- **MBTA_data.csv**: raw data that was made available by the MBTA
- **formulation.jl**: formulations proposed
- **preprocessing.py**: pre-processing file

## License

[MIT](https://choosealicense.com/licenses/mit/)
