using JuMP, Gurobi, CSV, DataFrames

Blue1 = Matrix(CSV.read("processed_data/Blue1_dataset.csv", DataFrame));
Blue2 = Matrix(CSV.read("processed_data/Blue2_dataset.csv", DataFrame));

Orange1 = Matrix(CSV.read("processed_data/Orange1_dataset.csv", DataFrame));
Orange2 = Matrix(CSV.read("processed_data/Orange2_dataset.csv", DataFrame));

Green1 = Matrix(CSV.read("processed_data/Green1_dataset.csv", DataFrame));
Green2 = Matrix(CSV.read("processed_data/Green2_dataset.csv", DataFrame));

Red1 = Matrix(CSV.read("processed_data/Red1_dataset.csv", DataFrame));
Red2 = Matrix(CSV.read("processed_data/Red2_dataset.csv", DataFrame));

Blue1_std = Matrix(CSV.read("processed_data/Blue1_std.csv", DataFrame));
Blue2_std = Matrix(CSV.read("processed_data/Blue2_std.csv", DataFrame));

Orange1_std = Matrix(CSV.read("processed_data/Orange1_std.csv", DataFrame));
Orange2_std = Matrix(CSV.read("processed_data/Orange2_std.csv", DataFrame));

Green1_std = Matrix(CSV.read("processed_data/Green1_std.csv", DataFrame));
Green2_std = Matrix(CSV.read("processed_data/Green2_std.csv", DataFrame));

Red1_std = Matrix(CSV.read("processed_data/Red1_std.csv", DataFrame));
Red2_std = Matrix(CSV.read("processed_data/Red2_std.csv", DataFrame));

AvgFlowBlue = cat(Blue1, Blue2, dims = 3);
AvgFlowOrange = cat(Orange1, Orange2, dims = 3);
AvgFlowGreen = cat(Green1, Green2, dims = 3);
AvgFlowRed = cat(Red1, Red2, dims = 3);

deltaBlue = cat(Blue1_std, Blue2_std, dims = 3);
deltaOrange = cat(Orange1_std, Orange2_std, dims = 3);
deltaGreen = cat(Green1_std, Green2_std, dims = 3);
deltaRed = cat(Red1_std, Red2_std, dims = 3);

function optimize_blue()

    cost_blue = 2000;
    capacity_blue = 300;
    q=5;
    stations_b, times, directions = size(AvgFlowBlue);

    modelBlue = Model(Gurobi.Optimizer)
    
    @variable(modelBlue, x[1:directions, 1:times] >= 0, Int)
    @variable(modelBlue, u[1:directions, 1:times, 1:stations_b] >= 0, Int)
    @variable(modelBlue, s[1:directions, 1:times] >= 0, Int)
    @variable(modelBlue, r[1:directions, 1:times] >= 0, Int)

    @constraint(modelBlue, [d=1:directions, j=2:times, i=1:stations_b], 
        u[d,j,i] + capacity_blue * (x[d,j] + s[d,j]) >= AvgFlowBlue[i,j,d] + u[d,j-1,i])
    @constraint(modelBlue, [d=1:directions, i=1:stations_b], 
        u[d,1,i] + capacity_blue * (x[d,1] + s[d,1]) >= AvgFlowBlue[i,1,d])
    @constraint(modelBlue,  [d=1:directions,  i=1:stations_b], u[d,9,i] == 0 )

    @constraint(modelBlue,  r[2,1] ==   x[1,1] )
    @constraint(modelBlue,  r[1,1] ==  x[2,1]  ) 
    @constraint(modelBlue, [j=2:times], r[2,j] ==   x[1,j] +  s[1,j]  - s[2,j] + r[2,j-1])
    @constraint(modelBlue, [j=2:times], r[1,j] ==  x[2,j] +  s[2,j] - s[1,j] + r[1,j-1])

    @constraint(modelBlue, [d=1:directions], s[d,1] == 0)
    @constraint(modelBlue, [d=1:directions, j=2:times], s[d,j] <= r[d,j-1])

    @constraint(modelBlue, [d=1:directions], r[1, times]  >=  x[1,1])
    @constraint(modelBlue, [d=1:directions], r[2, times]  >=  x[2,1])

    @constraint(modelBlue, [j=1:times, d=1:directions], x[d,j] + s[d,j] >= 1)

    #@constraint(modelBlue, [j=1:times], x[1,j] + x[2,j] + s[1,j] + s[2,j] + r[1,j] + r[2,j] <= number_trains)

    @objective(modelBlue, Min, sum(sum(cost_blue * x[d,j] + 0.95 * cost_blue * s[d,j] + 
                sum(q * u[d,j,i] for i=1:stations_b) for d=1:directions) for j=1:times))

    optimize!(modelBlue)

    return value.(x), value.(s), value.(r), value.(u), objective_value(modelBlue)
end


function optimize_blue_robust(Gamma)
    cost_blue = 2000;
    capacity_blue = 300;
    q=5;
    stations_b, times, directions = size(AvgFlowBlue);

    modelBlue = Model(Gurobi.Optimizer)

    @variable(modelBlue, x[1:directions, 1:times] >= 0, Int)
    @variable(modelBlue, u[1:directions, 1:times, 1:stations_b] >= 0, Int)
    @variable(modelBlue, s[1:directions, 1:times] >= 0, Int)
    @variable(modelBlue, r[1:directions, 1:times], Int)
    @variable(modelBlue, alpha[1:directions, 1:times, 1:stations_b] >=0, Int)
    @variable(modelBlue, beta[1:directions, 1:times, 1:stations_b]  >=0, Int)
    @variable(modelBlue, l  >=0, Int)
    @variable(modelBlue, lambda[1:directions, 1:times, 1:stations_b]  >=0, Int)
    @variable(modelBlue, phi[1:directions, 1:times, 1:stations_b]  >=0, Int)

    @constraint(modelBlue, [d=1:directions, j=2:times, i=1:stations_b], 
        u[d,j,i] + capacity_blue * (x[d,j] + s[d,j]) >= -AvgFlowBlue[i,j,d]*alpha[d,j,i] + AvgFlowBlue[i,j,d]*beta[d,j,i] + Gamma*l + u[d,j-1,i])
    @constraint(modelBlue, [d=1:directions, i=1:stations_b], 
        u[d,1,i] + capacity_blue * (x[d,1] + s[d,1]) >= -AvgFlowBlue[i,1,d]*alpha[d,1,i] + AvgFlowBlue[i,1,d]*beta[d,1,i] + Gamma*l )
    @constraint(modelBlue,  [d=1:directions,  i=1:stations_b], u[d,9,i] == 0 )

    @constraint(modelBlue, [d=1:directions, j=1:times, i=1:stations_b], -alpha[d,j,i] + beta[d,j,i] >= 1)
    @constraint(modelBlue, [d=1:directions, j=1:times, i=1:stations_b], -deltaBlue[i,j,d] * alpha[d,j,i] - deltaBlue[i,j,d] * beta[d,j,i] + lambda[d,j,i] - phi[d,j,i] >= 0)
    @constraint(modelBlue, [d=1:directions, j=1:times, i=1:stations_b], l - lambda[d,j,i] - phi[d,j,i] >= 0)

    @constraint(modelBlue,  r[2,1] ==   x[1,1] )
    @constraint(modelBlue,  r[1,1] ==  x[2,1]  ) 
    @constraint(modelBlue, [j=2:times], r[2,j] ==   x[1,j] +  s[1,j]  - s[2,j] + r[2,j-1])
    @constraint(modelBlue, [j=2:times], r[1,j] ==  x[2,j] +  s[2,j] - s[1,j] + r[1,j-1])

    @constraint(modelBlue, [d=1:directions], s[d,1] == 0)
    @constraint(modelBlue, [d=1:directions, j=2:times], s[d,j] <= r[d,j-1])

    @constraint(modelBlue, [d=1:directions], r[1, times]  >=  x[1,1])
    @constraint(modelBlue, [d=1:directions], r[2, times]  >=  x[2,1])

    @constraint(modelBlue, [j=1:times, d=1:directions], x[d,j] + s[d,j] >= 1)

    @objective(modelBlue, Min, sum(sum(cost_blue * x[d,j] + 0.95 * cost_blue * s[d,j] + 
                sum(q * u[d,j,i] for i=1:stations_b) for d=1:directions) for j=1:times))

    optimize!(modelBlue)

    return value.(x), value.(s), value.(r), value.(u), objective_value(modelBlue)
end

function optimize_orange()
    cost_orange = 5000;
    capacity_orange = 1000;
    q=5;
    stations_o, times, directions = size(AvgFlowOrange);

    modelOrange = Model(Gurobi.Optimizer)

    @variable(modelOrange, x[1:directions, 1:times] >= 0, Int)
    @variable(modelOrange, u[1:directions, 1:times, 1:stations_o] >= 0, Int)
    @variable(modelOrange, s[1:directions, 1:times] >= 0, Int)
    @variable(modelOrange, r[1:directions, 1:times], Int)

    @constraint(modelOrange, [d=1:directions, j=2:times, i=1:stations_o], 
        u[d,j,i] + capacity_orange * (x[d,j] + s[d,j]) >= AvgFlowOrange[i,j,d] + u[d,j-1,i])
    @constraint(modelOrange, [d=1:directions, i=1:stations_o], 
        u[d,1,i] + capacity_orange * (x[d,1] + s[d,1]) >= AvgFlowOrange[i,1,d])
    @constraint(modelOrange,  [d=1:directions, i=1:stations_o],  u[d,9,i] == 0 )

    modelOrange = Model(Gurobi.Optimizer)

    @variable(modelOrange, x[1:directions, 1:times] >= 0, Int)
    @variable(modelOrange, u[1:directions, 1:times, 1:stations_o] >= 0, Int)
    @variable(modelOrange, s[1:directions, 1:times] >= 0, Int)
    @variable(modelOrange, r[1:directions, 1:times], Int)

    @constraint(modelOrange, [d=1:directions, j=2:times, i=1:stations_o], 
        u[d,j,i] + capacity_orange * (x[d,j] + s[d,j]) >= AvgFlowOrange[i,j,d] + u[d,j-1,i])
    @constraint(modelOrange, [d=1:directions, i=1:stations_o], 
        u[d,1,i] + capacity_orange * (x[d,1] + s[d,1]) >= AvgFlowOrange[i,1,d])
    @constraint(modelOrange,  [d=1:directions, i=1:stations_o],  u[d,9,i] == 0 )

    @constraint(modelOrange,  r[2,1] ==   x[1,1] )
    @constraint(modelOrange,  r[1,1] ==  x[2,1]  ) 
    @constraint(modelOrange, [j=2:times], r[2,j] ==   x[1,j] +  s[1,j]  - s[2,j] + r[2,j-1])
    @constraint(modelOrange, [j=2:times], r[1,j] ==  x[2,j] +  s[2,j] - s[1,j] + r[1,j-1])

    @constraint(modelOrange, [d=1:directions], s[d,1] == 0)
    @constraint(modelOrange, [d=1:directions, j=2:times], s[d,j] <= r[d,j-1])

    @constraint(modelOrange, [d=1:directions], r[1, times]  >=  x[1,1])
    @constraint(modelOrange, [d=1:directions], r[2, times]  >=  x[2,1])

    @constraint(modelOrange, [j=1:times, d=1:directions], x[d,j] + s[d,j] >= 1)

    @objective(modelOrange, Min, sum(sum(cost_orange * x[d,j] + 0.9 * cost_orange * s[d,j] + 
                sum(q * u[d,j,i] for i=1:stations_o) for d=1:directions) for j=1:times))

    optimize!(modelOrange)

    return value.(x), value.(s), value.(r), value.(u), objective_value(modelOrange)
end

function optimize_orange_robust(Gamma)
    cost_orange = 5000;
    capacity_orange = 1000;
    q=5;
    stations_o, times, directions = size(AvgFlowOrange);

    modelOrange = Model(Gurobi.Optimizer)

    @variable(modelOrange, x[1:directions, 1:times] >= 0, Int)
    @variable(modelOrange, u[1:directions, 1:times, 1:stations_o] >= 0, Int)
    @variable(modelOrange, s[1:directions, 1:times] >= 0, Int)
    @variable(modelOrange, r[1:directions, 1:times], Int)
    @variable(modelOrange, alpha[1:directions, 1:times, 1:stations_o] >=0, Int)
    @variable(modelOrange, beta[1:directions, 1:times, 1:stations_o]  >=0, Int)
    @variable(modelOrange, l  >=0, Int)
    @variable(modelOrange, lambda[1:directions, 1:times, 1:stations_o]  >=0, Int)
    @variable(modelOrange, phi[1:directions, 1:times, 1:stations_o]  >=0, Int)


    @constraint(modelOrange, [d=1:directions, j=2:times, i=1:stations_o], 
        u[d,j,i] + capacity_orange * (x[d,j] + s[d,j]) >= -AvgFlowOrange[i,j,d]*alpha[d,j,i] + AvgFlowOrange[i,j,d]*beta[d,j,i] + Gamma*l + u[d,j-1,i])
    @constraint(modelOrange, [d=1:directions, i=1:stations_o], 
        u[d,1,i] + capacity_orange * (x[d,1] + s[d,1]) >= -AvgFlowOrange[i,1,d]*alpha[d,1,i] + AvgFlowOrange[i,1,d]*beta[d,1,i] + Gamma*l )
    @constraint(modelOrange,  [d=1:directions, i=1:stations_o],  u[d,9,i] == 0 )

    @constraint(modelOrange, [d=1:directions, j=1:times, i=1:stations_o], -alpha[d,j,i] + beta[d,j,i] >= 1)
    @constraint(modelOrange, [d=1:directions, j=1:times, i=1:stations_o], -deltaOrange[i,j,d] * alpha[d,j,i] - deltaOrange[i,j,d] * beta[d,j,i] + lambda[d,j,i] - phi[d,j,i] >= 0)
    @constraint(modelOrange, [d=1:directions, j=1:times, i=1:stations_o], l - lambda[d,j,i] - phi[d,j,i] >= 0)

    @constraint(modelOrange,  r[2,1] ==   x[1,1] )
    @constraint(modelOrange,  r[1,1] ==  x[2,1]  ) 
    @constraint(modelOrange, [j=2:times], r[2,j] ==   x[1,j] +  s[1,j]  - s[2,j] + r[2,j-1])
    @constraint(modelOrange, [j=2:times], r[1,j] ==  x[2,j] +  s[2,j] - s[1,j] + r[1,j-1])

    @constraint(modelOrange, [d=1:directions], s[d,1] == 0)
    @constraint(modelOrange, [d=1:directions, j=2:times], s[d,j] <= r[d,j-1])

    @constraint(modelOrange, [d=1:directions], r[1, times]  >=  x[1,1])
    @constraint(modelOrange, [d=1:directions], r[2, times]  >=  x[2,1])

    @constraint(modelOrange, [j=1:times, d=1:directions], x[d,j] + s[d,j] >= 1)

    @objective(modelOrange, Min, sum(sum(cost_orange * x[d,j] + 0.9 * cost_orange * s[d,j] + 
                sum(q * u[d,j,i] for i=1:stations_o) for d=1:directions) for j=1:times))

    optimize!(modelOrange)

    return value.(x), value.(s), value.(r), value.(u), objective_value(modelOrange)
end

function optimize_red()
    cost_red = 5000;
    capacity_red = 1000;
    q=5;
    stations_r, times, directions = size(AvgFlowRed);
    lines = 2;

    Red1 = ["Alewife", "Davis", "Porter", "Harvard", "Central", "Kendall/MIT", "Charles/MGH",
     "Park Street", "Downtown Crossing", "South Station", "Broadway", "Andrew", "JFK/Umass",
      "Savin Hill", "Fields Corner", "Shawmut", "Ashmont"];
    Red2 = ["Alewife", "Davis", "Porter", "Harvard", "Central", "Kendall/MIT", "Charles/MGH",
     "Park Street", "Downtown Crossing", "South Station", "Broadway", "Andrew", "JFK/Umass",
      "North Quincy", "Wollaston", "Quincy Center", "Quincy Adams", "Braintree"];
    red_stations = unique(vcat(Red1, Red2));   

    z_red = zeros((2,22));
    for i = 1:22
        if red_stations[i] in Red1
            z_red[1,i] = 1
        end
        if red_stations[i] in Red2
            z_red[2,i] = 1
        end
    end

    modelRed = Model(Gurobi.Optimizer)

    @variable(modelRed, x[1:directions, 1:times, 1:lines] >= 0, Int)
    @variable(modelRed, u[1:directions, 1:times, 1:stations_r] >= 0, Int)
    @variable(modelRed, s[1:directions, 1:times, 1:lines] >= 0, Int)
    @variable(modelRed, r[1:directions, 1:times, 1:lines], Int)

    @constraint(modelRed, [d=1:directions, j=2:times, i=1:stations_r], 
        u[d,j,i] + (sum(capacity_red * (x[d,j,l] + s[d,j,l]) * z_red[l,i] for l=1:lines)) >= AvgFlowRed[i,j,d] + u[d,j-1,i])
    @constraint(modelRed, [d=1:directions,  i=1:stations_r], 
        u[d,1,i] + (sum(capacity_red * (x[d,1,l] + s[d,1,l]) * z_red[l,i] for l=1:lines)) >= AvgFlowRed[i,1,d])
    @constraint(modelRed,  [d=1:directions, i=1:stations_r],  u[d,9,i] == 0 )
    

    @constraint(modelRed, [l=1:lines], r[2,1,l] ==   x[1,1,l] )
    @constraint(modelRed, [l=1:lines], r[1,1,l] ==  x[2,1,l]  ) 
    @constraint(modelRed, [j=2:times, l=1:lines], r[2,j,l] ==   x[1,j,l] +  s[1,j,l]  - s[2,j,l] + r[2,j-1,l])
    @constraint(modelRed, [j=2:times, l=1:lines], r[1,j,l] ==  x[2,j,l] +  s[2,j,l] - s[1,j,l] + r[1,j-1,l])

    @constraint(modelRed, [d=1:directions, l=1:lines], s[d,1,l] == 0)
    @constraint(modelRed, [d=1:directions, j=2:times, l=1:lines], s[d,j,l] <= r[d,j-1,l])

    @constraint(modelRed, [d=1:directions, l=1:lines], r[1, times, l]  >=  x[1,1,l])
    @constraint(modelRed, [d=1:directions, l=1:lines], r[2, times, l]  >=  x[2,1,l])

    @constraint(modelRed, [j=1:times, d=1:directions, l=1:lines], x[d,j,l] + s[d,j,l] >= 1)

    @objective(modelRed, Min, sum( sum(cost_red * x[d,j,l] + 0.9 * cost_red * s[d,j,l] for l=1:lines ) + 
                    sum(q * u[d,j,i] for i=1:stations_r) for d=1:directions,  j=1:times))

    optimize!(modelRed)

    return value.(x), value.(s), value.(u), objective_value(modelRed)
end

function optimize_red_robust(Gamma)
    cost_red = 5000;
    capacity_red = 1000;
    q=5;
    stations_r, times, directions = size(AvgFlowRed);
    lines = 2;

    Red1 = ["Alewife", "Davis", "Porter", "Harvard", "Central", "Kendall/MIT", "Charles/MGH",
     "Park Street", "Downtown Crossing", "South Station", "Broadway", "Andrew", "JFK/Umass",
      "Savin Hill", "Fields Corner", "Shawmut", "Ashmont"];
    Red2 = ["Alewife", "Davis", "Porter", "Harvard", "Central", "Kendall/MIT", "Charles/MGH",
     "Park Street", "Downtown Crossing", "South Station", "Broadway", "Andrew", "JFK/Umass",
      "North Quincy", "Wollaston", "Quincy Center", "Quincy Adams", "Braintree"];
    red_stations = unique(vcat(Red1, Red2));   

    z_red = zeros((2,22));
    for i = 1:22
        if red_stations[i] in Red1
            z_red[1,i] = 1
        end
        if red_stations[i] in Red2
            z_red[2,i] = 1
        end
    end
    modelRed = Model(Gurobi.Optimizer)

    @variable(modelRed, x[1:directions, 1:times, 1:lines] >= 0, Int)
    @variable(modelRed, u[1:directions, 1:times, 1:stations_r] >= 0, Int)
    @variable(modelRed, s[1:directions, 1:times, 1:lines] >= 0, Int)
    @variable(modelRed, r[1:directions, 1:times, 1:lines], Int)
    @variable(modelRed, alpha[1:directions, 1:times, 1:stations_r] >=0, Int)
    @variable(modelRed, beta[1:directions, 1:times, 1:stations_r]  >=0, Int)
    @variable(modelRed, l  >=0, Int)
    @variable(modelRed, lambda[1:directions, 1:times, 1:stations_r]  >=0, Int)
    @variable(modelRed, phi[1:directions, 1:times, 1:stations_r]  >=0, Int)

    @constraint(modelRed, [d=1:directions, j=2:times, i=1:stations_r], 
        u[d,j,i] + (sum(capacity_red * (x[d,j,l] + s[d,j,l]) * z_red[l,i] for l=1:lines)) >= -AvgFlowRed[i,j,d]*alpha[d,j,i] + AvgFlowRed[i,j,d]*beta[d,j,i] + Gamma*l + u[d,j-1,i])
    @constraint(modelRed, [d=1:directions,  i=1:stations_r], 
        u[d,1,i] + (sum(capacity_red * (x[d,1,l] + s[d,1,l]) * z_red[l,i] for l=1:lines)) >= -AvgFlowRed[i,1,d]*alpha[d,1,i] + AvgFlowRed[i,1,d]*beta[d,1,i] + Gamma*l )
    @constraint(modelRed,  [d=1:directions, i=1:stations_r],  u[d,9,i] == 0 )

    @constraint(modelRed, [d=1:directions, j=1:times, i=1:stations_r], -alpha[d,j,i] + beta[d,j,i] >= 1)
    @constraint(modelRed, [d=1:directions, j=1:times, i=1:stations_r], -deltaRed[i,j,d] * alpha[d,j,i] - deltaRed[i,j,d] * beta[d,j,i] + lambda[d,j,i] - phi[d,j,i] >= 0)
    @constraint(modelRed, [d=1:directions, j=1:times, i=1:stations_r], l - lambda[d,j,i] - phi[d,j,i] >= 0)

    @constraint(modelRed, [l=1:lines], r[2,1,l] ==   x[1,1,l] )
    @constraint(modelRed, [l=1:lines], r[1,1,l] ==  x[2,1,l]  ) 
    @constraint(modelRed, [j=2:times, l=1:lines], r[2,j,l] ==   x[1,j,l] +  s[1,j,l]  - s[2,j,l] + r[2,j-1,l])
    @constraint(modelRed, [j=2:times, l=1:lines], r[1,j,l] ==  x[2,j,l] +  s[2,j,l] - s[1,j,l] + r[1,j-1,l])

    @constraint(modelRed, [d=1:directions, l=1:lines], s[d,1,l] == 0)
    @constraint(modelRed, [d=1:directions, j=2:times, l=1:lines], s[d,j,l] <= r[d,j-1,l])

    @constraint(modelRed, [d=1:directions, l=1:lines], r[1, times, l]  >=  x[1,1,l])
    @constraint(modelRed, [d=1:directions, l=1:lines], r[2, times, l]  >=  x[2,1,l])

    @constraint(modelRed, [j=1:times, d=1:directions, l=1:lines], x[d,j,l] + s[d,j,l] >= 1)

    @objective(modelRed, Min, sum( sum(cost_red * x[d,j,l] + 0.9 * cost_red * s[d,j,l] for l=1:lines ) + 
                    sum(q * u[d,j,i] for i=1:stations_r) for d=1:directions,  j=1:times))

    optimize!(modelRed)

    return value.(x), value.(s), value.(u), value.(r), objective_value(modelRed)
end

function optimize_green()
    capacity_green = 200;
    cost_green = 1300;
    q = 5;
    stations_g, times, directions = size(AvgFlowGreen);
    lines = 4;

    Green1 = ["Lechmere", "Science Park", "North Station", "Haymarket", "Government Center", "Park Street", "Boylston",
    "Arlington", "Copley", "Hynes Convention Center", "Kenmore", "Blandford Street", "Boston Univ. East", 
    "Boston Univ. Central", "Boston Univ. West", "Saint Paul Street", "Pleasant Street", "Babcock Street",
    "Harvard Ave.","Griggs Street", "Allston Street", "Warren Street", "Washington Street", "Sutherland Road",
    "Chiswick Road", "Chestnut Hill Ave.", "South Street"];
    Green2 = ["Government Center", "Park Street", "Boylston", "Arlington", "Copley", "Hynes Convention Center", "Kenmore", 
    "Saint Mary Street", "Hawes Street", "Kent Street", "Saint Paul Street", "Coolidge Corner", "Summit Ave.", 
    "Brandon Hall", "Fairbanks Street", "Washington Square", "Tappan Street", "Dean Road", "Englewood Ave.", "Cleveland Circle"];
    Green3 = ["Lechmere", "Science Park", "North Station", "Haymarket", "Government Center", "Park Street", "Boylston", 
    "Arlington", "Copley", "Hynes Convention Center", "Kenmore", "Fenway", "Longwood", "Brookline Village", "Brookline Hills",
    "Beaconsfield", "Reservoir", "Chestnut Hill Ave.", "Newton Centre", "Newton Highlands", "Eliot", "Waban", "Woodland", 
    "Riverside"];
    Green4 = ["Lechmere", "Science Park", "North Station", "Haymarket", "Government Center", "Park Street", "Boylston",
        "Arlington", "Copley", "Prudential", "Symphony", "Northeastern University", "Museum of Fine Arts", "Longwood",
        "Brigham Circle", "Fenwood Road", "Mission Park", "Riverway", "Back of the Hill", "Heath Street"];
    green_stations = unique(vcat(Green1, Green2, Green3, Green4));
    stations_g=61;
    
    z_green = zeros((4,61));
    for i = 1:61
        if green_stations[i] in Green1
            z_green[1,i] = 1
        end
        if green_stations[i] in Green2
            z_green[2,i] = 1
        end
        if green_stations[i] in Green3 
            z_green[3,i] = 1
        end
        if green_stations[i] in Green4 
            z_green[4,i] = 1
        end
    end

    modelGreen = Model(Gurobi.Optimizer)

    @variable(modelGreen, x[1:directions, 1:times, 1:lines] >= 0, Int)
    @variable(modelGreen, u[1:directions, 1:times, 1:stations_g] >= 0, Int)
    @variable(modelGreen, s[1:directions, 1:times, 1:lines] >= 0, Int)
    @variable(modelGreen, r[1:directions, 1:times, 1:lines] >= 0, Int)

    @constraint(modelGreen, [d=1:directions, j=2:times, i=1:stations_g], 
        (sum(capacity_green * (x[d,j,l]) * z_green[l,i] for l=1:lines)) >= AvgFlowGreen[i,j,d])
    @constraint(modelGreen, [d=1:directions, i=1:stations_g], 
        u[d,1,i] + (sum(capacity_green * (x[d,1,l] + s[d,1,l]) * z_green[l,i] for l=1:lines)) >= AvgFlowGreen[i,1,d])
    @constraint(modelGreen, [d=1:directions, i=1:stations_g], u[d,9,i]==0)

    @constraint(modelGreen, [l=1:lines], r[2,1,l] ==   x[1,1,l])
    @constraint(modelGreen, [l=1:lines], r[1,1,l] ==  x[2,1,l]) 
    @constraint(modelGreen, [j=2:times, l=1:lines], r[2,j,l] ==   x[1,j,l] +  s[1,j,l]  - s[2,j,l] + r[2,j-1,l])
    @constraint(modelGreen, [j=2:times, l=1:lines], r[1,j,l] ==  x[2,j,l] +  s[2,j,l] - s[1,j,l] + r[1,j-1,l])

    @constraint(modelGreen, [d=1:directions, l=1:lines], s[d,1,l] == 0)
    @constraint(modelGreen, [d=1:directions, j=2:times, l=1:lines], s[d,j,l] <= r[d,j-1,l])

    @constraint(modelGreen, [d=1:directions, l=1:lines], r[1, times, l]  >=  x[1,1,l])
    @constraint(modelGreen, [d=1:directions, l=1:lines], r[2, times, l]  >=  x[2,1,l])

    @constraint(modelGreen, [j=1:times, d=1:directions, l=1:lines], x[d,j,l] + s[d,j,l] >= 1)

    @objective(modelGreen, Min, sum( sum(cost_green * x[d,j,l] + 0.9 * cost_green * s[d,j,l] for l=1:lines ) + 
                    sum(q * u[d,j,i] for i=1:stations_g) for d=1:directions,  j=1:times))

    optimize!(modelGreen)

    return value.(x), value.(s), value.(u), value.(r), objective_value(modelGreen)
end

function optimize_green_robust(Gamma)
    capacity_green = 200;
    cost_green = 1300;
    q = 5;
    stations_g, times, directions = size(AvgFlowGreen);
    lines = 4;

    Green1 = ["Lechmere", "Science Park", "North Station", "Haymarket", "Government Center", "Park Street", "Boylston",
    "Arlington", "Copley", "Hynes Convention Center", "Kenmore", "Blandford Street", "Boston Univ. East", 
    "Boston Univ. Central", "Boston Univ. West", "Saint Paul Street", "Pleasant Street", "Babcock Street",
    "Harvard Ave.","Griggs Street", "Allston Street", "Warren Street", "Washington Street", "Sutherland Road",
    "Chiswick Road", "Chestnut Hill Ave.", "South Street"];
    Green2 = ["Government Center", "Park Street", "Boylston", "Arlington", "Copley", "Hynes Convention Center", "Kenmore", 
    "Saint Mary Street", "Hawes Street", "Kent Street", "Saint Paul Street", "Coolidge Corner", "Summit Ave.", 
    "Brandon Hall", "Fairbanks Street", "Washington Square", "Tappan Street", "Dean Road", "Englewood Ave.", "Cleveland Circle"];
    Green3 = ["Lechmere", "Science Park", "North Station", "Haymarket", "Government Center", "Park Street", "Boylston", 
    "Arlington", "Copley", "Hynes Convention Center", "Kenmore", "Fenway", "Longwood", "Brookline Village", "Brookline Hills",
    "Beaconsfield", "Reservoir", "Chestnut Hill Ave.", "Newton Centre", "Newton Highlands", "Eliot", "Waban", "Woodland", 
    "Riverside"];
    Green4 = ["Lechmere", "Science Park", "North Station", "Haymarket", "Government Center", "Park Street", "Boylston",
        "Arlington", "Copley", "Prudential", "Symphony", "Northeastern University", "Museum of Fine Arts", "Longwood",
        "Brigham Circle", "Fenwood Road", "Mission Park", "Riverway", "Back of the Hill", "Heath Street"];
    green_stations = unique(vcat(Green1, Green2, Green3, Green4));
    stations_g=61;
    
    z_green = zeros((4,61));
    for i = 1:61
        if green_stations[i] in Green1
            z_green[1,i] = 1
        end
        if green_stations[i] in Green2
            z_green[2,i] = 1
        end
        if green_stations[i] in Green3 
            z_green[3,i] = 1
        end
        if green_stations[i] in Green4 
            z_green[4,i] = 1
        end
    end

    modelGreen = Model(Gurobi.Optimizer)

    @variable(modelGreen, x[1:directions, 1:times, 1:lines] >= 0, Int)
    @variable(modelGreen, u[1:directions, 1:times, 1:stations_g] >= 0, Int)
    @variable(modelGreen, s[1:directions, 1:times, 1:lines] >= 0, Int)
    @variable(modelGreen, r[1:directions, 1:times, 1:lines] >= 0, Int)
    @variable(modelGreen, alpha[1:directions, 1:times, 1:stations_g] >=0, Int)
    @variable(modelGreen, beta[1:directions, 1:times, 1:stations_g]  >=0, Int)
    @variable(modelGreen, l  >=0, Int)
    @variable(modelGreen, lambda[1:directions, 1:times, 1:stations_g]  >=0, Int)
    @variable(modelGreen, phi[1:directions, 1:times, 1:stations_g]  >=0, Int)

    @constraint(modelGreen, [d=1:directions, j=2:times, i=1:stations_g], 
        (sum(capacity_green * (x[d,j,l]) * z_green[l,i] for l=1:lines)) >= -AvgFlowGreen[i,j,d]*alpha[d,j,i] + AvgFlowGreen[i,j,d]*beta[d,j,i] + Gamma*l + u[d,j-1,i])
    @constraint(modelGreen, [d=1:directions, i=1:stations_g], 
        u[d,1,i] + (sum(capacity_green * (x[d,1,l] + s[d,1,l]) * z_green[l,i] for l=1:lines)) >= -AvgFlowGreen[i,1,d]*alpha[d,1,i] + AvgFlowGreen[i,1,d]*beta[d,1,i] + Gamma*l )
    @constraint(modelGreen, [d=1:directions, i=1:stations_g], u[d,9,i]==0)

    @constraint(modelGreen, [d=1:directions, j=1:times, i=1:stations_g], -alpha[d,j,i] + beta[d,j,i] >= 1)
    @constraint(modelGreen, [d=1:directions, j=1:times, i=1:stations_g], -deltaGreen[i,j,d] * alpha[d,j,i] - deltaGreen[i,j,d] * beta[d,j,i] + lambda[d,j,i] - phi[d,j,i] >= 0)
    @constraint(modelGreen, [d=1:directions, j=1:times, i=1:stations_g], l - lambda[d,j,i] - phi[d,j,i] >= 0)

    @constraint(modelGreen, [l=1:lines], r[2,1,l] ==   x[1,1,l])
    @constraint(modelGreen, [l=1:lines], r[1,1,l] ==  x[2,1,l]) 
    @constraint(modelGreen, [j=2:times, l=1:lines], r[2,j,l] ==   x[1,j,l] +  s[1,j,l]  - s[2,j,l] + r[2,j-1,l])
    @constraint(modelGreen, [j=2:times, l=1:lines], r[1,j,l] ==  x[2,j,l] +  s[2,j,l] - s[1,j,l] + r[1,j-1,l])

    @constraint(modelGreen, [d=1:directions, l=1:lines], s[d,1,l] == 0)
    @constraint(modelGreen, [d=1:directions, j=2:times, l=1:lines], s[d,j,l] <= r[d,j-1,l])

    @constraint(modelGreen, [d=1:directions, l=1:lines], r[1, times, l]  >=  x[1,1,l])
    @constraint(modelGreen, [d=1:directions, l=1:lines], r[2, times, l]  >=  x[2,1,l])

    @constraint(modelGreen, [j=1:times, d=1:directions, l=1:lines], x[d,j,l] + s[d,j,l] >= 1)

    @objective(modelGreen, Min, sum( sum(cost_green * x[d,j,l] + 0.9 * cost_green * s[d,j,l] for l=1:lines ) + 
                    sum(q * u[d,j,i] for i=1:stations_g) for d=1:directions,  j=1:times))

    optimize!(modelGreen)

    return value.(x), value.(s), value.(r), value.(u), objective_value(modelGreen)
end