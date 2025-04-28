from pulp import *

# Create the model
model = LpProblem("Resource_Allocation", LpMinimize)

# Decision variables
x = {}
for i in range(1, 7):
    for j in ['A', 'B', 'C']:
        x[i, j] = LpVariable(f"x_{i}_{j}", lowBound=0)

y = {}
for i in range(1, 7):
    y[i] = LpVariable(f"y_{i}", cat=LpBinary)

# Objective function
obj_func = (
    18*x[1,'A'] + 15*x[1,'B'] + 12*x[1,'C'] +
    13*x[2,'A'] + 10*x[2,'B'] + 17*x[2,'C'] +
    16*x[3,'A'] + 14*x[3,'B'] + 18*x[3,'C'] +
    19*x[4,'A'] + 15*x[4,'B'] + 16*x[4,'C'] +
    17*x[5,'A'] + 19*x[5,'B'] + 12*x[5,'C'] +
    14*x[6,'A'] + 16*x[6,'B'] + 12*x[6,'C'] +
    405*y[1] + 390*y[2] + 450*y[3] + 368*y[4] + 520*y[5] + 465*y[6]
)
model += obj_func

# Row constraints
model += x[1,'A'] + x[1,'B'] + x[1,'C'] == 11.2 * y[1]
model += x[2,'A'] + x[2,'B'] + x[2,'C'] == 10.5 * y[2]
model += x[3,'A'] + x[3,'B'] + x[3,'C'] == 12.8 * y[3]
model += x[4,'A'] + x[4,'B'] + x[4,'C'] == 9.3 * y[4]
model += x[5,'A'] + x[5,'B'] + x[5,'C'] == 10.8 * y[5]
model += x[6,'A'] + x[6,'B'] + x[6,'C'] == 9.6 * y[6]

# Column constraints
model += sum(x[i,'A'] for i in range(1, 7)) == 12
model += sum(x[i,'B'] for i in range(1, 7)) == 10
model += sum(x[i,'C'] for i in range(1, 7)) == 14

# Solve
model.solve()
