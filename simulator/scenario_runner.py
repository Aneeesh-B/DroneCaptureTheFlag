import os
os.system('cls')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import casadi as ca
import pandas as pd
import multiprocessing as mp
import optuna

"""
To use this solver, install the prerequisites using the following steps
1. Install Julia:
- wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.1-linux-x86_64.tar.gz
- tar zxvf julia-1.10.1-linux-x86_64.tar.gz
- export PATH="$PATH:/path/to/<Julia directory>/bin"
2. Install Julia packages:
- In the Julia REPL package manager: 
-- add PyCall
-- add PATHSolver@1.1.1 (side note, only version 1.1.1 works when called from pyjulia)
3. Install pyjulia:
- python3 -m pip install julia
"""


N_teams = 2
Team1_players = 3
Team2_players = 3
num_scenarios = 20
num_threads = min(mp.cpu_count(), 16)

# Define the game
X_min, X_max = -10, 10
Y_min, Y_max = -10, 10
Flag_Position = np.array([8.0, 0.0])
Capture_radius = 1.0
Max_Velocity = 2
d_tag = 0.5
Max_TurboVelocity = 4
Turbo_duration = 1.0
Turbo_cooldown = 3.0
dT = 0.1 # Time step
Tpred = 1 # Prediction time
Tsim = 5    # Simulation time
N = int(Tpred/dT) # Number of time steps


def generate_random_positions(team, num_players, x_min, x_max, y_min, y_max, flag_pos, capture_radius):
    positions = []
    while len(positions) < num_players:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        # Ensure players do not start in the flag area
        if np.linalg.norm(np.array([x, y]) - flag_pos) > capture_radius:
            positions.append(np.array([x, y]))
    return positions


def generate_scenarios(n_sets):
    """
    Returns a list of n_sets tuples:
      [ (players1_list, players2_list), … ]
    where each playersX_list is a list of np.arrays of starting coords.
    """
    scenarios = []
    for i in range(n_sets):
        # Team 1 on left half
        players1 = generate_random_positions(
            team=1,
            num_players=Team1_players,
            x_min=X_min,
            x_max=0,
            y_min=Y_min,
            y_max=Y_max,
            flag_pos=Flag_Position,
            capture_radius=Capture_radius
        )
        # Team 2 on right half
        players2 = generate_random_positions(
            team=2,
            num_players=Team2_players,
            x_min=0,
            x_max=X_max,
            y_min=Y_min,
            y_max=Y_max,
            flag_pos=Flag_Position,
            capture_radius=Capture_radius
        )
        scenarios.append((players1, players2))

    # Validation check
    for idx, pair in enumerate(scenarios):
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise ValueError(f"Scenario {idx} must be a tuple of (players1, players2), got: {pair}")
    return scenarios

# game_data = pd.DataFrame(columns=["scenario", "time", "player_id", "x", "y", "tagged", "velocity_x", "velocity_y"])

def store_game_state_to_csv(filename, scenario, time_step, x, player_taged, u_sol):
    global game_data
    
    data = []
    for i in range(len(x) // 2):  # Iterate over players
        player_id = i
        pos_x, pos_y = x[2*i], x[2*i+1]
        tagged = player_taged[i] if i < 3 else False
        vel_x, vel_y = u_sol[0, 2*i], u_sol[0, 2*i+1]
        data.append([scenario, time_step, player_id, pos_x, pos_y, tagged, vel_x, vel_y])
    
    temp_df = pd.DataFrame(data, columns=game_data.columns)
    game_data = pd.concat([game_data, temp_df], ignore_index=True)


def compute_scenario_score(player_positions, flag_position, capture_radius, max_score=100):
    distances = [np.linalg.norm(pos - flag_position) for pos in player_positions]
    min_distance = min(distances)
    
    if min_distance <= capture_radius:
        return max_score
    
    return max_score * (1 - min_distance / np.linalg.norm([X_max - X_min, Y_max - Y_min]))

def run_scenario(scenario, result_queue, alpha1=10, alpha2=0.1, vel_weight=0.5, avoidance=4.0, attraction=20, positions=None):

    alpha_1_1 = alpha1
    alpha_1_2 = alpha1
    alpha_1_3 = alpha1
    alpha_2_1 = alpha2
    alpha_2_2 = alpha2
    alpha_2_3 = alpha2

    scenario_data = []

    # Generate random starting positions for both teams
    if positions is None:
        Players1 = generate_random_positions(team=1, num_players=Team1_players, 
                                            x_min=X_min, x_max=0, y_min=Y_min, y_max=Y_max, 
                                            flag_pos=Flag_Position, capture_radius=Capture_radius)

        Players2 = generate_random_positions(team=2, num_players=Team2_players, 
                                            x_min=0, x_max=X_max, y_min=Y_min, y_max=Y_max, 
                                            flag_pos=Flag_Position, capture_radius=Capture_radius)
    else:
        Players1, Players2 = positions

    # Store in teams
    Teams = [Players1, Players2]
    # # Initialize the players states
    # Teams = []
    # Players1 = []
    # Players1.append(np.array([-5, 0]))
    # Players1.append(np.array([-5, -5]))
    # Players1.append(np.array([-5, 5]))
    # Teams.append(Players1)
    # Players2 = []
    # Players2.append(np.array([5, 0]))
    # Players2.append(np.array([5, -5]))
    # Players2.append(np.array([5, 5]))
    # Teams.append(Players2)

    syms_x = ca.SX.sym('x', N+1, 2*(Team1_players+Team2_players))
    syms_u = ca.SX.sym('u', N, 2*(Team1_players+Team2_players))
    syms_x0 = ca.SX.sym('x0', 2*(Team1_players+Team2_players))

    indx_1_1 = [0, 1]
    indx_1_2 = [2, 3]
    indx_1_3 = [4, 5]
    indx_2_1 = [6, 7]
    indx_2_2 = [8, 9]
    indx_2_3 = [10, 11]

    # def initialize_player(index, alpha, syms_x, syms_u, syms_x0, dT, N, Flag_Position, Max_Velocity, X_min, X_max, Y_min, Y_max, opponents):
    #     """Initializes the player state including cost function, dynamics, and constraints."""
    #     L = 0.5 * ca.sumsqr(syms_u[:, index]) + 0.5 * alpha * ca.sumsqr(syms_x[:, index] - np.ones((N+1, 2)) @ np.diag(Flag_Position))
        
    #     for k in range(N):
    #         for opp in opponents:
    #             L += 4.0 / ca.norm_2(syms_x[k+1, index] - syms_x[k+1, opp])
        
    #     h = [syms_x[0, index[0]] - syms_x0[index[0]], syms_x[0, index[1]] - syms_x0[index[1]]]
    #     for k in range(N):
    #         h.append(syms_x[k+1, index[0]] - syms_x[k, index[0]] - dT * syms_u[k, index[0]])
    #         h.append(syms_x[k+1, index[1]] - syms_x[k, index[1]] - dT * syms_u[k, index[1]])
    #     h = ca.vertcat(*h)
    #     mu = ca.SX.sym(f'mu_{index[0]}', h.shape[0])
    #     L += ca.dot(mu, h)
        
    #     g = []
    #     for k in range(N):
    #         g.append(Max_Velocity - ca.norm_2(syms_u[k, index]))
    #         g.append(syms_x[k+1, index[0]] - X_min)
    #         g.append(X_max - syms_x[k+1, index[0]])
    #         g.append(syms_x[k+1, index[1]] - Y_min)
    #         g.append(Y_max - syms_x[k+1, index[1]])
    #     g = ca.vertcat(*g)
    #     lambda_ = ca.SX.sym(f'lambda_{index[0]}', g.shape[0])
    #     L -= ca.dot(lambda_, g)
        
    #     return L, h, g, mu, lambda_

    # # Initialize players using the function
    # players = [
    #     (indx_1_1, alpha_1_1, [indx_2_1, indx_2_2]),
    #     (indx_1_2, alpha_1_2, [indx_2_1, indx_2_2]),
    #     (indx_1_3, alpha_1_3, [indx_2_1, indx_2_2]),
    #     (indx_2_1, alpha_2_1, [indx_1_1, indx_1_2, indx_1_3]),
    #     (indx_2_2, alpha_2_2, [indx_1_1, indx_1_2, indx_1_3]),
    #     (indx_2_3, alpha_2_3, [indx_1_1, indx_1_2, indx_1_3])
    # ]

    # L, Ch, Cgp, mus, lambdas = [], [], [], [], []
    # for index, alpha, opponents in players:
    #     L_i, h_i, g_i, mu_i, lambda_i = initialize_player(index, alpha, syms_x, syms_u, syms_x0, dT, N, Flag_Position, Max_Velocity, X_min, X_max, Y_min, Y_max, opponents)
    #     L.append(L_i)
    #     Ch.append(h_i)
    #     Cgp.append(g_i)
    #     mus.append(mu_i)
    #     lambdas.append(lambda_i)

    # Calculate the Lagrange for lead player in team 1 (1,1)
    L = []
    # Cost Function
    L.append(vel_weight*ca.sumsqr(syms_u[:,indx_1_1]) + 0.5*alpha_1_1*ca.sumsqr(syms_x[:,indx_1_1]-np.ones((N+1,2))@np.diag(Flag_Position)))
    for k in range(N):
        L[-1] += avoidance/ca.norm_2(syms_x[k+1,indx_1_1]-syms_x[k+1,indx_2_1])
        L[-1] += avoidance/ca.norm_2(syms_x[k+1,indx_1_1]-syms_x[k+1,indx_2_2])
        L[-1] += avoidance/ca.norm_2(syms_x[k+1,indx_1_1]-syms_x[k+1,indx_2_3])
    # Dynamics
    h1_1 = []
    h1_1.append(syms_x[0,indx_1_1[0]] - syms_x0[indx_1_1[0]])
    h1_1.append(syms_x[0,indx_1_1[1]] - syms_x0[indx_1_1[1]])
    for k in range(N):
        h1_1.append(syms_x[k+1,indx_1_1[0]] - syms_x[k,indx_1_1[0]] - dT*syms_u[k,indx_1_1[0]])
        h1_1.append(syms_x[k+1,indx_1_1[1]] - syms_x[k,indx_1_1[1]] - dT*syms_u[k,indx_1_1[1]])
    h1_1 = ca.vertcat(*h1_1)
    mu_1_1 = ca.SX.sym('mu_1_1', h1_1.shape[0])
    L[-1] += ca.dot(mu_1_1, h1_1)
    #Private constraints:
    g1_1 = []
    for k in range(N):
        # Max velocity
        g1_1.append(Max_Velocity - ca.norm_2(syms_u[k,indx_1_1]))
        # Track boundaries
        g1_1.append(syms_x[k+1,indx_1_1[0]] - X_min)
        g1_1.append(X_max - syms_x[k+1,indx_1_1[0]])
        g1_1.append(syms_x[k+1,indx_1_1[1]] - Y_min)
        g1_1.append(Y_max - syms_x[k+1,indx_1_1[1]])
    g1_1 = ca.vertcat(*g1_1)
    lambda_1_1 = ca.SX.sym('lambda_1_1', g1_1.shape[0])
    L[-1] -= ca.dot(lambda_1_1, g1_1)

    # Calculate the Lagrange for second player in team 1 (1,2)
    L.append(vel_weight*ca.sumsqr(syms_u[:,indx_1_2]) + 0.5*alpha_1_2*ca.sumsqr(syms_x[:,indx_1_2]-np.ones((N+1,2))@np.diag(Flag_Position)))
    # Cost Function
    for k in range(N):
        L[-1] += avoidance/ca.norm_2(syms_x[k+1,indx_1_2]-syms_x[k+1,indx_2_1])
        L[-1] += avoidance/ca.norm_2(syms_x[k+1,indx_1_2]-syms_x[k+1,indx_2_2])
        L[-1] += avoidance/ca.norm_2(syms_x[k+1,indx_1_2]-syms_x[k+1,indx_2_3])
    # Dynamics
    h1_2 = []
    h1_2.append(syms_x[0,indx_1_2[0]] - syms_x0[indx_1_2[0]])
    h1_2.append(syms_x[0,indx_1_2[1]] - syms_x0[indx_1_2[1]])
    for k in range(N):
        h1_2.append(syms_x[k+1,indx_1_2[0]] - syms_x[k,indx_1_2[0]] - dT*syms_u[k,indx_1_2[0]])
        h1_2.append(syms_x[k+1,indx_1_2[1]] - syms_x[k,indx_1_2[1]] - dT*syms_u[k,indx_1_2[1]])
    h1_2 = ca.vertcat(*h1_2)
    mu_1_2 = ca.SX.sym('mu_1_2', h1_2.shape[0])
    L[-1] += ca.dot(mu_1_2, h1_2)
    #Private constraints:
    g1_2 = []
    for k in range(N):
        # Max velocity
        g1_2.append(Max_Velocity - ca.norm_2(syms_u[k,indx_1_2]))
        # Track boundaries
        g1_2.append(syms_x[k+1,indx_1_2[0]] - X_min)
        g1_2.append(X_max - syms_x[k+1,indx_1_2[0]])
        g1_2.append(syms_x[k+1,indx_1_2[1]] - Y_min)
        g1_2.append(Y_max - syms_x[k+1,indx_1_2[1]])
    g1_2 = ca.vertcat(*g1_2)
    lambda_1_2 = ca.SX.sym('lambda_1_2', g1_2.shape[0])
    L[-1] -= ca.dot(lambda_1_2, g1_2)

    # Calculate the Lagrange for third player in team 1 (1,3)
    L.append(vel_weight*ca.sumsqr(syms_u[:,indx_1_3]) + 0.5*alpha_1_3*ca.sumsqr(syms_x[:,indx_1_3]-np.ones((N+1,2))@np.diag(Flag_Position)))
    # Cost Function
    for k in range(N):
        L[-1] += avoidance/ca.norm_2(syms_x[k+1,indx_1_3]-syms_x[k+1,indx_2_1])
        L[-1] += avoidance/ca.norm_2(syms_x[k+1,indx_1_3]-syms_x[k+1,indx_2_2])
        L[-1] += avoidance/ca.norm_2(syms_x[k+1,indx_1_3]-syms_x[k+1,indx_2_3])
    # Dynamics
    h1_3 = []
    h1_3.append(syms_x[0,indx_1_3[0]] - syms_x0[indx_1_3[0]])
    h1_3.append(syms_x[0,indx_1_3[1]] - syms_x0[indx_1_3[1]])
    for k in range(N):
        h1_3.append(syms_x[k+1,indx_1_3[0]] - syms_x[k,indx_1_3[0]] - dT*syms_u[k,indx_1_3[0]])
        h1_3.append(syms_x[k+1,indx_1_3[1]] - syms_x[k,indx_1_3[1]] - dT*syms_u[k,indx_1_3[1]])
    h1_3 = ca.vertcat(*h1_3)
    mu_1_3 = ca.SX.sym('mu_1_3', h1_3.shape[0])
    L[-1] += ca.dot(mu_1_3, h1_3)
    #Private constraints:
    g1_3 = []
    for k in range(N):
        # Max velocity
        g1_3.append(Max_Velocity - ca.norm_2(syms_u[k,indx_1_3]))
        # Track boundaries
        g1_3.append(syms_x[k+1,indx_1_3[0]] - X_min)
        g1_3.append(X_max - syms_x[k+1,indx_1_3[0]])
        g1_3.append(syms_x[k+1,indx_1_3[1]] - Y_min)
        g1_3.append(Y_max - syms_x[k+1,indx_1_3[1]])
    g1_3 = ca.vertcat(*g1_3)
    lambda_1_3 = ca.SX.sym('lambda_1_3', g1_3.shape[0])
    L[-1] -= ca.dot(lambda_1_3, g1_3)

    # Calculate the Lagrange for first player in team 2 (2,1)
    L.append(vel_weight*ca.sumsqr(syms_u[:,indx_2_1]) + 0.5*alpha_2_1*ca.sumsqr(syms_x[:,indx_2_1]-np.ones((N+1,2))@np.diag(Flag_Position)))
    # Cost Function
    for k in range(N):
        dx_1_1 = ca.norm_2(syms_x[k+1,indx_2_1]-syms_x[k+1,indx_1_1])**2
        dx_1_2 = ca.norm_2(syms_x[k+1,indx_2_1]-syms_x[k+1,indx_1_2])**2
        dx_1_3 = ca.norm_2(syms_x[k+1,indx_2_1]-syms_x[k+1,indx_1_3])**2
        L[-1] += attraction/np.pi*ca.atan(0.25*dx_1_1)
        L[-1] += attraction/np.pi*ca.atan(0.25*dx_1_2)
        L[-1] += attraction/np.pi*ca.atan(0.25*dx_1_3)
    # Dynamics
    h2_1 = []
    h2_1.append(syms_x[0,indx_2_1[0]] - syms_x0[indx_2_1[0]])
    h2_1.append(syms_x[0,indx_2_1[1]] - syms_x0[indx_2_1[1]])
    for k in range(N):
        h2_1.append(syms_x[k+1,indx_2_1[0]] - syms_x[k,indx_2_1[0]] - dT*syms_u[k,indx_2_1[0]])
        h2_1.append(syms_x[k+1,indx_2_1[1]] - syms_x[k,indx_2_1[1]] - dT*syms_u[k,indx_2_1[1]])
    h2_1 = ca.vertcat(*h2_1)
    mu_2_1 = ca.SX.sym('mu_2_1', h2_1.shape[0])
    L[-1] += ca.dot(mu_2_1, h2_1)
    #Private constraints:
    g2_1 = []
    for k in range(N):
        # Max velocity
        g2_1.append(Max_Velocity - ca.norm_2(syms_u[k,indx_2_1]))
        # Track boundaries
        g2_1.append(syms_x[k+1,indx_2_1[0]] - X_min)
        g2_1.append(X_max - syms_x[k+1,indx_2_1[0]])
        g2_1.append(syms_x[k+1,indx_2_1[1]] - Y_min)
        g2_1.append(Y_max - syms_x[k+1,indx_2_1[1]])
        # No Entry Zone around the flag
        g2_1.append(ca.norm_2(syms_x[k+1,indx_2_1]-Flag_Position.reshape(1,-1)) - Capture_radius)
    g2_1 = ca.vertcat(*g2_1)
    lambda_2_1 = ca.SX.sym('lambda_2_1', g2_1.shape[0])
    L[-1] -= ca.dot(lambda_2_1, g2_1)

    # Calculate the Lagrange for second player in team 2 (2,2)
    L.append(vel_weight*ca.sumsqr(syms_u[:,indx_2_2]) + 0.5*alpha_2_2*ca.sumsqr(syms_x[:,indx_2_2]-np.ones((N+1,2))@np.diag(Flag_Position)))
    # Cost Function
    for k in range(N):
        dx_1_1 = ca.norm_2(syms_x[k+1,indx_2_2]-syms_x[k+1,indx_1_1])**2
        dx_1_2 = ca.norm_2(syms_x[k+1,indx_2_2]-syms_x[k+1,indx_1_2])**2
        dx_1_3 = ca.norm_2(syms_x[k+1,indx_2_2]-syms_x[k+1,indx_1_3])**2
        L[-1] += attraction/np.pi*ca.atan(0.25*dx_1_1)
        L[-1] += attraction/np.pi*ca.atan(0.25*dx_1_2)
        L[-1] += attraction/np.pi*ca.atan(0.25*dx_1_3)
    # Dynamics
    h2_2 = []
    h2_2.append(syms_x[0,indx_2_2[0]] - syms_x0[indx_2_2[0]])
    h2_2.append(syms_x[0,indx_2_2[1]] - syms_x0[indx_2_2[1]])
    for k in range(N):
        h2_2.append(syms_x[k+1,indx_2_2[0]] - syms_x[k,indx_2_2[0]] - dT*syms_u[k,indx_2_2[0]])
        h2_2.append(syms_x[k+1,indx_2_2[1]] - syms_x[k,indx_2_2[1]] - dT*syms_u[k,indx_2_2[1]])
    h2_2 = ca.vertcat(*h2_2)
    mu_2_2 = ca.SX.sym('mu_2_2', h2_2.shape[0])
    L[-1] += ca.dot(mu_2_2, h2_2)
    #Private constraints:
    g2_2 = []
    for k in range(N):
        # Max velocity
        g2_2.append(Max_Velocity - ca.norm_2(syms_u[k,indx_2_2]))
        # Track boundaries
        g2_2.append(syms_x[k+1,indx_2_2[0]] - X_min)
        g2_2.append(X_max - syms_x[k+1,indx_2_2[0]])
        g2_2.append(syms_x[k+1,indx_2_2[1]] - Y_min)
        g2_2.append(Y_max - syms_x[k+1,indx_2_2[1]])
        # No Entry Zone around the flag
        g2_2.append(ca.norm_2(syms_x[k+1,indx_2_2]-Flag_Position.reshape(1,-1)) - Capture_radius)
    g2_2 = ca.vertcat(*g2_2)
    lambda_2_2 = ca.SX.sym('lambda_2_2', g2_2.shape[0])
    L[-1] -= ca.dot(lambda_2_2, g2_2)

    # Calculate the Lagrange for third player in team 2 (2,3)
    L.append(vel_weight*ca.sumsqr(syms_u[:,indx_2_3]) + 0.5*alpha_2_3*ca.sumsqr(syms_x[:,indx_2_3]-np.ones((N+1,2))@np.diag(Flag_Position)))
    # Cost Function
    for k in range(N):
        dx_1_1 = ca.norm_2(syms_x[k+1,indx_2_2]-syms_x[k+1,indx_1_1])**2
        dx_1_2 = ca.norm_2(syms_x[k+1,indx_2_2]-syms_x[k+1,indx_1_2])**2
        dx_1_3 = ca.norm_2(syms_x[k+1,indx_2_2]-syms_x[k+1,indx_1_3])**2
        L[-1] += attraction/np.pi*ca.atan(0.25*dx_1_1)
        L[-1] += attraction/np.pi*ca.atan(0.25*dx_1_2)
        L[-1] += attraction/np.pi*ca.atan(0.25*dx_1_3)
    # Dynamics
    h2_3 = []
    h2_3.append(syms_x[0,indx_2_3[0]] - syms_x0[indx_2_3[0]])
    h2_3.append(syms_x[0,indx_2_3[1]] - syms_x0[indx_2_3[1]])
    for k in range(N):
        h2_3.append(syms_x[k+1,indx_2_3[0]] - syms_x[k,indx_2_3[0]] - dT*syms_u[k,indx_2_3[0]])
        h2_3.append(syms_x[k+1,indx_2_3[1]] - syms_x[k,indx_2_3[1]] - dT*syms_u[k,indx_2_3[1]])
    h2_3 = ca.vertcat(*h2_3)
    mu_2_3 = ca.SX.sym('mu_2_2', h2_3.shape[0])
    L[-1] += ca.dot(mu_2_3, h2_3)
    #Private constraints:
    g2_3 = []
    for k in range(N):
        # Max velocity
        g2_3.append(Max_Velocity - ca.norm_2(syms_u[k,indx_2_3]))
        # Track boundaries
        g2_3.append(syms_x[k+1,indx_2_3[0]] - X_min)
        g2_3.append(X_max - syms_x[k+1,indx_2_3[0]])
        g2_3.append(syms_x[k+1,indx_2_3[1]] - Y_min)
        g2_3.append(Y_max - syms_x[k+1,indx_2_3[1]])
        # No Entry Zone around the flag
        g2_3.append(ca.norm_2(syms_x[k+1,indx_2_3]-Flag_Position.reshape(1,-1)) - Capture_radius)
    g2_3 = ca.vertcat(*g2_3)
    lambda_2_3 = ca.SX.sym('lambda_2_3', g2_3.shape[0])
    L[-1] -= ca.dot(lambda_2_3, g2_3)

    Z_vec, Z_len = [], []
    for j in range(2*(Team1_players+Team2_players)):
        for k in range(N+1):
            Z_vec.append(syms_x[k,j])
    Z_len.append(len(Z_vec))
    for j in range(2*(Team1_players+Team2_players)):
        for k in range(N):
            Z_vec.append(syms_u[k,j])
    Z_len.append(len(Z_vec)-Z_len[-1])
    for j in range(h1_1.shape[0]):
        Z_vec.append(mu_1_1[j])
    for j in range(h1_2.shape[0]):
        Z_vec.append(mu_1_2[j])
    for j in range(h1_3.shape[0]):
        Z_vec.append(mu_1_3[j])
    for j in range(h2_1.shape[0]):
        Z_vec.append(mu_2_1[j])
    for j in range(h2_2.shape[0]):
        Z_vec.append(mu_2_2[j])
    for j in range(h2_3.shape[0]):
        Z_vec.append(mu_2_3[j])
    Z_len.append(len(Z_vec)-np.sum(Z_len))
    for j in range(g1_1.shape[0]):
        Z_vec.append(lambda_1_1[j])
    for j in range(g1_2.shape[0]):
        Z_vec.append(lambda_1_2[j])
    for j in range(g1_3.shape[0]):
        Z_vec.append(lambda_1_3[j])
    for j in range(g2_1.shape[0]):
        Z_vec.append(lambda_2_1[j])
    for j in range(g2_2.shape[0]):
        Z_vec.append(lambda_2_2[j])
    for j in range(g2_3.shape[0]):
        Z_vec.append(lambda_2_3[j])
    Z_len.append(len(Z_vec)-np.sum(Z_len))
    Z_vec = ca.vertcat(*Z_vec)

    # Z_vec = ca.vertcat(*[syms_x[:, i] for i in range(2*(Team1_players+Team2_players))],
    #                    *[syms_u[:, i] for i in range(2*(Team1_players+Team2_players))],
    #                    *mus, *lambdas)

    # Z_len = [
    #     (N+1) * 2 * (Team1_players + Team2_players),
    #     N * 2 * (Team1_players + Team2_players),
    #     sum(mu.shape[0] for mu in mus),
    #     sum(lambda_.shape[0] for lambda_ in lambdas)
    # ]
        

    N_x_per_player = 2*(N+1)
    N_u_per_player = 2*N
    Total_N_x = N_x_per_player*(Team1_players+Team2_players)
    Dxu_L, Ch, Cgp = [], [], []
    for i in range(Team1_players + Team2_players):
        cur_x = Z_vec[N_x_per_player*i:N_x_per_player*(i+1)]
        cur_u = Z_vec[Total_N_x + N_u_per_player*i:Total_N_x + N_u_per_player*(i+1)]
        xu = ca.vertcat(cur_x, cur_u)
        Dxu_L.append(ca.jacobian(L[i], xu).T)
    Ch.append(h1_1)
    Cgp.append(g1_1)
    Ch.append(h1_2)
    Cgp.append(g1_2)
    Ch.append(h1_3)
    Cgp.append(g1_3)
    Ch.append(h2_1)
    Cgp.append(g2_1)
    Ch.append(h2_2)
    Cgp.append(g2_2)
    Ch.append(h2_3)
    Cgp.append(g2_3)

    # Define the F and J functions
    F = ca.vertcat(*Dxu_L, *Ch, *Cgp)
    F_fun = ca.Function('F', [Z_vec, syms_x0], [F])
    J = ca.jacobian(F, Z_vec)
    J_fun = ca.Function('J', [Z_vec, syms_x0], [J])

    ub = np.inf*np.ones(Z_vec.shape[0])
    lb = np.concatenate((-np.inf*np.ones(Z_len[0]+Z_len[1]+Z_len[2]), np.zeros(Z_len[3])))

    tsim_vec = np.linspace(0, Tsim, int(Tsim/dT)+1)

    x = np.array([Players1[0][0], Players1[0][1], Players1[1][0], Players1[1][1], Players1[2][0], Players1[2][1], Players2[0][0], Players2[0][1], Players2[1][0], Players2[1][1], Players2[2][0], Players2[2][1]])

    z0 = np.zeros(Z_vec.shape[0]) + 1e-8
    for i in range(N+1):
        for j in range(Team1_players + Team2_players):
            z0[i + j * (N+1)] = Teams[j//Team1_players][j % Team1_players][0]
            z0[i + j * (N+1) + (N+1) * (Team1_players+Team2_players)] = Teams[j//Team1_players][j % Team1_players][1]

    # Set up plotting
    theta = np.linspace(0, 2*np.pi, 100)
    x_radius = np.cos(theta)
    y_radius = np.sin(theta)
    plt.ion()
    f, (ax_xy, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(10,10))

    ax_xy.set_xlabel('x [m]')
    ax_xy.set_ylabel('y [m]')
    ax_xy.set_xlim([X_min-1, X_max+1])
    ax_xy.set_ylim([Y_min-1, Y_max+1])
    ax_xy.plot([X_min, X_min, X_max, X_max, X_min], [Y_min, Y_max, Y_max, Y_min, Y_min], 'k-')
    ax_xy.plot([0,0], [Y_min, Y_max], 'k--')
    ax_xy.plot(Flag_Position[0], Flag_Position[1], 'kx', label='Flag')
    ax_xy.plot(Capture_radius*x_radius+Capture_radius*Flag_Position[0], y_radius+Flag_Position[1], 'k--')
    ax_xy.plot(-Flag_Position[0], Flag_Position[1], 'kx')
    ax_xy.plot(Capture_radius*x_radius-Flag_Position[0], Capture_radius*y_radius+Flag_Position[1], 'k--')

    plot_p1_pred = ax_xy.plot([], [], 'r', label='(Team 1)')[0]
    plot_p2_pred = ax_xy.plot([], [], 'r')[0]
    plot_p3_pred = ax_xy.plot([], [], 'r')[0]
    plot_p4_pred = ax_xy.plot([], [], 'g', label='(Team 2)')[0]
    plot_p5_pred = ax_xy.plot([], [], 'g')[0]
    plot_p6_pred = ax_xy.plot([], [], 'g')[0]
    plot_p1_cur = ax_xy.plot([], [], 'ro')[0]
    plot_p2_cur = ax_xy.plot([], [], 'ro')[0]
    plot_p3_cur = ax_xy.plot([], [], 'ro')[0]
    plot_p4_cur = ax_xy.plot([], [], 'go')[0]
    plot_p5_cur = ax_xy.plot([], [], 'go')[0]
    plot_p6_cur = ax_xy.plot([], [], 'go')[0]
    plot_p1_tag = ax_xy.plot([], [], 'r:')[0]
    plot_p2_tag = ax_xy.plot([], [], 'r:')[0]
    plot_p3_tag = ax_xy.plot([], [], 'r:')[0]
    plot_p4_tag = ax_xy.plot([], [], 'g:')[0]
    plot_p5_tag = ax_xy.plot([], [], 'g:')[0]
    plot_p6_tag = ax_xy.plot([], [], 'g:')[0]
    plot_p1_his = ax_xy.plot([], [], 'r--')[0]
    plot_p2_his = ax_xy.plot([], [], 'r--')[0]
    plot_p3_his = ax_xy.plot([], [], 'r--')[0]
    plot_p4_his = ax_xy.plot([], [], 'g--')[0]
    plot_p5_his = ax_xy.plot([], [], 'g--')[0]
    plot_p6_his = ax_xy.plot([], [], 'g--')[0]

    plot_p1_vel = ax2.plot([], [], 'r-', label='Player 1 (Team 1)')[0]
    plot_p2_vel = ax2.plot([], [], 'r-')[0]
    plot_p3_vel = ax2.plot([], [], 'r-')[0]
    plot_p4_vel = ax2.plot([], [], 'g-', label='Player 4 (Team 2)')[0]
    plot_p5_vel = ax2.plot([], [], 'g-')[0]
    plot_p6_vel = ax2.plot([], [], 'g-')[0]
    ax2.grid()
    ax2.legend()


    # Load Julia and PATHSolver
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    jl.using("PyCall")
    jl.using("PATHSolver")
        

    x_history = []
    u_history = []
    x_history.append(x)
    player1_taged = False
    player2_taged = False
    player3_taged = False
    MakeMovie = True
    Frames = []

    for t in tsim_vec:

        

        x_game = x.copy()
        if player1_taged:
            x_game[indx_1_1] = [X_min, 0]
        if player2_taged:
            x_game[indx_1_2] = [X_min, 0]
        if player3_taged:
            x_game[indx_1_3] = [X_min, 0]
        
        Main.z0 = z0
        Main.ub = ub
        Main.lb = lb

        Main.nnz = J_fun.numel_out(0)

        Main.F_py = lambda z: np.array(F_fun(z, x_game)).squeeze()
        Main.J_py = lambda z: np.array(J_fun(z, x_game))
        Main.tol = 1e-3


        F_def = """
                function F(n::Cint, z::Vector{Cdouble}, f::Vector{Cdouble})
                    @assert n == length(z)
                    f .= F_py(z)
                    return Cint(0)
                end
                return(F)
                """
        Main.F = jl.eval(F_def)

        J_def = """
                function J(
                    n::Cint,
                    nnz::Cint,
                    z::Vector{Cdouble},
                    col::Vector{Cint},
                    len::Vector{Cint},
                    row::Vector{Cint},
                    data::Vector{Cdouble},
                )
                    @assert n == length(z)  == length(col) == length(len)
                    @assert nnz == length(row) == length(data)
                    j = Array{Float64}(undef, n, n)
                    j .= J_py(z)
                    i = 1
                    for c in 1:n
                        col[c], len[c] = i, 0
                        for r in 1:n
                            # if !iszero(j[r, c])
                            #     row[i], data[i] = r, j[r, c]
                            #     len[c] += 1
                            #     i += 1
                            # end
                            row[i], data[i] = r, j[r, c]
                            len[c] += 1
                            i += 1
                        end
                    end
                    return Cint(0)
                end
                return(J)
                """
        Main.J = jl.eval(J_def)

        
        output = 'no' # 'no' for no output
        nms = 'yes'

        solve = f"""
        PATHSolver.c_api_License_SetString("2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0")
        status, z, info = PATHSolver.solve_mcp(F, 
                                                J,
                                                lb,
                                                ub,
                                                z0,
                                                nnz=nnz,
                                                output="{output}",
                                                convergence_tolerance=tol,
                                                nms="{nms}",
                                                crash_nbchange_limit=5,
                                                major_iteration_limit=100000,
                                                minor_iteration_limit=100000,
                                                cumulative_iteration_limit=100000,
                                                restart_limit=100)
        success = status == PATHSolver.MCP_Solved

        return z, success, info.residual, status
        """
        z, success, res, status = jl.eval(solve)
        z0 = z

        print("success: ",success, ", solution residual: ",res, ", status: ",status)

        x_sol = np.zeros((N+1, 2*(Team1_players+Team2_players)))
        u_sol = np.zeros((N, 2*(Team1_players+Team2_players)))
        j = 0
        for i in range(2*(Team1_players+Team2_players)):
            for k in range(N+1):
                x_sol[k,i] = z[j]
                j += 1
        for i in range(2*(Team1_players+Team2_players)):
            for k in range(N):
                u_sol[k,i] = z[j]
                j += 1
        mu_1_1_sol = z[j:j+h1_1.shape[0]]
        j += h1_1.shape[0]
        mu_1_2_sol = z[j:j+h1_2.shape[0]]
        j += h1_2.shape[0]
        mu_1_3_sol = z[j:j+h1_3.shape[0]]
        j += h1_3.shape[0]
        mu_2_1_sol = z[j:j+h2_1.shape[0]]
        j += h2_1.shape[0]
        mu_2_2_sol = z[j:j+h2_2.shape[0]]
        j += h2_2.shape[0]
        mu_2_3_sol = z[j:j+h2_3.shape[0]]
        j += h2_3.shape[0]    
        lambda_1_1_sol = z[j:j+g1_1.shape[0]]
        j += g1_1.shape[0]
        lambda_1_2_sol = z[j:j+g1_2.shape[0]]
        j += g1_2.shape[0]
        lambda_1_3_sol = z[j:j+g1_3.shape[0]]
        j += g1_3.shape[0]
        lambda_2_1_sol = z[j:j+g2_1.shape[0]]
        j += g2_1.shape[0]
        lambda_2_2_sol = z[j:j+g2_2.shape[0]]
        j += g2_2.shape[0]
        lambda_2_3_sol = z[j:j+g2_3.shape[0]]
        j += g2_3.shape[0]

        if player1_taged:
            u_sol[0,indx_1_1] = [-Max_Velocity, 0]
        if player2_taged:
            u_sol[0,indx_1_2] = [-Max_Velocity, 0]
        if player3_taged:
            u_sol[0,indx_1_3] = [-Max_Velocity, 0]

        x = x + dT*u_sol[0,:]
        if np.min([np.linalg.norm(x[indx_2_1]-x[indx_1_1]), np.linalg.norm(x[indx_2_2]-x[indx_1_1]), np.linalg.norm(x[indx_2_3]-x[indx_1_1])]) < d_tag:
            print("Player 1 tagged!")
            player1_taged = True
        if np.min([np.linalg.norm(x[indx_2_1]-x[indx_1_2]), np.linalg.norm(x[indx_2_2]-x[indx_1_2]) ,np.linalg.norm(x[indx_2_3]-x[indx_1_2])]) < d_tag:
            print("Player 2 tagged!")
            player2_taged = True
        if np.min([np.linalg.norm(x[indx_2_1]-x[indx_1_3]), np.linalg.norm(x[indx_2_2]-x[indx_1_3]) ,np.linalg.norm(x[indx_2_3]-x[indx_1_3])]) < d_tag:
            print("Player 3 tagged!")
            player3_taged = True

        if player1_taged and x[indx_1_1[0]] < 0:
            player1_taged = False
        if player2_taged and x[indx_1_2[0]] < 0:
            player2_taged = False
        if player3_taged and x[indx_1_3[0]] < 0:
            player3_taged = False
        
        x_history.append(x)
        u_history.append(u_sol[0,:])

        # store_game_state_to_csv(csv_filename, scenario, t, x, [player1_taged, player2_taged, player3_taged], u_sol)
        
        player_taged = [player1_taged, player2_taged, player3_taged]

# def store_game_state_to_csv(filename, scenario, time_step, x, player_taged, u_sol):
#     global game_data
    
#     data = []
#     for i in range(len(x) // 2):  # Iterate over players
#         player_id = i
#         pos_x, pos_y = x[2*i], x[2*i+1]
#         tagged = player_taged[i] if i < 3 else False
#         vel_x, vel_y = u_sol[0, 2*i], u_sol[0, 2*i+1]
#         data.append([scenario, time_step, player_id, pos_x, pos_y, tagged, vel_x, vel_y])
    
#     temp_df = pd.DataFrame(data, columns=game_data.columns)
#     game_data = pd.concat([game_data, temp_df], ignore_index=True)

        if not player1_taged:
            plot_p1_pred.set_data(x_sol[:,0], x_sol[:,1])
        else:
            plot_p1_pred.set_data([x[0], x[0]], [x[1], x[1]])
        if not player2_taged:
            plot_p2_pred.set_data(x_sol[:,2], x_sol[:,3])
        else:
            plot_p2_pred.set_data([x[2], x[2]], [x[3], x[3]])
        if not player3_taged:
            plot_p3_pred.set_data(x_sol[:,4], x_sol[:,5])
        else:
            plot_p3_pred.set_data([x[4], x[4]], [x[5], x[5]])
        plot_p4_pred.set_data(x_sol[:,6], x_sol[:,7])
        plot_p5_pred.set_data(x_sol[:,8], x_sol[:,9])
        plot_p6_pred.set_data(x_sol[:,10], x_sol[:,11])

        plot_p1_cur.set_data([x[0], x[0]], [x[1], x[1]])
        plot_p2_cur.set_data([x[2], x[2]], [x[3], x[3]])
        plot_p3_cur.set_data([x[4], x[4]], [x[5], x[5]])
        plot_p4_cur.set_data([x[6], x[6]], [x[7], x[7]])
        plot_p5_cur.set_data([x[8], x[8]], [x[9], x[    9]])
        plot_p6_cur.set_data([x[10], x[10]], [x[11], x[11]])
        if x[0]<0:
            plot_p1_tag.set_data(x[0]+d_tag*x_radius, x[1]+d_tag*y_radius)
        else:
            plot_p1_tag.set_data([],[])
        if x[2]<0:
            plot_p2_tag.set_data(x[2]+d_tag*x_radius, x[3]+d_tag*y_radius)
        else:
            plot_p2_tag.set_data([],[])
        if x[4]<0:
            plot_p3_tag.set_data(x[4]+d_tag*x_radius, x[5]+d_tag*y_radius)
        else:
            plot_p3_tag.set_data([],[])
        if x[6]>0:
            plot_p4_tag.set_data(x[6]+d_tag*x_radius, x[7]+d_tag*y_radius)
        else:
            plot_p4_tag.set_data([],[])
        if x[8]>0:
            plot_p5_tag.set_data(x[8]+d_tag*x_radius, x[9]+d_tag*y_radius)
        else:
            plot_p5_tag.set_data([],[])
        if x[10]>0:
            plot_p6_tag.set_data(x[10]+d_tag*x_radius, x[11]+d_tag*y_radius)
        else:
            plot_p6_tag.set_data([],[])
        x_hist = np.array(x_history)
        plot_p1_his.set_data(x_hist[:,0], x_hist[:,1])
        plot_p2_his.set_data(x_hist[:,2], x_hist[:,3])
        plot_p3_his.set_data(x_hist[:,4], x_hist[:,5])
        plot_p4_his.set_data(x_hist[:,6], x_hist[:,7])
        plot_p5_his.set_data(x_hist[:,8], x_hist[:,9])
        plot_p6_his.set_data(x_hist[:,10], x_hist[:,11])

        ax_xy.grid('on')
        ax_xy.legend()
        ax_xy.set_aspect('equal')

        t_vec = np.linspace(t, t+Tpred, N+1)
        plot_p1_vel.set_data(t_vec[:-1], np.linalg.norm(u_sol[:,indx_1_1],axis=1))
        plot_p2_vel.set_data(t_vec[:-1], np.linalg.norm(u_sol[:,indx_1_2],axis=1))
        plot_p3_vel.set_data(t_vec[:-1], np.linalg.norm(u_sol[:,indx_1_3],axis=1))
        plot_p4_vel.set_data(t_vec[:-1], np.linalg.norm(u_sol[:,indx_2_1],axis=1))
        plot_p5_vel.set_data(t_vec[:-1], np.linalg.norm(u_sol[:,indx_2_2],axis=1))
        plot_p6_vel.set_data(t_vec[:-1], np.linalg.norm(u_sol[:,indx_2_3],axis=1))
        
        ax2.grid('on')
        ax2.set_xlim([t, t+Tpred])
        ax2.set_ylim([0, Max_Velocity+1])
        ax2.legend()
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Velocity [m/s]')
        f.canvas.draw()
        f.canvas.flush_events()

        if MakeMovie:
            # image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
            # print(f"Image size: {image.size}")
            # print(f"Expected shape: {f.canvas.get_width_height()[::-1] + (3,)}")
            # image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
            # Frames.append(image)

            width, height = f.canvas.get_width_height()
            image = np.frombuffer(f.canvas.tostring_argb(), dtype='uint8')
            image = image[:width * height * 3]  # Trim extra data
            image = image.reshape((height, width, 3))
            
        if np.linalg.norm(x[indx_1_1]-Flag_Position) < Capture_radius or \
        np.linalg.norm(x[indx_1_2]-Flag_Position) < Capture_radius or \
        np.linalg.norm(x[indx_1_3]-Flag_Position) < Capture_radius:
            print("Flag captured!")
            break

        for i in range(len(x) // 2):
            scenario_data.append({
                "scenario": scenario,
                "time": t,
                "player_id": i,
                "x": x[2*i],
                "y": x[2*i+1],
                "tagged": player_taged[i] if i < Team1_players else False, 
                "velocity_x": u_sol[0, 2*i],
                "velocity_y": u_sol[0, 2*i+1]
            })
        

    df = pd.DataFrame(scenario_data)
    
    final_time = df["time"].max()
    final_positions = df[df["time"] == final_time][["x", "y"]].values
    final_positions_np = [np.array(pos) for pos in final_positions]
    scenario_score = compute_scenario_score(final_positions_np, Flag_Position, Capture_radius)

    result_queue.put(scenario_data)

    f.clf()
    plt.close(f)

    return scenario_score