function config = algorithm_config()
    % ALGORITHM_CONFIG - Centralized configuration for optimization algorithms
    % 
    % This function provides centralized configuration for all optimization
    % algorithms used in 5G network slicing research. Parameters are based
    % on research literature and empirical studies.
    %
    % Returns:
    %   config: Structure containing all algorithm configuration parameters
    %
    % References:
    % - Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization
    % - Watkins, C. J., & Dayan, P. (1992). Q-learning
    % - Recent 5G network slicing optimization literature
    
    %% PSO (Particle Swarm Optimization) Parameters
    config.pso = struct();
    
    % Population parameters
    config.pso.num_particles = 30;        % Population size
    config.pso.num_iterations = 100;      % Maximum iterations
    config.pso.convergence_threshold = 1e-6;  % Convergence tolerance
    
    % PSO coefficients (based on Clerc's constriction method)
    config.pso.c1 = 1.5;                  % Cognitive learning factor
    config.pso.c2 = 1.5;                  % Social learning factor
    config.pso.w = 0.7;                   % Inertia weight
    config.pso.w_min = 0.4;               % Minimum inertia weight
    config.pso.w_max = 0.9;               % Maximum inertia weight
    
    % Velocity constraints
    config.pso.velocity_max = 10;         % Maximum velocity
    config.pso.velocity_min = -10;        % Minimum velocity
    
    % Adaptive parameters
    config.pso.adaptive_inertia = true;   % Use adaptive inertia weight
    config.pso.adaptive_coefficients = false; % Use adaptive coefficients
    
    %% Q-Learning Parameters
    config.qlearning = struct();
    
    % Learning parameters
    config.qlearning.alpha = 0.1;         % Learning rate
    config.qlearning.gamma = 0.9;         % Discount factor
    config.qlearning.epsilon = 0.1;       % Exploration rate
    config.qlearning.epsilon_min = 0.01;  % Minimum exploration rate
    config.qlearning.epsilon_decay = 0.995; % Exploration decay rate
    
    % Training parameters
    config.qlearning.num_episodes = 1000; % Number of training episodes
    config.qlearning.max_steps_per_episode = 100; % Max steps per episode
    config.qlearning.num_actions = 10;    % Action space size
    
    % Q-table initialization
    config.qlearning.initial_q_value = 0; % Initial Q-values
    config.qlearning.optimistic_init = false; % Optimistic initialization
    
    % Convergence parameters
    config.qlearning.convergence_threshold = 1e-4;
    config.qlearning.min_episodes = 100;  % Minimum episodes before convergence check
    
    %% HPQL (Hybrid PSO + Q-Learning) Parameters
    config.hpql = struct();
    
    % Hybrid strategy parameters
    config.hpql.pso_weight = 0.6;         % Weight for PSO component
    config.hpql.ql_weight = 0.4;          % Weight for Q-Learning component
    config.hpql.beta = 0.1;               % Hybrid coupling parameter
    
    % Integration parameters
    config.hpql.use_pso_seeding = true;   % Use PSO results to seed Q-table
    config.hpql.iterative_refinement = true; % Iterative refinement between algorithms
    
    %% Priority-Aware Resource Allocation Parameters
    config.priority = struct();
    
    % Priority calculation method
    config.priority.method = 'weighted_efficiency'; % 'weighted_efficiency', 'pure_priority'
    config.priority.efficiency_weight = 0.7;        % Weight for efficiency vs priority
    config.priority.priority_weight = 0.3;          % Weight for pure priority
    
    % Reservation strategy
    config.priority.use_reservation = true;         % Use resource reservation
    config.priority.dynamic_reservation = false;    % Dynamic reservation adjustment
    
    %% General Optimization Parameters
    config.general = struct();
    
    % Random seed for reproducibility
    config.general.random_seed = 42;      % Fixed seed for reproducible results
    
    % Optimization constraints
    config.general.enforce_constraints = true;      % Enforce allocation constraints
    config.general.allow_oversubscription = false;  % Allow total allocation > spectrum
    
    % Performance evaluation
    config.general.evaluation_frequency = 10;       % Evaluate every N iterations
    config.general.save_intermediate_results = true; % Save intermediate results
    
    % Logging and output
    config.general.verbose = true;        % Verbose output
    config.general.save_plots = true;     % Save generated plots
    config.general.plot_frequency = 50;   % Plot every N iterations
    
    %% Validation
    validate_algorithm_config(config);
end

function validate_algorithm_config(config)
    % VALIDATE_ALGORITHM_CONFIG - Validates algorithm configuration parameters
    
    % PSO validation
    pso = config.pso;
    assert(pso.num_particles > 0, 'PSO: Number of particles must be positive');
    assert(pso.num_iterations > 0, 'PSO: Number of iterations must be positive');
    assert(pso.c1 > 0 && pso.c2 > 0, 'PSO: Learning factors must be positive');
    assert(pso.w >= pso.w_min && pso.w <= pso.w_max, 'PSO: Inertia weight out of range');
    assert(pso.velocity_max > pso.velocity_min, 'PSO: Invalid velocity range');
    
    % Q-Learning validation
    ql = config.qlearning;
    assert(ql.alpha > 0 && ql.alpha <= 1, 'QL: Learning rate must be in (0,1]');
    assert(ql.gamma >= 0 && ql.gamma <= 1, 'QL: Discount factor must be in [0,1]');
    assert(ql.epsilon >= 0 && ql.epsilon <= 1, 'QL: Exploration rate must be in [0,1]');
    assert(ql.num_episodes > 0, 'QL: Number of episodes must be positive');
    assert(ql.num_actions > 0, 'QL: Number of actions must be positive');
    
    % HPQL validation
    hpql = config.hpql;
    assert(hpql.pso_weight >= 0 && hpql.pso_weight <= 1, 'HPQL: PSO weight must be in [0,1]');
    assert(hpql.ql_weight >= 0 && hpql.ql_weight <= 1, 'HPQL: QL weight must be in [0,1]');
    assert(abs(hpql.pso_weight + hpql.ql_weight - 1.0) < 1e-6, 'HPQL: Weights must sum to 1.0');
    
    % Priority validation
    priority = config.priority;
    assert(priority.efficiency_weight >= 0 && priority.efficiency_weight <= 1, ...
           'Priority: Efficiency weight must be in [0,1]');
    assert(priority.priority_weight >= 0 && priority.priority_weight <= 1, ...
           'Priority: Priority weight must be in [0,1]');
    assert(abs(priority.efficiency_weight + priority.priority_weight - 1.0) < 1e-6, ...
           'Priority: Weights must sum to 1.0');
    
    % General validation
    general = config.general;
    assert(general.evaluation_frequency > 0, 'General: Evaluation frequency must be positive');
    assert(general.plot_frequency > 0, 'General: Plot frequency must be positive');
    
    fprintf('✓ Algorithm configuration validated successfully\n');
end 