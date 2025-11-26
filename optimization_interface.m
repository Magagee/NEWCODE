function [results, metrics] = optimization_interface(config, algorithm_type, varargin)
    % OPTIMIZATION_INTERFACE - Standardized interface for optimization algorithms
    %
    % This function provides a unified interface for all optimization algorithms
    % in the 5G network slicing research framework. It ensures consistent
    % function signatures, proper error handling, and standardized output
    % formats for research reproducibility.
    %
    % Inputs:
    %   config: Configuration structure from master_config()
    %   algorithm_type: String specifying the algorithm ('PSO', 'QL', 'HPQL', 'Priority')
    %   varargin: Additional algorithm-specific parameters
    %
    % Returns:
    %   results: Structure containing algorithm results
    %   metrics: Structure containing performance metrics
    %
    % Usage:
    %   [results, metrics] = optimization_interface(config, 'PSO');
    %   [results, metrics] = optimization_interface(config, 'QL', 'custom_param', value);
    %
    % Author: [Your Name]
    % Date: [Current Date]
    % Institution: [Your Institution]
    % Research: 5G Network Slicing Resource Allocation Optimization
    
    %% Input Validation
    validate_inputs(config, algorithm_type);
    
    %% Initialize Results Structure
    results = initialize_results_structure(config, algorithm_type);
    metrics = initialize_metrics_structure(config);
    
    %% Algorithm Execution with Error Handling
    try
        fprintf('Running %s optimization...\n', algorithm_type);
        tic; % Start timing
        
        switch upper(algorithm_type)
            case 'PSO'
                [results, metrics] = run_pso_optimization(config, results, metrics);
                
            case 'QL'
                [results, metrics] = run_qlearning_optimization(config, results, metrics, varargin);
                
            case 'HPQL'
                [results, metrics] = run_hpql_optimization(config, results, metrics, varargin);
                
            case 'PRIORITY'
                [results, metrics] = run_priority_optimization(config, results, metrics, varargin);
                
            case 'FIXED'
                [results, metrics] = run_fixed_allocation(config, results, metrics);
                
            case 'RESERVATION'
                [results, metrics] = run_reservation_allocation(config, results, metrics);
                
            otherwise
                error('Unknown algorithm type: %s', algorithm_type);
        end
        
        execution_time = toc;
        results.execution_time = execution_time;
        
        %% Post-Processing and Validation
        results = post_process_results(results, config);
        metrics = calculate_performance_metrics(results, config);
        
        %% Log Results
        log_optimization_results(results, metrics, config, algorithm_type);
        
        %% Display Summary
        display_optimization_summary(results, metrics, algorithm_type);
        
    catch ME
        % Error handling with detailed logging
        error_msg = sprintf('Optimization failed for %s: %s', algorithm_type, ME.message);
        log_error(error_msg, ME, config, algorithm_type);
        rethrow(ME);
    end
end

function validate_inputs(config, algorithm_type)
    % VALIDATE_INPUTS - Validates input parameters
    
    % Check configuration structure
    if ~isstruct(config) || ~isfield(config, 'network') || ~isfield(config, 'algorithm')
        error('Invalid configuration structure. Use master_config() to create configuration.');
    end
    
    % Check algorithm type
    valid_algorithms = {'PSO', 'QL', 'HPQL', 'PRIORITY', 'FIXED', 'RESERVATION'};
    if ~ismember(upper(algorithm_type), valid_algorithms)
        error('Invalid algorithm type. Valid options: %s', strjoin(valid_algorithms, ', '));
    end
    
    % Validate network configuration
    if ~isfield(config.network, 'network') || ~isfield(config.network.network, 'total_spectrum')
        error('Invalid network configuration. Check network_config.m');
    end
    
    % Validate algorithm configuration
    if ~isfield(config.algorithm, 'general') || ~isfield(config.algorithm.general, 'random_seed')
        error('Invalid algorithm configuration. Check algorithm_config.m');
    end
end

function results = initialize_results_structure(config, algorithm_type)
    % INITIALIZE_RESULTS_STRUCTURE - Creates standardized results structure
    
    results = struct();
    results.algorithm_type = upper(algorithm_type);
    results.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    results.config = config;
    
    % Algorithm-specific fields
    results.allocation = [];
    results.fitness_history = [];
    results.convergence_iteration = [];
    results.best_fitness = [];
    results.execution_time = [];
    
    % Performance tracking
    results.throughput = [];
    results.latency = [];
    results.utilization = [];
    results.fairness = [];
    results.efficiency = [];
    
    % Convergence analysis
    results.convergence_analysis = struct();
    results.convergence_analysis.converged = false;
    results.convergence_analysis.iterations_to_converge = [];
    results.convergence_analysis.final_fitness = [];
    results.convergence_analysis.fitness_improvement = [];
end

function metrics = initialize_metrics_structure(config)
    % INITIALIZE_METRICS_STRUCTURE - Creates standardized metrics structure
    
    metrics = struct();
    metrics.throughput = struct();
    metrics.latency = struct();
    metrics.utilization = struct();
    metrics.fairness = struct();
    metrics.efficiency = struct();
    metrics.qos_satisfaction = struct();
    
    % Initialize slice-specific metrics
    slice_types = config.network.network.slice_types;
    for i = 1:length(slice_types)
        slice_name = slice_types{i};
        metrics.throughput.(slice_name) = [];
        metrics.latency.(slice_name) = [];
        metrics.utilization.(slice_name) = [];
    end
end

function [results, metrics] = run_pso_optimization(config, results, metrics)
    % RUN_PSO_OPTIMIZATION - Executes PSO optimization
    
    % Extract PSO parameters
    pso_config = config.algorithm.pso;
    network_config = config.network.network;
    
    % Set random seed for reproducibility
    rng(config.algorithm.general.random_seed);
    
    % Initialize particles
    num_particles = pso_config.num_particles;
    num_slices = network_config.num_slices;
    total_spectrum = network_config.total_spectrum;
    demands = network_config.demands;
    
    particles = rand(num_particles, num_slices);
    particles = particles ./ sum(particles, 2) * total_spectrum;
    
    % Enforce demand constraints
    for i = 1:num_particles
        particles(i, :) = min(particles(i, :), demands);
        if sum(particles(i, :)) < total_spectrum
            remaining = total_spectrum - sum(particles(i, :));
            unmet = demands - particles(i, :);
            if sum(unmet) > 0
                distribute = remaining * (unmet / sum(unmet));
                particles(i, :) = particles(i, :) + distribute;
                particles(i, :) = min(particles(i, :), demands);
            end
        end
    end
    
    % PSO parameters
    velocities = zeros(size(particles));
    fitness_values = zeros(num_particles, 1);
    
    % Initialize fitness values
    for i = 1:num_particles
        fitness_values(i) = calculate_fitness(particles(i, :), config);
    end
    
    % Initialize personal and global best
    personal_best = particles;
    personal_best_values = fitness_values;
    [global_best_value, idx] = min(fitness_values);
    global_best = particles(idx, :);
    
    % PSO main loop
    fitness_history = zeros(1, pso_config.num_iterations);
    
    for iter = 1:pso_config.num_iterations
        for i = 1:num_particles
            % Update velocity and position
            velocities(i, :) = pso_config.w * velocities(i, :) + ...
                pso_config.c1 * rand * (personal_best(i, :) - particles(i, :)) + ...
                pso_config.c2 * rand * (global_best - particles(i, :));
            
            particles(i, :) = particles(i, :) + velocities(i, :);
            
            % Apply constraints
            particles(i, :) = min(max(particles(i, :), 0), demands);
            
            % Re-adjust total allocation
            if sum(particles(i, :)) > total_spectrum
                particles(i, :) = particles(i, :) / sum(particles(i, :)) * total_spectrum;
                particles(i, :) = min(particles(i, :), demands);
            elseif sum(particles(i, :)) < total_spectrum
                remaining = total_spectrum - sum(particles(i, :));
                unmet = demands - particles(i, :);
                if sum(unmet) > 0
                    distribute = remaining * (unmet / sum(unmet));
                    particles(i, :) = particles(i, :) + distribute;
                    particles(i, :) = min(particles(i, :), demands);
                end
            end
            
            % Evaluate fitness
            fitness = calculate_fitness(particles(i, :), config);
            
            % Update personal best
            if fitness < personal_best_values(i)
                personal_best(i, :) = particles(i, :);
                personal_best_values(i) = fitness;
            end
            
            % Update global best
            if fitness < global_best_value
                global_best = particles(i, :);
                global_best_value = fitness;
            end
        end
        
        % Store fitness history
        fitness_history(iter) = global_best_value;
        
        % Check convergence
        if iter > 10 && abs(fitness_history(iter) - fitness_history(iter-10)) < pso_config.convergence_threshold
            results.convergence_analysis.converged = true;
            results.convergence_analysis.iterations_to_converge = iter;
            break;
        end
    end
    
    % Store results
    results.allocation = round(global_best);
    results.fitness_history = fitness_history(1:iter);
    results.best_fitness = global_best_value;
    results.convergence_iteration = iter;
    
    % Calculate performance metrics
    metrics = calculate_performance_metrics(results, config);
end

function [results, metrics] = run_qlearning_optimization(config, results, metrics, varargin)
    % RUN_QLEARNING_OPTIMIZATION - Executes Q-Learning optimization
    
    % Extract Q-Learning parameters
    ql_config = config.algorithm.qlearning;
    network_config = config.network.network;
    
    % Set random seed for reproducibility
    rng(config.algorithm.general.random_seed);
    
    % Initialize Q-table
    q_table = zeros(network_config.num_slices, ql_config.num_actions);
    
    % Q-Learning training
    for episode = 1:ql_config.num_episodes
        state = randi(network_config.num_slices);
        
        for step = 1:ql_config.max_steps_per_episode
            % Choose action using epsilon-greedy policy
            if rand < ql_config.epsilon
                action = randi(ql_config.num_actions);
            else
                [~, action] = max(q_table(state, :));
            end
            
            % Calculate reward
            allocation = (action / ql_config.num_actions) * network_config.demands(state);
            throughput = calculate_throughput(allocation);
            latency = calculate_latency(allocation);
            reward = calculate_reward(throughput, latency, config);
            
            % Get next state
            next_state = randi(network_config.num_slices);
            
            % Q-learning update
            q_table(state, action) = q_table(state, action) + ql_config.alpha * ...
                (reward + ql_config.gamma * max(q_table(next_state, :)) - q_table(state, action));
            
            state = next_state;
        end
        
        % Decay epsilon
        ql_config.epsilon = max(ql_config.epsilon_min, ql_config.epsilon * ql_config.epsilon_decay);
    end
    
    % Derive allocation from Q-table
    [~, best_actions] = max(q_table, [], 2);
    tentative_allocation = (best_actions / sum(best_actions)) * network_config.total_spectrum;
    
    % Apply constraints
    results.allocation = apply_allocation_constraints(tentative_allocation, config);
    results.best_fitness = sum(max(q_table, [], 2));
    
    % Calculate performance metrics
    metrics = calculate_performance_metrics(results, config);
end

function [results, metrics] = run_hpql_optimization(config, results, metrics, varargin)
    % RUN_HPQL_OPTIMIZATION - Executes Hybrid PSO + Q-Learning optimization
    
    % First run PSO to get initial solution
    pso_results = struct();
    [pso_results, ~] = run_pso_optimization(config, pso_results, metrics);
    
    % Use PSO results to seed Q-Learning
    ql_config = config.algorithm.qlearning;
    network_config = config.network.network;
    
    % Initialize Q-table with PSO results
    q_table = zeros(network_config.num_slices, ql_config.num_actions);
    for i = 1:network_config.num_slices
        action = round((pso_results.allocation(i) / network_config.total_spectrum) * ql_config.num_actions);
        action = max(1, min(action, ql_config.num_actions));
        q_table(i, action) = 1; % Optimistic initialization
    end
    
    % Q-Learning refinement
    for episode = 1:ql_config.num_episodes
        state = randi(network_config.num_slices);
        
        for step = 1:ql_config.max_steps_per_episode
            % Choose action using epsilon-greedy policy
            if rand < ql_config.epsilon
                action = randi(ql_config.num_actions);
            else
                [~, action] = max(q_table(state, :));
            end
            
            % Calculate reward with PSO influence
            allocation = (action / ql_config.num_actions) * network_config.demands(state);
            throughput = calculate_throughput(allocation);
            latency = calculate_latency(allocation);
            reward = calculate_reward(throughput, latency, config);
            
            % Add PSO influence
            pso_influence = config.algorithm.hpql.beta * (pso_results.best_fitness - pso_results.fitness_history(end));
            reward = reward + pso_influence;
            
            % Get next state
            next_state = randi(network_config.num_slices);
            
            % Q-learning update
            q_table(state, action) = q_table(state, action) + ql_config.alpha * ...
                (reward + ql_config.gamma * max(q_table(next_state, :)) - q_table(state, action));
            
            state = next_state;
        end
    end
    
    % Derive final allocation
    [~, best_actions] = max(q_table, [], 2);
    tentative_allocation = (best_actions / sum(best_actions)) * network_config.total_spectrum;
    
    % Apply constraints
    results.allocation = apply_allocation_constraints(tentative_allocation, config);
    results.best_fitness = pso_results.best_fitness * config.algorithm.hpql.pso_weight + ...
                          sum(max(q_table, [], 2)) * config.algorithm.hpql.ql_weight;
    
    % Calculate performance metrics
    metrics = calculate_performance_metrics(results, config);
end

function [results, metrics] = run_priority_optimization(config, results, metrics, varargin)
    % RUN_PRIORITY_OPTIMIZATION - Executes priority-aware resource allocation
    
    network_config = config.network.network;
    priority_config = config.algorithm.priority;
    
    % Initialize allocation
    num_slices = network_config.num_slices;
    total_spectrum = network_config.total_spectrum;
    demands = network_config.demands;
    priorities = network_config.priorities;
    reserve_percent = network_config.reserve_percent;
    
    allocation = zeros(1, num_slices);
    
    % Step 1: Reserve minimum PRBs per slice
    for i = 1:num_slices
        allocation(i) = min(demands(i), reserve_percent(i) * total_spectrum);
    end
    
    % Step 2: Calculate remaining spectrum
    reserved_total = sum(allocation);
    remaining = total_spectrum - reserved_total;
    
    % Step 3: Priority-aware allocation of remaining spectrum
    unmet_demands = demands - allocation;
    
    if strcmp(priority_config.method, 'weighted_efficiency')
        efficiency = priorities ./ (unmet_demands + 1e-3);
        [~, idx] = sort(efficiency, 'descend');
    else
        [~, idx] = sort(priorities, 'descend');
    end
    
    for i = 1:num_slices
        slice = idx(i);
        alloc = min(unmet_demands(slice), remaining);
        allocation(slice) = allocation(slice) + alloc;
        remaining = remaining - alloc;
        if remaining <= 0
            break;
        end
    end
    
    % Store results
    results.allocation = round(allocation);
    results.best_fitness = calculate_fitness(allocation, config);
    
    % Calculate performance metrics
    metrics = calculate_performance_metrics(results, config);
end

function [results, metrics] = run_fixed_allocation(config, results, metrics, varargin)
    % RUN_FIXED_ALLOCATION - Executes fixed resource allocation
    
    network_config = config.network.network;
    total_spectrum = network_config.total_spectrum;
    num_slices = network_config.num_slices;
    
    % Equal allocation
    allocation = ones(1, num_slices) * (total_spectrum / num_slices);
    
    % Store results
    results.allocation = round(allocation);
    results.best_fitness = calculate_fitness(allocation, config);
    
    % Calculate performance metrics
    metrics = calculate_performance_metrics(results, config);
end

function [results, metrics] = run_reservation_allocation(config, results, metrics, varargin)
    % RUN_RESERVATION_ALLOCATION - Executes reservation-based allocation
    
    network_config = config.network.network;
    total_spectrum = network_config.total_spectrum;
    demands = network_config.demands;
    reserved = network_config.reserved;
    
    % Allocate reserved resources
    allocation = min(demands, reserved);
    
    % Distribute remaining resources proportionally
    remaining = total_spectrum - sum(allocation);
    if remaining > 0
        unmet = demands - allocation;
        if sum(unmet) > 0
            distribute = remaining * (unmet / sum(unmet));
            allocation = allocation + distribute;
            allocation = min(allocation, demands);
        end
    end
    
    % Store results
    results.allocation = round(allocation);
    results.best_fitness = calculate_fitness(allocation, config);
    
    % Calculate performance metrics
    metrics = calculate_performance_metrics(results, config);
end

function fitness = calculate_fitness(allocation, config)
    % CALCULATE_FITNESS - Calculates fitness value for allocation
    
    % Calculate throughput, latency, and utilization
    throughput = calculate_throughput(allocation);
    latency = calculate_latency(allocation);
    utilization = calculate_utilization(allocation, config);
    
    % Multi-objective fitness calculation
    weights = config.network.performance.weights;
    fitness = -(weights.throughput * sum(throughput) - ...
                weights.latency * sum(latency) + ...
                weights.utilization * utilization);
end

function throughput = calculate_throughput(allocation)
    % CALCULATE_THROUGHPUT - Calculates throughput for allocation
    
    SINR = 10 * log10(allocation + 1);
    throughput = allocation .* log2(1 + SINR);
end

function latency = calculate_latency(allocation)
    % CALCULATE_LATENCY - Calculates latency for allocation
    
    latency = 1 ./ (allocation + 0.01); % Avoid division by zero
end

function utilization = calculate_utilization(allocation, config)
    % CALCULATE_UTILIZATION - Calculates resource utilization
    
    total_spectrum = config.network.network.total_spectrum;
    utilization = sum(allocation) / total_spectrum;
end

function reward = calculate_reward(throughput, latency, config)
    % CALCULATE_REWARD - Calculates reward for Q-Learning
    
    weights = config.network.performance.reward_weights;
    latency_threshold = config.network.performance.latency_threshold;
    
    throughput_reward = weights.throughput * (throughput / (1 + throughput));
    latency_penalty = 0;
    
    if latency > latency_threshold
        latency_penalty = weights.latency * (-1 * (latency - latency_threshold) / latency_threshold);
    end
    
    reward = throughput_reward + latency_penalty;
end

function allocation = apply_allocation_constraints(tentative_allocation, config)
    % APPLY_ALLOCATION_CONSTRAINTS - Applies allocation constraints
    
    network_config = config.network.network;
    total_spectrum = network_config.total_spectrum;
    demands = network_config.demands;
    
    % Cap at demands
    allocation = min(tentative_allocation, demands);
    
    % Ensure total doesn't exceed spectrum
    if sum(allocation) > total_spectrum
        allocation = allocation / sum(allocation) * total_spectrum;
        allocation = min(allocation, demands);
    end
    
    % Distribute remaining spectrum
    remaining = total_spectrum - sum(allocation);
    if remaining > 0
        unmet = demands - allocation;
        if sum(unmet) > 0
            distribute = remaining * (unmet / sum(unmet));
            allocation = allocation + distribute;
            allocation = min(allocation, demands);
        end
    end
end

function results = post_process_results(results, config)
    % POST_PROCESS_RESULTS - Post-processes optimization results
    
    % Round allocation to integers
    results.allocation = round(results.allocation);
    
    % Validate final allocation
    utils = config_utils();
    is_valid = utils.validate_allocation_constraints(results.allocation, config);
    if ~is_valid
        warning('Final allocation validation failed');
    end
    
    % Calculate convergence metrics
    if ~isempty(results.fitness_history)
        results.convergence_analysis.final_fitness = results.fitness_history(end);
        if length(results.fitness_history) > 1
            results.convergence_analysis.fitness_improvement = ...
                results.fitness_history(1) - results.fitness_history(end);
        end
    end
end

function metrics = calculate_performance_metrics(results, config)
    % CALCULATE_PERFORMANCE_METRICS - Calculates comprehensive performance metrics
    
    allocation = results.allocation;
    slice_types = config.network.network.slice_types;
    
    % Calculate basic metrics
    throughput = calculate_throughput(allocation);
    latency = calculate_latency(allocation);
    utilization = calculate_utilization(allocation, config);
    
    % Store slice-specific metrics
    for i = 1:length(slice_types)
        slice_name = slice_types{i};
        metrics.throughput.(slice_name) = throughput(i);
        metrics.latency.(slice_name) = latency(i);
        metrics.utilization.(slice_name) = allocation(i) / config.network.network.demands(i);
    end
    
    % Calculate fairness (Jain's fairness index)
    if sum(throughput) > 0
        metrics.fairness.jains_index = (sum(throughput))^2 / (length(throughput) * sum(throughput.^2));
    else
        metrics.fairness.jains_index = 0;
    end
    
    % Calculate efficiency
    metrics.efficiency.spectrum_efficiency = sum(throughput) / config.network.network.total_spectrum;
    metrics.efficiency.utilization_efficiency = utilization;
    
    % Calculate QoS satisfaction
    qos_requirements = config.network.qos;
    qos_satisfaction = zeros(1, length(slice_types));
    
    for i = 1:length(slice_types)
        throughput_satisfied = throughput(i) >= qos_requirements.throughput_requirements(i);
        latency_satisfied = latency(i) <= qos_requirements.latency_thresholds(i);
        qos_satisfaction(i) = (throughput_satisfied + latency_satisfied) / 2;
    end
    
    metrics.qos_satisfaction.overall = mean(qos_satisfaction);
    metrics.qos_satisfaction.per_slice = qos_satisfaction;
end

function log_optimization_results(results, metrics, config, algorithm_type)
    % LOG_OPTIMIZATION_RESULTS - Logs optimization results
    
    log_file = fullfile(config.output.logs_dir, ...
                       sprintf('%s_%s_results.log', config.output.file_prefix, algorithm_type));
    
    fid = fopen(log_file, 'w');
    if fid ~= -1
        fprintf(fid, '=== %s Optimization Results ===\n', algorithm_type);
        fprintf(fid, 'Timestamp: %s\n', results.timestamp);
        fprintf(fid, 'Execution Time: %.4f seconds\n', results.execution_time);
        fprintf(fid, 'Best Fitness: %.4f\n', results.best_fitness);
        fprintf(fid, 'Allocation: [%s]\n', num2str(results.allocation));
        fprintf(fid, 'Convergence: %s\n', mat2str(results.convergence_analysis.converged));
        
        if results.convergence_analysis.converged
            fprintf(fid, 'Iterations to Converge: %d\n', results.convergence_analysis.iterations_to_converge);
        end
        
        fprintf(fid, '\n--- Performance Metrics ---\n');
        fprintf(fid, 'Overall Throughput: %.4f\n', sum([metrics.throughput.eMBB, metrics.throughput.URLLC, metrics.throughput.mMTC]));
        fprintf(fid, 'Overall Latency: %.4f\n', mean([metrics.latency.eMBB, metrics.latency.URLLC, metrics.latency.mMTC]));
        fprintf(fid, 'Utilization: %.4f\n', metrics.efficiency.utilization_efficiency);
        fprintf(fid, 'Fairness Index: %.4f\n', metrics.fairness.jains_index);
        fprintf(fid, 'QoS Satisfaction: %.4f\n', metrics.qos_satisfaction.overall);
        
        fclose(fid);
    end
end

function display_optimization_summary(results, metrics, algorithm_type)
    % DISPLAY_OPTIMIZATION_SUMMARY - Displays optimization summary
    
    fprintf('\n=== %s Optimization Summary ===\n', algorithm_type);
    fprintf('Best Fitness: %.4f\n', results.best_fitness);
    fprintf('Allocation: [%s]\n', num2str(results.allocation));
    fprintf('Execution Time: %.4f seconds\n', results.execution_time);
    
    if results.convergence_analysis.converged
        fprintf('Converged in %d iterations\n', results.convergence_analysis.iterations_to_converge);
    else
        fprintf('Did not converge within iteration limit\n');
    end
    
    fprintf('Overall Throughput: %.4f\n', sum([metrics.throughput.eMBB, metrics.throughput.URLLC, metrics.throughput.mMTC]));
    fprintf('Overall Latency: %.4f\n', mean([metrics.latency.eMBB, metrics.latency.URLLC, metrics.latency.mMTC]));
    fprintf('Utilization: %.4f\n', metrics.efficiency.utilization_efficiency);
    fprintf('Fairness Index: %.4f\n', metrics.fairness.jains_index);
    fprintf('QoS Satisfaction: %.4f\n', metrics.qos_satisfaction.overall);
    fprintf('================================\n\n');
end

function log_error(error_msg, ME, config, algorithm_type)
    % LOG_ERROR - Logs error information
    
    error_file = fullfile(config.output.logs_dir, ...
                         sprintf('%s_%s_error.log', config.output.file_prefix, algorithm_type));
    
    fid = fopen(error_file, 'w');
    if fid ~= -1
        fprintf(fid, '=== %s Optimization Error ===\n', algorithm_type);
        fprintf(fid, 'Timestamp: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        fprintf(fid, 'Error Message: %s\n', error_msg);
        fprintf(fid, 'Exception: %s\n', ME.message);
        fprintf(fid, 'Stack Trace:\n');
        
        for i = 1:length(ME.stack)
            fprintf(fid, '  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
        
        fclose(fid);
    end
end 