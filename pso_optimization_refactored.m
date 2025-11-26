function [results, metrics] = pso_optimization_refactored(config)
    % PSO_OPTIMIZATION_REFACTORED - Refactored PSO optimization algorithm
    %
    % This function implements a research-grade Particle Swarm Optimization
    % algorithm for 5G network slicing resource allocation. It uses the
    % centralized configuration system and provides comprehensive performance
    % tracking and validation.
    %
    % Inputs:
    %   config: Configuration structure from master_config()
    %
    % Returns:
    %   results: Structure containing optimization results
    %   metrics: Structure containing performance metrics
    %
    % Author: [Your Name]
    % Date: [Current Date]
    % Institution: [Your Institution]
    % Research: 5G Network Slicing Resource Allocation Optimization
    %
    % References:
    % - Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization
    % - Clerc, M., & Kennedy, J. (2002). The particle swarm-explosion, stability, and convergence
    
    %% Input Validation
    validate_pso_inputs(config);
    
    %% Extract Configuration Parameters
    pso_config = config.algorithm.pso;
    network_config = config.network.network;
    
    % Set random seed for reproducibility
    rng(config.algorithm.general.random_seed);
    
    %% Initialize Algorithm Parameters
    num_particles = pso_config.num_particles;
    num_iterations = pso_config.num_iterations;
    num_slices = network_config.num_slices;
    total_spectrum = network_config.total_spectrum;
    demands = network_config.demands;
    
    %% Initialize Results Structure
    results = struct();
    results.algorithm_type = 'PSO';
    results.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    results.config = config;
    results.fitness_history = zeros(1, num_iterations);
    results.convergence_analysis = struct();
    
    %% Initialize Particle Swarm
    [particles, velocities] = initialize_particle_swarm(num_particles, num_slices, ...
                                                      total_spectrum, demands, pso_config);
    
    %% Initialize Personal and Global Best
    [personal_best, personal_best_values, global_best, global_best_value] = ...
        initialize_best_positions(particles, config);
    
    %% PSO Main Optimization Loop
    fprintf('Starting PSO optimization with %d particles and %d iterations...\n', ...
            num_particles, num_iterations);
    
    tic; % Start timing
    
    for iter = 1:num_iterations
        % Update particles
        [particles, velocities] = update_particles(particles, velocities, ...
                                                 personal_best, global_best, ...
                                                 pso_config, iter, num_iterations);
        
        % Apply constraints
        particles = apply_pso_constraints(particles, total_spectrum, demands);
        
        % Evaluate fitness and update best positions
        [personal_best, personal_best_values, global_best, global_best_value] = ...
            update_best_positions(particles, personal_best, personal_best_values, ...
                                global_best, global_best_value, config);
        
        % Store fitness history
        results.fitness_history(iter) = global_best_value;
        
        % Check convergence
        if check_convergence(results.fitness_history, iter, pso_config)
            results.convergence_analysis.converged = true;
            results.convergence_analysis.iterations_to_converge = iter;
            fprintf('PSO converged at iteration %d\n', iter);
            break;
        end
        
        % Display progress
        if mod(iter, 10) == 0
            fprintf('Iteration %d/%d: Best fitness = %.4f\n', iter, num_iterations, global_best_value);
        end
    end
    
    execution_time = toc;
    
    %% Store Final Results
    results.allocation = round(global_best);
    results.best_fitness = global_best_value;
    results.execution_time = execution_time;
    results.convergence_iteration = iter;
    
    % Truncate fitness history to actual iterations
    results.fitness_history = results.fitness_history(1:iter);
    
    %% Post-Processing
    results = post_process_pso_results(results, config);
    
    %% Calculate Performance Metrics
    metrics = calculate_pso_metrics(results, config);
    
    %% Display Results
    display_pso_results(results, metrics);
    
    fprintf('PSO optimization completed successfully!\n');
end

function validate_pso_inputs(config)
    % VALIDATE_PSO_INPUTS - Validates PSO input parameters
    
    if ~isstruct(config) || ~isfield(config, 'algorithm') || ~isfield(config, 'network')
        error('Invalid configuration structure. Use master_config() to create configuration.');
    end
    
    if ~isfield(config.algorithm, 'pso')
        error('PSO configuration not found. Check algorithm_config.m');
    end
    
    pso_config = config.algorithm.pso;
    required_fields = {'num_particles', 'num_iterations', 'c1', 'c2', 'w'};
    
    for i = 1:length(required_fields)
        if ~isfield(pso_config, required_fields{i})
            error('Missing PSO parameter: %s', required_fields{i});
        end
    end
    
    % Validate parameter ranges
    if pso_config.num_particles <= 0
        error('Number of particles must be positive');
    end
    
    if pso_config.num_iterations <= 0
        error('Number of iterations must be positive');
    end
    
    if pso_config.c1 <= 0 || pso_config.c2 <= 0
        error('Learning factors c1 and c2 must be positive');
    end
    
    if pso_config.w < 0 || pso_config.w > 1
        error('Inertia weight w must be between 0 and 1');
    end
end

function [particles, velocities] = initialize_particle_swarm(num_particles, num_slices, ...
                                                           total_spectrum, demands, pso_config)
    % INITIALIZE_PARTICLE_SWARM - Initializes particle swarm
    
    % Initialize particles with random positions
    particles = rand(num_particles, num_slices);
    particles = particles ./ sum(particles, 2) * total_spectrum;
    
    % Enforce demand constraints on initial particles
    for i = 1:num_particles
        particles(i, :) = min(particles(i, :), demands);
        
        % Redistribute remaining spectrum if needed
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
    
    % Initialize velocities
    velocities = zeros(size(particles));
    
    % Apply velocity constraints if specified
    if isfield(pso_config, 'velocity_max') && isfield(pso_config, 'velocity_min')
        velocities = velocities + (pso_config.velocity_max - pso_config.velocity_min) * ...
                    rand(size(velocities)) + pso_config.velocity_min;
    end
end

function [personal_best, personal_best_values, global_best, global_best_value] = ...
    initialize_best_positions(particles, config)
    % INITIALIZE_BEST_POSITIONS - Initializes personal and global best positions
    
    num_particles = size(particles, 1);
    
    % Initialize personal best
    personal_best = particles;
    personal_best_values = zeros(num_particles, 1);
    
    % Evaluate initial fitness values
    for i = 1:num_particles
        personal_best_values(i) = calculate_fitness(particles(i, :), config);
    end
    
    % Initialize global best
    [global_best_value, idx] = min(personal_best_values);
    global_best = particles(idx, :);
end

function [particles, velocities] = update_particles(particles, velocities, ...
                                                  personal_best, global_best, ...
                                                  pso_config, iter, num_iterations)
    % UPDATE_PARTICLES - Updates particle positions and velocities
    
    num_particles = size(particles, 1);
    
    % Adaptive inertia weight if enabled
    w = pso_config.w;
    if isfield(pso_config, 'adaptive_inertia') && pso_config.adaptive_inertia
        w = pso_config.w_max - (pso_config.w_max - pso_config.w_min) * iter / num_iterations;
    end
    
    % Update each particle
    for i = 1:num_particles
        % Update velocity
        velocities(i, :) = w * velocities(i, :) + ...
            pso_config.c1 * rand * (personal_best(i, :) - particles(i, :)) + ...
            pso_config.c2 * rand * (global_best - particles(i, :));
        
        % Update position
        particles(i, :) = particles(i, :) + velocities(i, :);
    end
end

function particles = apply_pso_constraints(particles, total_spectrum, demands)
    % APPLY_PSO_CONSTRAINTS - Applies constraints to particle positions
    
    num_particles = size(particles, 1);
    
    for i = 1:num_particles
        % Clamp values to [0, demand]
        particles(i, :) = min(max(particles(i, :), 0), demands);
        
        % Re-adjust total to not exceed total_spectrum
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
    end
end

function [personal_best, personal_best_values, global_best, global_best_value] = ...
    update_best_positions(particles, personal_best, personal_best_values, ...
                         global_best, global_best_value, config)
    % UPDATE_BEST_POSITIONS - Updates personal and global best positions
    
    num_particles = size(particles, 1);
    
    for i = 1:num_particles
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
end

function converged = check_convergence(fitness_history, iter, pso_config)
    % CHECK_CONVERGENCE - Checks if PSO has converged
    
    if iter < 10
        converged = false;
        return;
    end
    
    % Check if fitness has improved significantly in recent iterations
    recent_window = max(1, iter - pso_config.convergence_threshold + 1):iter;
    recent_fitness = fitness_history(recent_window);
    
    % Calculate improvement
    improvement = abs(recent_fitness(end) - recent_fitness(1));
    
    % Check if improvement is below threshold
    if improvement < pso_config.convergence_threshold
        converged = true;
    else
        converged = false;
    end
end

function fitness = calculate_fitness(allocation, config)
    % CALCULATE_FITNESS - Calculates fitness value for allocation
    
    % Calculate performance metrics
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

function results = post_process_pso_results(results, config)
    % POST_PROCESS_PSO_RESULTS - Post-processes PSO results
    
    % Round allocation to integers
    results.allocation = round(results.allocation);
    
    % Validate final allocation
    utils = config_utils();
    is_valid = utils.validate_allocation_constraints(results.allocation, config);
    if ~is_valid
        warning('PSO final allocation validation failed');
    end
    
    % Calculate convergence metrics
    if ~isempty(results.fitness_history)
        results.convergence_analysis.final_fitness = results.fitness_history(end);
        if length(results.fitness_history) > 1
            results.convergence_analysis.fitness_improvement = ...
                results.fitness_history(1) - results.fitness_history(end);
        end
    end
    
    % Calculate convergence rate
    if length(results.fitness_history) > 1
        results.convergence_analysis.convergence_rate = ...
            (results.fitness_history(1) - results.fitness_history(end)) / ...
            results.fitness_history(1) * 100;
    end
end

function metrics = calculate_pso_metrics(results, config)
    % CALCULATE_PSO_METRICS - Calculates PSO-specific performance metrics
    
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
    
    % PSO-specific metrics
    metrics.pso.convergence_iterations = results.convergence_iteration;
    metrics.pso.final_fitness = results.best_fitness;
    metrics.pso.execution_time = results.execution_time;
    metrics.pso.convergence_rate = results.convergence_analysis.convergence_rate;
end

function display_pso_results(results, metrics)
    % DISPLAY_PSO_RESULTS - Displays PSO optimization results
    
    fprintf('\n=== PSO Optimization Results ===\n');
    fprintf('Best Fitness: %.4f\n', results.best_fitness);
    fprintf('Allocation: [%s]\n', num2str(results.allocation));
    fprintf('Execution Time: %.4f seconds\n', results.execution_time);
    fprintf('Convergence Iterations: %d\n', results.convergence_iteration);
    
    if results.convergence_analysis.converged
        fprintf('Convergence Rate: %.2f%%\n', results.convergence_analysis.convergence_rate);
    end
    
    fprintf('Overall Throughput: %.4f\n', sum([metrics.throughput.eMBB, metrics.throughput.URLLC, metrics.throughput.mMTC]));
    fprintf('Overall Latency: %.4f\n', mean([metrics.latency.eMBB, metrics.latency.URLLC, metrics.latency.mMTC]));
    fprintf('Utilization: %.4f\n', metrics.efficiency.utilization_efficiency);
    fprintf('Fairness Index: %.4f\n', metrics.fairness.jains_index);
    fprintf('QoS Satisfaction: %.4f\n', metrics.qos_satisfaction.overall);
    fprintf('================================\n\n');
end 