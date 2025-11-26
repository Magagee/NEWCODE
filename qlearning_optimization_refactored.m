function [results, metrics] = qlearning_optimization_refactored(config)
    % QLEARNING_OPTIMIZATION_REFACTORED - Refactored Q-Learning optimization algorithm
    %
    % This function implements a research-grade Q-Learning algorithm for 5G
    % network slicing resource allocation. It uses the centralized configuration
    % system and provides comprehensive performance tracking and validation.
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
    % - Watkins, C. J., & Dayan, P. (1992). Q-learning
    % - Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction
    
    %% Input Validation
    validate_qlearning_inputs(config);
    
    %% Extract Configuration Parameters
    ql_config = config.algorithm.qlearning;
    network_config = config.network.network;
    
    % Set random seed for reproducibility
    rng(config.algorithm.general.random_seed);
    
    %% Initialize Algorithm Parameters
    num_slices = network_config.num_slices;
    num_actions = ql_config.num_actions;
    num_episodes = ql_config.num_episodes;
    max_steps = ql_config.max_steps_per_episode;
    
    %% Initialize Results Structure
    results = struct();
    results.algorithm_type = 'QL';
    results.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    results.config = config;
    results.q_table = zeros(num_slices, num_actions);
    results.episode_rewards = zeros(1, num_episodes);
    results.convergence_analysis = struct();
    
    %% Initialize Q-Table
    q_table = initialize_q_table(num_slices, num_actions, ql_config);
    
    %% Q-Learning Training
    fprintf('Starting Q-Learning optimization with %d episodes...\n', num_episodes);
    
    tic; % Start timing
    
    for episode = 1:num_episodes
        % Initialize episode
        state = randi(num_slices);
        episode_reward = 0;
        
        for step = 1:max_steps
            % Choose action using epsilon-greedy policy
            action = choose_action(q_table, state, ql_config, episode);
            
            % Execute action and observe reward
            reward = execute_action(state, action, config);
            episode_reward = episode_reward + reward;
            
            % Get next state
            next_state = get_next_state(num_slices);
            
            % Q-learning update
            q_table = update_q_table(q_table, state, action, reward, next_state, ql_config);
            
            % Move to next state
            state = next_state;
        end
        
        % Store episode reward
        results.episode_rewards(episode) = episode_reward;
        
        % Decay epsilon
        ql_config.epsilon = max(ql_config.epsilon_min, ...
                               ql_config.epsilon * ql_config.epsilon_decay);
        
        % Check convergence
        if check_qlearning_convergence(results.episode_rewards, episode, ql_config)
            results.convergence_analysis.converged = true;
            results.convergence_analysis.episodes_to_converge = episode;
            fprintf('Q-Learning converged at episode %d\n', episode);
            break;
        end
        
        % Display progress
        if mod(episode, 100) == 0
            avg_reward = mean(results.episode_rewards(max(1, episode-99):episode));
            fprintf('Episode %d/%d: Average reward = %.4f, Epsilon = %.4f\n', ...
                    episode, num_episodes, avg_reward, ql_config.epsilon);
        end
    end
    
    execution_time = toc;
    
    %% Derive Allocation from Q-Table
    allocation = derive_allocation_from_qtable(q_table, config);
    
    %% Store Final Results
    results.q_table = q_table;
    results.allocation = allocation;
    results.best_fitness = calculate_fitness(allocation, config);
    results.execution_time = execution_time;
    results.convergence_episode = episode;
    
    % Truncate episode rewards to actual episodes
    results.episode_rewards = results.episode_rewards(1:episode);
    
    %% Post-Processing
    results = post_process_qlearning_results(results, config);
    
    %% Calculate Performance Metrics
    metrics = calculate_qlearning_metrics(results, config);
    
    %% Display Results
    display_qlearning_results(results, metrics);
    
    fprintf('Q-Learning optimization completed successfully!\n');
end

function validate_qlearning_inputs(config)
    % VALIDATE_QLEARNING_INPUTS - Validates Q-Learning input parameters
    
    if ~isstruct(config) || ~isfield(config, 'algorithm') || ~isfield(config, 'network')
        error('Invalid configuration structure. Use master_config() to create configuration.');
    end
    
    if ~isfield(config.algorithm, 'qlearning')
        error('Q-Learning configuration not found. Check algorithm_config.m');
    end
    
    ql_config = config.algorithm.qlearning;
    required_fields = {'alpha', 'gamma', 'epsilon', 'num_episodes', 'num_actions'};
    
    for i = 1:length(required_fields)
        if ~isfield(ql_config, required_fields{i})
            error('Missing Q-Learning parameter: %s', required_fields{i});
        end
    end
    
    % Validate parameter ranges
    if ql_config.alpha <= 0 || ql_config.alpha > 1
        error('Learning rate alpha must be in (0, 1]');
    end
    
    if ql_config.gamma < 0 || ql_config.gamma > 1
        error('Discount factor gamma must be in [0, 1]');
    end
    
    if ql_config.epsilon < 0 || ql_config.epsilon > 1
        error('Exploration rate epsilon must be in [0, 1]');
    end
    
    if ql_config.num_episodes <= 0
        error('Number of episodes must be positive');
    end
    
    if ql_config.num_actions <= 0
        error('Number of actions must be positive');
    end
end

function q_table = initialize_q_table(num_slices, num_actions, ql_config)
    % INITIALIZE_Q_TABLE - Initializes Q-table
    
    if ql_config.optimistic_init
        % Optimistic initialization
        q_table = ones(num_slices, num_actions) * ql_config.initial_q_value;
    else
        % Standard initialization
        q_table = zeros(num_slices, num_actions);
    end
end

function action = choose_action(q_table, state, ql_config, episode)
    % CHOOSE_ACTION - Chooses action using epsilon-greedy policy
    
    if rand < ql_config.epsilon
        % Explore: choose random action
        action = randi(size(q_table, 2));
    else
        % Exploit: choose best action
        [~, action] = max(q_table(state, :));
    end
end

function reward = execute_action(state, action, config)
    % EXECUTE_ACTION - Executes action and returns reward
    
    % Convert action to allocation
    network_config = config.network.network;
    demands = network_config.demands;
    num_actions = config.algorithm.qlearning.num_actions;
    
    allocation = (action / num_actions) * demands(state);
    
    % Calculate performance metrics
    throughput = calculate_throughput(allocation);
    latency = calculate_latency(allocation);
    
    % Calculate reward
    reward = calculate_reward(throughput, latency, config);
    
    % Apply priority weighting
    priorities = network_config.priorities;
    priority = priorities(state);
    reward = reward * (1 + priority);
end

function next_state = get_next_state(num_slices)
    % GET_NEXT_STATE - Gets next state (slice)
    
    next_state = randi(num_slices);
end

function q_table = update_q_table(q_table, state, action, reward, next_state, ql_config)
    % UPDATE_Q_TABLE - Updates Q-table using Q-learning update rule
    
    % Q-learning update
    current_q = q_table(state, action);
    max_next_q = max(q_table(next_state, :));
    
    q_table(state, action) = current_q + ql_config.alpha * ...
        (reward + ql_config.gamma * max_next_q - current_q);
end

function converged = check_qlearning_convergence(episode_rewards, episode, ql_config)
    % CHECK_QLEARNING_CONVERGENCE - Checks if Q-Learning has converged
    
    if episode < ql_config.min_episodes
        converged = false;
        return;
    end
    
    % Check if average reward has stabilized
    window_size = min(100, episode);
    recent_rewards = episode_rewards(max(1, episode - window_size + 1):episode);
    
    % Calculate coefficient of variation
    if mean(recent_rewards) > 0
        cv = std(recent_rewards) / mean(recent_rewards);
        converged = cv < ql_config.convergence_threshold;
    else
        converged = false;
    end
end

function allocation = derive_allocation_from_qtable(q_table, config)
    % DERIVE_ALLOCATION_FROM_QTABLE - Derives allocation from Q-table
    
    network_config = config.network.network;
    total_spectrum = network_config.total_spectrum;
    demands = network_config.demands;
    num_actions = config.algorithm.qlearning.num_actions;
    
    % Get best actions for each slice
    [~, best_actions] = max(q_table, [], 2);
    
    % Convert actions to proportional allocation based on Q-values
    % Normalize Q-values to get proportional allocation
    q_values = zeros(1, length(best_actions));
    for i = 1:length(best_actions)
        q_values(i) = q_table(i, best_actions(i));
    end
    
    % Convert to proportional allocation, ensuring it doesn't exceed total spectrum
    if sum(q_values) > 0
        tentative_allocation = (q_values / sum(q_values)) * total_spectrum;
    else
        % If all Q-values are zero, distribute equally
        tentative_allocation = ones(1, length(best_actions)) * (total_spectrum / length(best_actions));
    end
    
    % Apply constraints
    allocation = apply_allocation_constraints(tentative_allocation, config);
    
    % Final validation and fixing using config_utils
    utils = config_utils();
    if ~utils.validate_allocation_constraints(allocation, config)
        fprintf('QL allocation validation failed, fixing constraints...\n');
        allocation = utils.fix_allocation_constraints(allocation, config);
    end
end

function allocation = apply_allocation_constraints(tentative_allocation, config)
    % APPLY_ALLOCATION_CONSTRAINTS - Applies allocation constraints
    
    network_config = config.network.network;
    total_spectrum = network_config.total_spectrum;
    demands = network_config.demands;
    
    % Ensure allocation is a row vector
    if iscolumn(tentative_allocation)
        tentative_allocation = tentative_allocation';
    end
    
    % Cap at demands
    allocation = min(tentative_allocation, demands);
    
    % Ensure total doesn't exceed spectrum
    if sum(allocation) > total_spectrum
        allocation = allocation / sum(allocation) * total_spectrum;
        allocation = min(allocation, demands);
    end
    
    % Ensure minimum allocation per slice (at least 1 PRB)
    allocation = max(allocation, 1);
    
    % Redistribute if still over total spectrum
    if sum(allocation) > total_spectrum
        allocation = allocation / sum(allocation) * total_spectrum;
        allocation = max(round(allocation), 1); % Round and ensure minimum
    end
    
    % Distribute remaining spectrum proportionally
    remaining = total_spectrum - sum(allocation);
    if remaining > 0
        unmet = demands - allocation;
        if sum(unmet) > 0
            distribute = remaining * (unmet / sum(unmet));
            allocation = allocation + distribute;
            allocation = min(allocation, demands);
        end
    end
    
    % Final validation and rounding
    allocation = max(round(allocation), 1); % Ensure minimum 1 PRB per slice
    if sum(allocation) > total_spectrum
        allocation = allocation / sum(allocation) * total_spectrum;
        allocation = round(allocation);
    end
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

function throughput = calculate_throughput(allocation)
    % CALCULATE_THROUGHPUT - Calculates throughput for allocation
    
    SINR = 10 * log10(allocation + 1);
    throughput = allocation * log2(1 + SINR);
end

function latency = calculate_latency(allocation)
    % CALCULATE_LATENCY - Calculates latency for allocation
    
    latency = 1 / (allocation + 0.01); % Avoid division by zero
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

function utilization = calculate_utilization(allocation, config)
    % CALCULATE_UTILIZATION - Calculates resource utilization
    
    total_spectrum = config.network.network.total_spectrum;
    utilization = sum(allocation) / total_spectrum;
end

function results = post_process_qlearning_results(results, config)
    % POST_PROCESS_QLEARNING_RESULTS - Post-processes Q-Learning results
    
    % Round allocation to integers
    results.allocation = round(results.allocation);
    
    % Validate final allocation
    utils = config_utils();
    is_valid = utils.validate_allocation_constraints(results.allocation, config);
    if ~is_valid
        warning('Q-Learning final allocation validation failed');
        % Fix the allocation
        results.allocation = utils.fix_allocation_constraints(results.allocation, config);
        fprintf('Fixed QL allocation: [%.1f %.1f %.1f], Total: %.1f\n', ...
                results.allocation(1), results.allocation(2), results.allocation(3), sum(results.allocation));
    end
    
    % Calculate convergence metrics
    if ~isempty(results.episode_rewards)
        results.convergence_analysis.final_reward = results.episode_rewards(end);
        if length(results.episode_rewards) > 1
            results.convergence_analysis.reward_improvement = ...
                results.episode_rewards(end) - results.episode_rewards(1);
        end
    end
    
    % Calculate learning efficiency
    if length(results.episode_rewards) > 1
        results.convergence_analysis.learning_efficiency = ...
            (results.episode_rewards(end) - results.episode_rewards(1)) / ...
            length(results.episode_rewards);
    end
end

function metrics = calculate_qlearning_metrics(results, config)
    % CALCULATE_QLEARNING_METRICS - Calculates Q-Learning-specific performance metrics
    
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
    
    % Q-Learning-specific metrics
    metrics.qlearning.convergence_episodes = results.convergence_episode;
    metrics.qlearning.final_reward = results.convergence_analysis.final_reward;
    metrics.qlearning.execution_time = results.execution_time;
    metrics.qlearning.learning_efficiency = results.convergence_analysis.learning_efficiency;
    metrics.qlearning.avg_episode_reward = mean(results.episode_rewards);
    metrics.qlearning.reward_std = std(results.episode_rewards);
end

function display_qlearning_results(results, metrics)
    % DISPLAY_QLEARNING_RESULTS - Displays Q-Learning optimization results
    
    fprintf('\n=== Q-Learning Optimization Results ===\n');
    fprintf('Best Fitness: %.4f\n', results.best_fitness);
    fprintf('Allocation: [%s]\n', num2str(results.allocation));
    fprintf('Execution Time: %.4f seconds\n', results.execution_time);
    fprintf('Convergence Episodes: %d\n', results.convergence_episode);
    
    if results.convergence_analysis.converged
        fprintf('Final Reward: %.4f\n', results.convergence_analysis.final_reward);
        fprintf('Learning Efficiency: %.4f\n', results.convergence_analysis.learning_efficiency);
    end
    
    fprintf('Average Episode Reward: %.4f ± %.4f\n', ...
            metrics.qlearning.avg_episode_reward, metrics.qlearning.reward_std);
    fprintf('Overall Throughput: %.4f\n', sum([metrics.throughput.eMBB, metrics.throughput.URLLC, metrics.throughput.mMTC]));
    fprintf('Overall Latency: %.4f\n', mean([metrics.latency.eMBB, metrics.latency.URLLC, metrics.latency.mMTC]));
    fprintf('Utilization: %.4f\n', metrics.efficiency.utilization_efficiency);
    fprintf('Fairness Index: %.4f\n', metrics.fairness.jains_index);
    fprintf('QoS Satisfaction: %.4f\n', metrics.qos_satisfaction.overall);
    fprintf('=====================================\n\n');
end 