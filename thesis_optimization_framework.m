function [results, metrics] = thesis_optimization_framework(config, traffic_scenario)
    % THESIS_OPTIMIZATION_FRAMEWORK - PhD Thesis Optimization Framework
    %
    % This function implements the specific optimization objectives for the PhD thesis:
    % 1. Maximize eMBB throughput
    % 2. Minimize URLLC latency  
    % 3. Maximize mMTC utilization efficiency
    % 4. Ensure slice isolation
    % 5. Support different traffic scenarios
    %
    % Inputs:
    %   config: Configuration structure
    %   traffic_scenario: 'low', 'medium', or 'high'
    %
    % Returns:
    %   results: Structure containing optimization results
    %   metrics: Structure containing performance metrics
    %
    % Author: [Your Name]
    % Date: [Current Date]
    % Institution: [Your Institution]
    % Research: 5G Network Slicing Resource Allocation Optimization
    
    %% Input Validation
    validate_thesis_inputs(config, traffic_scenario);
    
    %% Configure Traffic Scenario
    config = configure_traffic_scenario(config, traffic_scenario);
    
    %% Initialize Results Structure
    results = struct();
    results.traffic_scenario = traffic_scenario;
    results.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    results.config = config;
    
    %% Run All Optimization Algorithms
    algorithms = {'FIXED', 'PSO', 'QL', 'HPQL'};
    
    for i = 1:length(algorithms)
        algorithm = algorithms{i};
        fprintf('Running %s optimization for %s traffic scenario...\n', algorithm, traffic_scenario);
        
        try
            [results.(algorithm), metrics.(algorithm)] = optimization_interface(config, algorithm);
            results.(algorithm).traffic_scenario = traffic_scenario;
        catch ME
            warning('Algorithm %s failed: %s', algorithm, ME.message);
            results.(algorithm) = struct('failed', true, 'error', ME.message);
            metrics.(algorithm) = struct();
        end
    end
    
    %% Calculate Thesis-Specific Metrics
    results = calculate_thesis_metrics(results, config);
    
    %% Validate Slice Isolation
    results = validate_slice_isolation(results, config);
    
    %% Final Validation and Fixing
    results = final_validation_and_fixing(results, config);
    
    %% Display Results
    display_thesis_results(results, metrics, traffic_scenario);
    
    fprintf('Thesis optimization framework completed for %s traffic scenario!\n', traffic_scenario);
end

function validate_thesis_inputs(config, traffic_scenario)
    % VALIDATE_THESIS_INPUTS - Validates thesis-specific inputs
    
    if ~isstruct(config) || ~isfield(config, 'network')
        error('Invalid configuration structure');
    end
    
    valid_scenarios = {'low', 'medium', 'high'};
    if ~ismember(lower(traffic_scenario), valid_scenarios)
        error('Invalid traffic scenario. Must be: %s', strjoin(valid_scenarios, ', '));
    end
end

function config = configure_traffic_scenario(config, traffic_scenario)
    % CONFIGURE_TRAFFIC_SCENARIO - Configures network for specific traffic scenario
    
    base_demands = config.network.network.demands;
    
    switch lower(traffic_scenario)
        case 'low'
            % Low traffic: 50% of base demands
            config.network.network.demands = base_demands * 0.5;
            config.network.network.total_spectrum = 50; % Reduced spectrum
            
        case 'medium'
            % Medium traffic: 100% of base demands
            config.network.network.demands = base_demands;
            config.network.network.total_spectrum = 100;
            
        case 'high'
            % High traffic: 150% of base demands
            config.network.network.demands = base_demands * 1.5;
            config.network.network.total_spectrum = 150; % Increased spectrum
    end
    
    % Ensure demands don't exceed spectrum
    config.network.network.demands = min(config.network.network.demands, ...
                                       config.network.network.total_spectrum * 0.8);
    
    % Update QoS requirements based on traffic scenario
    config = update_qos_requirements(config, traffic_scenario);
end

function config = update_qos_requirements(config, traffic_scenario)
    % UPDATE_QOS_REQUIREMENTS - Updates QoS requirements based on traffic scenario
    
    switch lower(traffic_scenario)
        case 'low'
            % Relaxed QoS for low traffic
            config.qos.throughput_requirements = [50, 5, 0.5];  % [eMBB, URLLC, mMTC]
            config.qos.latency_thresholds = [20, 2, 200];       % [eMBB, URLLC, mMTC]
            
        case 'medium'
            % Standard QoS for medium traffic
            config.qos.throughput_requirements = [100, 10, 1];  % [eMBB, URLLC, mMTC]
            config.qos.latency_thresholds = [10, 1, 100];       % [eMBB, URLLC, mMTC]
            
        case 'high'
            % Strict QoS for high traffic
            config.qos.throughput_requirements = [150, 15, 1.5]; % [eMBB, URLLC, mMTC]
            config.qos.latency_thresholds = [5, 0.5, 50];        % [eMBB, URLLC, mMTC]
    end
end

function results = calculate_thesis_metrics(results, config)
    % CALCULATE_THESIS_METRICS - Calculates thesis-specific performance metrics
    
    algorithms = fieldnames(results);
    
    for i = 1:length(algorithms)
        alg = algorithms{i};
        
        if isfield(results.(alg), 'failed') && results.(alg).failed
            continue;
        end
        
        % Extract thesis-specific metrics
        if isfield(results.(alg), 'metrics')
            metrics = results.(alg).metrics;
            
            % eMBB Throughput
            if isfield(metrics, 'throughput') && isfield(metrics.throughput, 'eMBB')
                results.(alg).thesis_metrics.eMBB_throughput = metrics.throughput.eMBB;
            else
                results.(alg).thesis_metrics.eMBB_throughput = 0;
            end
            
            % URLLC Latency
            if isfield(metrics, 'latency') && isfield(metrics.latency, 'URLLC')
                results.(alg).thesis_metrics.URLLC_latency = metrics.latency.URLLC;
            else
                results.(alg).thesis_metrics.URLLC_latency = inf;
            end
            
            % mMTC Utilization Efficiency
            if isfield(metrics, 'utilization') && isfield(metrics.utilization, 'mMTC')
                results.(alg).thesis_metrics.mMTC_utilization = metrics.utilization.mMTC;
            else
                results.(alg).thesis_metrics.mMTC_utilization = 0;
            end
            
            % Overall Performance Index
            results.(alg).thesis_metrics.overall_performance = calculate_overall_performance_index(metrics, config);
            
            % Slice Isolation Score
            results.(alg).thesis_metrics.slice_isolation_score = calculate_slice_isolation_score(results.(alg).allocation, config);
        end
    end
end

function overall_performance = calculate_overall_performance_index(metrics, config)
    % CALCULATE_OVERALL_PERFORMANCE_INDEX - Calculates overall performance index
    
    % Normalize metrics to [0, 1] range
    eMBB_throughput_norm = min(metrics.throughput.eMBB / config.qos.throughput_requirements(1), 1);
    URLLC_latency_norm = max(0, 1 - (metrics.latency.URLLC / config.qos.latency_thresholds(2)));
    mMTC_utilization_norm = metrics.utilization.mMTC;
    
    % Weighted sum (thesis objectives)
    weights = [0.4, 0.4, 0.2]; % [eMBB, URLLC, mMTC]
    overall_performance = weights(1) * eMBB_throughput_norm + ...
                         weights(2) * URLLC_latency_norm + ...
                         weights(3) * mMTC_utilization_norm;
end

function isolation_score = calculate_slice_isolation_score(allocation, config)
    % CALCULATE_SLICE_ISOLATION_SCORE - Calculates slice isolation score
    
    % Slice isolation is measured by how well resources are allocated
    % according to priorities and demands
    
    priorities = config.network.network.priorities;
    demands = config.network.network.demands;
    total_spectrum = config.network.network.total_spectrum;
    
    % Calculate isolation score based on priority compliance
    priority_scores = zeros(1, length(priorities));
    
    for i = 1:length(priorities)
        % Higher priority slices should get proportionally more resources
        expected_ratio = priorities(i) / sum(priorities);
        actual_ratio = allocation(i) / total_spectrum;
        
        % Score based on how close actual is to expected
        priority_scores(i) = 1 - abs(actual_ratio - expected_ratio) / expected_ratio;
        priority_scores(i) = max(0, min(1, priority_scores(i))); % Clamp to [0, 1]
    end
    
    isolation_score = mean(priority_scores);
end

function results = validate_slice_isolation(results, config)
    % VALIDATE_SLICE_ISOLATION - Validates slice isolation requirements
    
    algorithms = fieldnames(results);
    
    for i = 1:length(algorithms)
        alg = algorithms{i};
        
        if isfield(results.(alg), 'failed') && results.(alg).failed
            continue;
        end
        
        allocation = results.(alg).allocation;
        
        % Check if allocation respects slice boundaries
        total_allocated = sum(allocation);
        total_spectrum = config.network.network.total_spectrum;
        
        % Validate total allocation
        if total_allocated > total_spectrum * 1.01 % Allow 1% tolerance
            warning('Algorithm %s: Total allocation exceeds spectrum', alg);
            results.(alg).isolation_valid = false;
        else
            results.(alg).isolation_valid = true;
        end
        
        % Check individual slice allocations
        demands = config.network.network.demands;
        for j = 1:length(allocation)
            if allocation(j) > demands(j) * 1.1 % Allow 10% tolerance
                warning('Algorithm %s: Slice %d allocation exceeds demand', alg, j);
                results.(alg).isolation_valid = false;
            end
        end
    end
end

function display_thesis_results(results, metrics, traffic_scenario)
    % DISPLAY_THESIS_RESULTS - Displays thesis-specific results
    
    fprintf('\n=== Thesis Optimization Results (%s Traffic) ===\n', upper(traffic_scenario));
    
    algorithms = fieldnames(results);
    valid_algorithms = {};
    
    % Collect valid algorithms
    for i = 1:length(algorithms)
        if ~isfield(results.(algorithms{i}), 'failed') || ~results.(algorithms{i}).failed
            valid_algorithms{end+1} = algorithms{i};
        end
    end
    
    % Display header
    fprintf('%-15s %-15s %-15s %-15s %-15s %-15s\n', ...
            'Algorithm', 'eMBB Throughput', 'URLLC Latency', 'mMTC Utilization', ...
            'Overall Perf.', 'Isolation Score');
    fprintf('%s\n', repmat('-', 1, 90));
    
    % Display results for each algorithm
    for i = 1:length(valid_algorithms)
        alg = valid_algorithms{i};
        
        if isfield(results.(alg), 'thesis_metrics')
            thesis_metrics = results.(alg).thesis_metrics;
            
            fprintf('%-15s %-15.2f %-15.4f %-15.2f %-15.3f %-15.3f\n', ...
                    get_algorithm_display_name(alg), ...
                    thesis_metrics.eMBB_throughput, ...
                    thesis_metrics.URLLC_latency, ...
                    thesis_metrics.mMTC_utilization * 100, ...
                    thesis_metrics.overall_performance, ...
                    thesis_metrics.slice_isolation_score);
        end
    end
    
    fprintf('%s\n', repmat('-', 1, 90));
    
    % Find best performing algorithm
    best_algorithm = find_best_algorithm(results, valid_algorithms);
    fprintf('Best Overall Performance: %s\n', get_algorithm_display_name(best_algorithm));
    
    fprintf('=====================================\n\n');
end

function best_algorithm = find_best_algorithm(results, valid_algorithms)
    % FIND_BEST_ALGORITHM - Finds the best performing algorithm
    
    best_score = -inf;
    best_algorithm = '';
    
    for i = 1:length(valid_algorithms)
        alg = valid_algorithms{i};
        
        if isfield(results.(alg), 'thesis_metrics')
            score = results.(alg).thesis_metrics.overall_performance;
            
            if score > best_score
                best_score = score;
                best_algorithm = alg;
            end
        end
    end
end

function display_name = get_algorithm_display_name(algorithm)
    % GET_ALGORITHM_DISPLAY_NAME - Returns display name for algorithm
    
    switch upper(algorithm)
        case 'PSO'
            display_name = 'PSO';
        case 'QL'
            display_name = 'Q-Learning';
        case 'HPQL'
            display_name = 'Hybrid PSO-QL';
        case 'FIXED'
            display_name = 'Fixed';
        case 'PRIORITY'
            display_name = 'Priority';
        otherwise
            display_name = algorithm;
    end
end

function run_complete_thesis_analysis()
    % RUN_COMPLETE_THESIS_ANALYSIS - Runs complete thesis analysis for all scenarios
    
    fprintf('=== Complete PhD Thesis Analysis ===\n');
    
    % Load configuration
    config = master_config('thesis_analysis');
    
    % Define traffic scenarios
    traffic_scenarios = {'low', 'medium', 'high'};
    
    % Run analysis for each scenario
    for i = 1:length(traffic_scenarios)
        scenario = traffic_scenarios{i};
        fprintf('\n--- Analyzing %s traffic scenario ---\n', scenario);
        
        [results.(scenario), metrics.(scenario)] = thesis_optimization_framework(config, scenario);
    end
    
    % Create comprehensive comparison
    create_thesis_comparison(results, traffic_scenarios);
    
    % Generate publication-ready visualizations
    thesis_visualization(config, results, true);
    
    fprintf('\nComplete thesis analysis completed successfully!\n');
end

function results = final_validation_and_fixing(results, config)
    % FINAL_VALIDATION_AND_FIXING - Final validation and fixing of all results
    
    fprintf('\n=== Final Validation and Fixing ===\n');
    
    algorithms = {'FIXED', 'PSO', 'QL', 'HPQL'};
    utils = config_utils();
    
    for i = 1:length(algorithms)
        alg = algorithms{i};
        
        if isfield(results, alg) && isfield(results.(alg), 'allocation')
            fprintf('Validating %s allocation...\n', alg);
            
            % Check if allocation exists and is valid
            if ~isempty(results.(alg).allocation)
                is_valid = utils.validate_allocation_constraints(results.(alg).allocation, config);
                
                if ~is_valid
                    fprintf('Fixing %s allocation constraints...\n', alg);
                    results.(alg).allocation = utils.fix_allocation_constraints(results.(alg).allocation, config);
                else
                    fprintf('%s allocation is valid: [%.1f %.1f %.1f], Total: %.1f\n', ...
                            alg, results.(alg).allocation(1), results.(alg).allocation(2), ...
                            results.(alg).allocation(3), sum(results.(alg).allocation));
                end
            else
                fprintf('Warning: %s has no allocation data\n', alg);
            end
        end
    end
    
    fprintf('Final validation completed!\n');
end

function create_thesis_comparison(results, traffic_scenarios)
    % CREATE_THESIS_COMPARISON - Creates comprehensive comparison across scenarios
    
    fprintf('\n=== Cross-Scenario Performance Comparison ===\n');
    
    algorithms = {'FIXED', 'PSO', 'QL', 'HPQL'};
    
    % Create comparison table
    fprintf('%-15s %-15s %-15s %-15s %-15s\n', ...
            'Algorithm', 'Low Traffic', 'Medium Traffic', 'High Traffic', 'Average');
    fprintf('%s\n', repmat('-', 1, 75));
    
    for i = 1:length(algorithms)
        alg = algorithms{i};
        scores = [];
        
        for j = 1:length(traffic_scenarios)
            scenario = traffic_scenarios{j};
            
            if isfield(results.(scenario), alg) && ...
               isfield(results.(scenario).(alg), 'thesis_metrics')
                score = results.(scenario).(alg).thesis_metrics.overall_performance;
                scores(j) = score;
                fprintf('%-15s %-15.3f', get_algorithm_display_name(alg), score);
            else
                scores(j) = 0;
                fprintf('%-15s %-15s', get_algorithm_display_name(alg), 'N/A');
            end
        end
        
        avg_score = mean(scores);
        fprintf('%-15.3f\n', avg_score);
    end
    
    fprintf('%s\n', repmat('-', 1, 75));
end 