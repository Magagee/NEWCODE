function utils = config_utils()
    % CONFIG_UTILS - Configuration utility functions for research
    %
    % This file provides utility functions for configuration management,
    % parameter validation, and research reproducibility in 5G network
    % slicing optimization research.
    
    %% Configuration Management Functions
    
    function config = load_experiment_config(experiment_name)
        % LOAD_EXPERIMENT_CONFIG - Loads configuration for a specific experiment
        %
        % Inputs:
        %   experiment_name: Name of the experiment
        % Returns:
        %   config: Configuration structure
        
        config_file = fullfile('results', experiment_name, 'logs', ...
                              sprintf('%s_config.mat', experiment_name));
        
        if exist(config_file, 'file')
            load(config_file, 'config');
            fprintf('Loaded configuration from: %s\n', config_file);
        else
            warning('Configuration file not found: %s\nCreating new configuration...', config_file);
            config = master_config(experiment_name);
        end
    end
    
    function save_experiment_config(config)
        % SAVE_EXPERIMENT_CONFIG - Saves configuration for reproducibility
        %
        % Inputs:
        %   config: Configuration structure to save
        
        config_file = fullfile(config.output.logs_dir, ...
                              sprintf('%s_config.mat', config.output.file_prefix));
        
        save(config_file, 'config');
        fprintf('Configuration saved to: %s\n', config_file);
    end
    
    function config = create_scenario_config(scenario_name, base_config)
        % CREATE_SCENARIO_CONFIG - Creates configuration for specific scenarios
        %
        % Inputs:
        %   scenario_name: Name of the scenario
        %   base_config: Base configuration to modify
        % Returns:
        %   config: Modified configuration for the scenario
        
        config = base_config;
        
        switch lower(scenario_name)
            case 'high_load'
                config.network.network.demands = [80, 120, 70];
                config.network.network.priorities = [1, 3, 2];
                
            case 'low_load'
                config.network.network.demands = [30, 50, 20];
                config.network.network.priorities = [2, 3, 1];
                
            case 'urllc_priority'
                config.network.network.priorities = [1, 3, 2];
                config.network.performance.weights.latency = 0.5;
                config.network.performance.weights.throughput = 0.3;
                config.network.performance.weights.utilization = 0.2;
                
            case 'embb_priority'
                config.network.network.priorities = [3, 1, 2];
                config.network.performance.weights.throughput = 0.7;
                config.network.performance.weights.latency = 0.2;
                config.network.performance.weights.utilization = 0.1;
                
            case 'mmtc_priority'
                config.network.network.priorities = [1, 2, 3];
                config.network.performance.weights.utilization = 0.6;
                config.network.performance.weights.throughput = 0.2;
                config.network.performance.weights.latency = 0.2;
                
            otherwise
                warning('Unknown scenario: %s. Using base configuration.', scenario_name);
        end
        
        config.experiment.name = sprintf('%s_%s', base_config.experiment.name, scenario_name);
        config.experiment.scenario = scenario_name;
    end
    
    %% Parameter Validation Functions
    
    function is_valid = validate_allocation_constraints(allocation, config)
        % VALIDATE_ALLOCATION_CONSTRAINTS - Validates resource allocation
        %
        % Inputs:
        %   allocation: Resource allocation vector
        %   config: Configuration structure
        % Returns:
        %   is_valid: Boolean indicating if allocation is valid
        
        is_valid = true;
        
        % Check total allocation constraint
        if sum(allocation) > config.network.network.total_spectrum
            warning('Total allocation exceeds available spectrum');
            is_valid = false;
        end
        
        % Check demand constraints
        for i = 1:length(allocation)
            if allocation(i) > config.network.network.demands(i)
                warning('Allocation for slice %d exceeds demand', i);
                is_valid = false;
            end
        end
        
        % Check non-negativity
        if any(allocation < 0)
            warning('Allocation contains negative values');
            is_valid = false;
        end
    end
    
    function fixed_allocation = fix_allocation_constraints(allocation, config)
        % FIX_ALLOCATION_CONSTRAINTS - Fixes allocation that violates constraints
        %
        % Inputs:
        %   allocation: Resource allocation vector
        %   config: Configuration structure
        % Returns:
        %   fixed_allocation: Corrected allocation vector
        
        network_config = config.network.network;
        total_spectrum = network_config.total_spectrum;
        demands = network_config.demands;
        
        % Ensure allocation is a row vector
        if iscolumn(allocation)
            allocation = allocation';
        end
        
        % Ensure non-negativity
        fixed_allocation = max(allocation, 0);
        
        % Cap at demands
        fixed_allocation = min(fixed_allocation, demands);
        
        % Ensure minimum allocation per slice (at least 1 PRB)
        fixed_allocation = max(fixed_allocation, 1);
        
        % Ensure total doesn't exceed spectrum
        if sum(fixed_allocation) > total_spectrum
            fixed_allocation = fixed_allocation / sum(fixed_allocation) * total_spectrum;
            fixed_allocation = max(round(fixed_allocation), 1);
        end
        
        % Final validation and rounding
        fixed_allocation = min(fixed_allocation, demands);
        fixed_allocation = max(round(fixed_allocation), 1);
        
        % Ensure total is exactly equal to total spectrum
        if sum(fixed_allocation) ~= total_spectrum
            diff = total_spectrum - sum(fixed_allocation);
            if diff > 0
                % Add to slice with highest priority
                [~, best_slice] = max(network_config.priorities);
                fixed_allocation(best_slice) = fixed_allocation(best_slice) + diff;
            else
                % Remove from slice with lowest priority
                [~, worst_slice] = min(network_config.priorities);
                fixed_allocation(worst_slice) = max(fixed_allocation(worst_slice) + diff, 1);
            end
        end
        
        fprintf('Fixed allocation: [%.1f %.1f %.1f], Total: %.1f\n', ...
                fixed_allocation(1), fixed_allocation(2), fixed_allocation(3), sum(fixed_allocation));
    end
    
    function is_valid = validate_performance_metrics(metrics, config)
        % VALIDATE_PERFORMANCE_METRICS - Validates performance metrics
        %
        % Inputs:
        %   metrics: Performance metrics structure
        %   config: Configuration structure
        % Returns:
        %   is_valid: Boolean indicating if metrics are valid
        
        is_valid = true;
        
        % Check throughput metrics
        if isfield(metrics, 'throughput')
            if any(metrics.throughput < 0)
                warning('Throughput contains negative values');
                is_valid = false;
            end
        end
        
        % Check latency metrics
        if isfield(metrics, 'latency')
            if any(metrics.latency <= 0)
                warning('Latency contains non-positive values');
                is_valid = false;
            end
        end
        
        % Check utilization metrics
        if isfield(metrics, 'utilization')
            if any(metrics.utilization < 0) || any(metrics.utilization > 1)
                warning('Utilization values must be between 0 and 1');
                is_valid = false;
            end
        end
    end
    
    %% Research Reproducibility Functions
    
    function set_reproducibility_seed(seed)
        % SET_REPRODUCIBILITY_SEED - Sets random seed for reproducibility
        %
        % Inputs:
        %   seed: Random seed value
        
        rng(seed);
        fprintf('Random seed set to: %d\n', seed);
    end
    
    function seed = get_current_seed()
        % GET_CURRENT_SEED - Gets current random seed
        %
        % Returns:
        %   seed: Current random seed value
        
        seed = rng;
        seed = seed.Seed;
    end
    
    function log_experiment_metadata(config, results)
        % LOG_EXPERIMENT_METADATA - Logs experiment metadata for research
        %
        % Inputs:
        %   config: Configuration structure
        %   results: Experiment results
        
        metadata_file = fullfile(config.output.logs_dir, ...
                                sprintf('%s_metadata.json', config.output.file_prefix));
        
        metadata = struct();
        metadata.experiment = config.experiment;
        metadata.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        metadata.random_seed = get_current_seed();
        metadata.matlab_version = version;
        metadata.results_summary = struct();
        
        % Add results summary
        if isfield(results, 'algorithms')
            algorithms = fieldnames(results.algorithms);
            for i = 1:length(algorithms)
                alg = algorithms{i};
                if isfield(results.algorithms, alg)
                    metadata.results_summary.(alg) = struct();
                    if isfield(results.algorithms.(alg), 'fitness')
                        metadata.results_summary.(alg).best_fitness = ...
                            min(results.algorithms.(alg).fitness);
                    end
                    if isfield(results.algorithms.(alg), 'convergence_iteration')
                        metadata.results_summary.(alg).convergence_iteration = ...
                            results.algorithms.(alg).convergence_iteration;
                    end
                end
            end
        end
        
        % Write JSON file
        json_str = jsonencode(metadata, 'PrettyPrint', true);
        fid = fopen(metadata_file, 'w');
        if fid ~= -1
            fprintf(fid, '%s', json_str);
            fclose(fid);
            fprintf('Experiment metadata logged to: %s\n', metadata_file);
        end
    end
    
    %% Configuration Comparison Functions
    
    function differences = compare_configurations(config1, config2)
        % COMPARE_CONFIGURATIONS - Compares two configurations
        %
        % Inputs:
        %   config1: First configuration
        %   config2: Second configuration
        % Returns:
        %   differences: Structure containing differences
        
        differences = struct();
        
        % Compare network parameters
        if ~isequal(config1.network, config2.network)
            differences.network = 'Different network configurations';
        end
        
        % Compare algorithm parameters
        if ~isequal(config1.algorithm, config2.algorithm)
            differences.algorithm = 'Different algorithm configurations';
        end
        
        % Compare evaluation parameters
        if ~isequal(config1.evaluation, config2.evaluation)
            differences.evaluation = 'Different evaluation configurations';
        end
        
        if isempty(fieldnames(differences))
            differences = 'Configurations are identical';
        end
    end
    
    function config = merge_configurations(config1, config2, merge_strategy)
        % MERGE_CONFIGURATIONS - Merges two configurations
        %
        % Inputs:
        %   config1: First configuration
        %   config2: Second configuration
        %   merge_strategy: 'config1_priority', 'config2_priority', 'selective'
        % Returns:
        %   config: Merged configuration
        
        config = config1;
        
        switch lower(merge_strategy)
            case 'config2_priority'
                config = config2;
                
            case 'selective'
                % Merge specific sections
                if isfield(config2, 'network')
                    config.network = config2.network;
                end
                if isfield(config2, 'algorithm')
                    config.algorithm = config2.algorithm;
                end
                
            otherwise % config1_priority
                % Keep config1 as base, no changes needed
        end
        
        fprintf('Configurations merged using strategy: %s\n', merge_strategy);
    end
    
    %% Export functions
    utils.load_experiment_config = @load_experiment_config;
    utils.save_experiment_config = @save_experiment_config;
    utils.create_scenario_config = @create_scenario_config;
    utils.validate_allocation_constraints = @validate_allocation_constraints;
    utils.validate_performance_metrics = @validate_performance_metrics;
    utils.set_reproducibility_seed = @set_reproducibility_seed;
    utils.get_current_seed = @get_current_seed;
    utils.log_experiment_metadata = @log_experiment_metadata;
    utils.compare_configurations = @compare_configurations;
    utils.merge_configurations = @merge_configurations;
end 