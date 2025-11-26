function config = master_config(experiment_name, varargin)
    % MASTER_CONFIG - Master configuration manager for 5G Network Slicing Research
    % 
    % This function provides a unified configuration system that combines
    % network and algorithm parameters, manages experiments, and ensures
    % research reproducibility.
    %
    % Inputs:
    %   experiment_name: String identifier for the experiment
    %   varargin: Optional parameter-value pairs for custom configuration
    %
    % Returns:
    %   config: Complete configuration structure for the experiment
    %
    % Usage:
    %   config = master_config('baseline_experiment');
    %   config = master_config('sensitivity_analysis', 'pso.num_particles', 50);
    %
    % Author: [Your Name]
    % Date: [Current Date]
    % Institution: [Your Institution]
    % Research: 5G Network Slicing Resource Allocation Optimization
    
    %% Initialize configuration
    config = struct();
    
    % Load base configurations
    config.network = network_config();
    config.algorithm = algorithm_config();
    
    %% Experiment Metadata
    config.experiment = struct();
    config.experiment.name = experiment_name;
    config.experiment.timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    config.experiment.description = '';
    config.experiment.version = '1.0';
    
    % Experiment types for different research scenarios
    experiment_types = {'baseline', 'sensitivity_analysis', 'comparison', ...
                       'parameter_optimization', 'scalability_test'};
    
    if contains(lower(experiment_name), experiment_types)
        config.experiment.type = lower(experiment_name);
    else
        config.experiment.type = 'custom';
    end
    
    %% Output and Logging Configuration
    config.output = struct();
    
    % Directory structure for research outputs
    config.output.base_dir = 'results';
    config.output.experiment_dir = fullfile(config.output.base_dir, experiment_name);
    config.output.plots_dir = fullfile(config.output.experiment_dir, 'plots');
    config.output.data_dir = fullfile(config.output.experiment_dir, 'data');
    config.output.logs_dir = fullfile(config.output.experiment_dir, 'logs');
    
    % File naming conventions
    config.output.file_prefix = sprintf('%s_%s', experiment_name, config.experiment.timestamp);
    config.output.save_format = 'mat';  % 'mat', 'csv', 'json'
    
    % Logging configuration
    config.output.log_level = 'INFO';  % 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    config.output.save_logs = true;
    config.output.console_output = true;
    
    %% Performance Evaluation Configuration
    config.evaluation = struct();
    
    % Metrics to evaluate
    config.evaluation.metrics = {'throughput', 'latency', 'utilization', ...
                                'fairness', 'efficiency', 'qos_satisfaction'};
    
    % Statistical analysis
    config.evaluation.statistical_tests = {'t_test', 'wilcoxon', 'anova'};
    config.evaluation.confidence_level = 0.95;
    config.evaluation.num_runs = 30;  % Number of independent runs for statistical significance
    
    % Performance thresholds
    config.evaluation.throughput_threshold = 0.8;  % 80% of demand
    config.evaluation.latency_threshold = 1.2;     % 120% of requirement
    config.evaluation.utilization_threshold = 0.9; % 90% utilization
    
    %% Research Validation Configuration
    config.validation = struct();
    
    % Cross-validation parameters
    config.validation.use_cross_validation = true;
    config.validation.k_folds = 5;
    config.validation.test_size = 0.2;
    
    % Robustness testing
    config.validation.noise_levels = [0, 0.05, 0.1, 0.15, 0.2];  % 0-20% noise
    config.validation.demand_variations = [0.8, 0.9, 1.0, 1.1, 1.2];  % ±20% demand variation
    
    % Convergence analysis
    config.validation.convergence_analysis = true;
    config.validation.convergence_window = 10;
    
    %% Apply Custom Parameters
    if ~isempty(varargin)
        config = apply_custom_parameters(config, varargin);
    end
    
    %% Initialize Output Directories
    create_output_directories(config);
    
    %% Set Random Seed for Reproducibility
    rng(config.algorithm.general.random_seed);
    
    %% Final Validation
    validate_master_config(config);
    
    %% Log Configuration
    log_configuration(config);
    
    fprintf('✓ Master configuration initialized for experiment: %s\n', experiment_name);
end

function config = apply_custom_parameters(config, custom_params)
    % APPLY_CUSTOM_PARAMETERS - Applies custom parameter overrides
    
    for i = 1:2:length(custom_params)
        if mod(i, 2) == 1 && i+1 <= length(custom_params)
            param_path = custom_params{i};
            param_value = custom_params{i+1};
            
            % Parse parameter path (e.g., 'pso.num_particles')
            path_parts = strsplit(param_path, '.');
            
            % Navigate to the correct location in config
            current = config;
            for j = 1:length(path_parts)-1
                if isfield(current, path_parts{j})
                    current = current.(path_parts{j});
                else
                    warning('Parameter path not found: %s', param_path);
                    break;
                end
            end
            
            % Set the parameter value
            if isfield(current, path_parts{end})
                current.(path_parts{end}) = param_value;
                fprintf('Applied custom parameter: %s = %s\n', param_path, mat2str(param_value));
            else
                warning('Parameter not found: %s', param_path);
            end
        end
    end
end

function create_output_directories(config)
    % CREATE_OUTPUT_DIRECTORIES - Creates necessary output directories
    
    dirs = {config.output.base_dir, config.output.experiment_dir, ...
            config.output.plots_dir, config.output.data_dir, config.output.logs_dir};
    
    for i = 1:length(dirs)
        if ~exist(dirs{i}, 'dir')
            mkdir(dirs{i});
            fprintf('Created directory: %s\n', dirs{i});
        end
    end
end

function validate_master_config(config)
    % VALIDATE_MASTER_CONFIG - Validates the complete configuration
    
    % Check experiment metadata
    assert(~isempty(config.experiment.name), 'Experiment name cannot be empty');
    assert(~isempty(config.experiment.timestamp), 'Experiment timestamp cannot be empty');
    
    % Check output configuration
    assert(~isempty(config.output.base_dir), 'Base output directory cannot be empty');
    assert(config.evaluation.num_runs > 0, 'Number of runs must be positive');
    assert(config.evaluation.confidence_level > 0 && config.evaluation.confidence_level < 1, ...
           'Confidence level must be between 0 and 1');
    
    % Check validation configuration
    if config.validation.use_cross_validation
        assert(config.validation.k_folds > 1, 'K-folds must be greater than 1');
        assert(config.validation.test_size > 0 && config.validation.test_size < 1, ...
               'Test size must be between 0 and 1');
    end
    
    fprintf('✓ Master configuration validation completed\n');
end

function log_configuration(config)
    % LOG_CONFIGURATION - Logs the configuration for research documentation
    
    log_file = fullfile(config.output.logs_dir, ...
                       sprintf('%s_config.log', config.output.file_prefix));
    
    fid = fopen(log_file, 'w');
    if fid ~= -1
        fprintf(fid, '=== 5G Network Slicing Research Configuration ===\n');
        fprintf(fid, 'Experiment: %s\n', config.experiment.name);
        fprintf(fid, 'Timestamp: %s\n', config.experiment.timestamp);
        fprintf(fid, 'Type: %s\n', config.experiment.type);
        fprintf(fid, 'Version: %s\n', config.experiment.version);
        fprintf(fid, '\n');
        
        % Log network configuration
        fprintf(fid, '--- Network Configuration ---\n');
        fprintf(fid, 'Total Spectrum: %d PRBs\n', config.network.network.total_spectrum);
        fprintf(fid, 'Demands: [%s] PRBs\n', num2str(config.network.network.demands));
        fprintf(fid, 'Priorities: [%s]\n', num2str(config.network.network.priorities));
        fprintf(fid, '\n');
        
        % Log algorithm configuration
        fprintf(fid, '--- Algorithm Configuration ---\n');
        fprintf(fid, 'PSO Particles: %d\n', config.algorithm.pso.num_particles);
        fprintf(fid, 'PSO Iterations: %d\n', config.algorithm.pso.num_iterations);
        fprintf(fid, 'QL Episodes: %d\n', config.algorithm.qlearning.num_episodes);
        fprintf(fid, 'QL Actions: %d\n', config.algorithm.qlearning.num_actions);
        fprintf(fid, '\n');
        
        % Log evaluation configuration
        fprintf(fid, '--- Evaluation Configuration ---\n');
        fprintf(fid, 'Number of Runs: %d\n', config.evaluation.num_runs);
        fprintf(fid, 'Confidence Level: %.2f\n', config.evaluation.confidence_level);
        fprintf(fid, 'Metrics: %s\n', strjoin(config.evaluation.metrics, ', '));
        fprintf(fid, '\n');
        
        fclose(fid);
        fprintf('Configuration logged to: %s\n', log_file);
    end
end 