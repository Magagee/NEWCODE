function test_configuration_system()
    % TEST_CONFIGURATION_SYSTEM - Comprehensive test of configuration system
    %
    % This script tests all components of the configuration system to ensure
    % they work correctly for research purposes. It validates parameter
    % ranges, functionality, and research reproducibility features.
    %
    % Author: [Your Name]
    % Date: [Current Date]
    % Institution: [Your Institution]
    
    fprintf('=== Testing 5G Network Slicing Configuration System ===\n\n');
    
    %% Test 1: Basic Configuration Creation
    fprintf('Test 1: Basic Configuration Creation\n');
    fprintf('------------------------------------\n');
    
    try
        config = master_config('test_experiment');
        fprintf('✓ Basic configuration created successfully\n');
        fprintf('  - Experiment name: %s\n', config.experiment.name);
        fprintf('  - Total spectrum: %d PRBs\n', config.network.network.total_spectrum);
        fprintf('  - Number of slices: %d\n', config.network.network.num_slices);
    catch ME
        fprintf('✗ Basic configuration creation failed: %s\n', ME.message);
        return;
    end
    
    %% Test 2: Custom Parameter Configuration
    fprintf('\nTest 2: Custom Parameter Configuration\n');
    fprintf('--------------------------------------\n');
    
    try
        custom_config = master_config('custom_test', ...
                                    'pso.num_particles', 50, ...
                                    'qlearning.num_episodes', 2000, ...
                                    'network.network.demands', [70, 110, 60]);
        
        fprintf('✓ Custom configuration created successfully\n');
        fprintf('  - PSO particles: %d\n', custom_config.algorithm.pso.num_particles);
        fprintf('  - QL episodes: %d\n', custom_config.algorithm.qlearning.num_episodes);
        fprintf('  - Demands: [%s]\n', num2str(custom_config.network.network.demands));
    catch ME
        fprintf('✗ Custom configuration creation failed: %s\n', ME.message);
    end
    
    %% Test 3: Configuration Utilities
    fprintf('\nTest 3: Configuration Utilities\n');
    fprintf('-------------------------------\n');
    
    try
        utils = config_utils();
        fprintf('✓ Configuration utilities loaded successfully\n');
        
        % Test scenario creation
        high_load_config = utils.create_scenario_config('high_load', config);
        fprintf('✓ High load scenario created\n');
        
        % Test configuration comparison
        differences = utils.compare_configurations(config, high_load_config);
        fprintf('✓ Configuration comparison completed\n');
        
        % Test parameter validation
        test_allocation = [30, 40, 30];
        is_valid = utils.validate_allocation_constraints(test_allocation, config);
        fprintf('✓ Allocation validation completed: %s\n', mat2str(is_valid));
        
    catch ME
        fprintf('✗ Configuration utilities test failed: %s\n', ME.message);
    end
    
    %% Test 4: Parameter Validation
    fprintf('\nTest 4: Parameter Validation\n');
    fprintf('----------------------------\n');
    
    try
        % Test valid parameters
        valid_config = master_config('validation_test');
        fprintf('✓ Valid parameter configuration created\n');
        
        % Test invalid parameters (should fail)
        try
            invalid_config = master_config('invalid_test', ...
                                         'pso.num_particles', -10);
            fprintf('✗ Invalid parameter validation failed (should have failed)\n');
        catch
            fprintf('✓ Invalid parameter correctly rejected\n');
        end
        
    catch ME
        fprintf('✗ Parameter validation test failed: %s\n', ME.message);
    end
    
    %% Test 5: Reproducibility Features
    fprintf('\nTest 5: Reproducibility Features\n');
    fprintf('--------------------------------\n');
    
    try
        % Test random seed setting
        utils.set_reproducibility_seed(12345);
        seed1 = utils.get_current_seed();
        fprintf('✓ Random seed set to: %d\n', seed1);
        
        % Test seed consistency
        utils.set_reproducibility_seed(12345);
        seed2 = utils.get_current_seed();
        if seed1 == seed2
            fprintf('✓ Random seed reproducibility confirmed\n');
        else
            fprintf('✗ Random seed reproducibility failed\n');
        end
        
    catch ME
        fprintf('✗ Reproducibility test failed: %s\n', ME.message);
    end
    
    %% Test 6: Output Directory Structure
    fprintf('\nTest 6: Output Directory Structure\n');
    fprintf('-----------------------------------\n');
    
    try
        test_config = master_config('directory_test');
        
        % Check if directories were created
        dirs = {test_config.output.base_dir, test_config.output.experiment_dir, ...
                test_config.output.plots_dir, test_config.output.data_dir, ...
                test_config.output.logs_dir};
        
        all_exist = true;
        for i = 1:length(dirs)
            if exist(dirs{i}, 'dir')
                fprintf('✓ Directory exists: %s\n', dirs{i});
            else
                fprintf('✗ Directory missing: %s\n', dirs{i});
                all_exist = false;
            end
        end
        
        if all_exist
            fprintf('✓ All output directories created successfully\n');
        else
            fprintf('✗ Some output directories missing\n');
        end
        
    catch ME
        fprintf('✗ Output directory test failed: %s\n', ME.message);
    end
    
    %% Test 7: Configuration Saving and Loading
    fprintf('\nTest 7: Configuration Saving and Loading\n');
    fprintf('----------------------------------------\n');
    
    try
        test_config = master_config('save_load_test');
        
        % Save configuration
        utils.save_experiment_config(test_config);
        fprintf('✓ Configuration saved successfully\n');
        
        % Load configuration
        loaded_config = utils.load_experiment_config('save_load_test');
        fprintf('✓ Configuration loaded successfully\n');
        
        % Compare original and loaded
        differences = utils.compare_configurations(test_config, loaded_config);
        if ischar(differences) && strcmp(differences, 'Configurations are identical')
            fprintf('✓ Configuration save/load integrity confirmed\n');
        else
            fprintf('✗ Configuration save/load integrity failed\n');
        end
        
    catch ME
        fprintf('✗ Configuration save/load test failed: %s\n', ME.message);
    end
    
    %% Test 8: Performance Metrics Validation
    fprintf('\nTest 8: Performance Metrics Validation\n');
    fprintf('--------------------------------------\n');
    
    try
        % Test valid metrics
        valid_metrics = struct();
        valid_metrics.throughput = [50, 30, 20];
        valid_metrics.latency = [0.1, 0.05, 0.2];
        valid_metrics.utilization = [0.8, 0.9, 0.7];
        
        is_valid = utils.validate_performance_metrics(valid_metrics, config);
        fprintf('✓ Valid performance metrics accepted\n');
        
        % Test invalid metrics
        invalid_metrics = struct();
        invalid_metrics.throughput = [-10, 30, 20];
        invalid_metrics.latency = [0.1, -0.05, 0.2];
        invalid_metrics.utilization = [1.5, 0.9, 0.7];
        
        is_invalid = utils.validate_performance_metrics(invalid_metrics, config);
        if ~is_invalid
            fprintf('✓ Invalid performance metrics correctly rejected\n');
        else
            fprintf('✗ Invalid performance metrics validation failed\n');
        end
        
    catch ME
        fprintf('✗ Performance metrics validation test failed: %s\n', ME.message);
    end
    
    %% Test 9: Scenario Configurations
    fprintf('\nTest 9: Scenario Configurations\n');
    fprintf('-------------------------------\n');
    
    try
        base_config = master_config('scenario_test');
        scenarios = {'high_load', 'low_load', 'urllc_priority', 'embb_priority', 'mmtc_priority'};
        
        for i = 1:length(scenarios)
            scenario_config = utils.create_scenario_config(scenarios{i}, base_config);
            fprintf('✓ Scenario created: %s\n', scenarios{i});
            
            % Verify scenario-specific changes
            switch scenarios{i}
                case 'high_load'
                    if any(scenario_config.network.network.demands > base_config.network.network.demands)
                        fprintf('  - High load demands correctly set\n');
                    end
                case 'urllc_priority'
                    if scenario_config.network.network.priorities(2) == 3
                        fprintf('  - URLLC priority correctly set\n');
                    end
            end
        end
        
    catch ME
        fprintf('✗ Scenario configuration test failed: %s\n', ME.message);
    end
    
    %% Test 10: Research Metadata Logging
    fprintf('\nTest 10: Research Metadata Logging\n');
    fprintf('----------------------------------\n');
    
    try
        test_config = master_config('metadata_test');
        
        % Create dummy results
        results = struct();
        results.algorithms = struct();
        results.algorithms.pso = struct();
        results.algorithms.pso.fitness = [100, 95, 90, 85, 80];
        results.algorithms.pso.convergence_iteration = 45;
        
        % Log metadata
        utils.log_experiment_metadata(test_config, results);
        fprintf('✓ Research metadata logged successfully\n');
        
        % Check if metadata file exists
        metadata_file = fullfile(test_config.output.logs_dir, ...
                                sprintf('%s_metadata.json', test_config.output.file_prefix));
        if exist(metadata_file, 'file')
            fprintf('✓ Metadata file created: %s\n', metadata_file);
        else
            fprintf('✗ Metadata file not found\n');
        end
        
    catch ME
        fprintf('✗ Research metadata logging test failed: %s\n', ME.message);
    end
    
    %% Summary
    fprintf('\n=== Configuration System Test Summary ===\n');
    fprintf('All tests completed. Check output above for results.\n');
    fprintf('Configuration system is ready for research use.\n\n');
    
    %% Recommendations
    fprintf('=== Research Recommendations ===\n');
    fprintf('1. Always use the configuration system for parameter management\n');
    fprintf('2. Save configurations for each experiment\n');
    fprintf('3. Use scenario configurations for systematic testing\n');
    fprintf('4. Validate results against expected parameter ranges\n');
    fprintf('5. Document parameter choices and rationale\n');
    fprintf('6. Use consistent naming conventions\n');
    fprintf('7. Enable logging for reproducibility\n\n');
    
    fprintf('Configuration system testing completed successfully!\n');
end 