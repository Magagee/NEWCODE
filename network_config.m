function config = network_config()
    % NETWORK_CONFIG - Centralized configuration for 5G Network Slicing Research
    % 
    % This function provides a centralized configuration system for 5G network
    % slicing resource allocation research. All network parameters are defined
    % here to ensure consistency across experiments and reproducibility.
    %
    % Returns:
    %   config: Structure containing all network configuration parameters
    %
    % Author: [Your Name]
    % Date: [Current Date]
    % Institution: [Your Institution]
    % Research: 5G Network Slicing Resource Allocation Optimization
    
    %% Network Infrastructure Parameters
    config.network = struct();
    
    % Total available spectrum (Physical Resource Blocks)
    config.network.total_spectrum = 100;  % PRBs
    
    % Number of network slices
    config.network.num_slices = 3;
    
    % Slice types and their characteristics
    config.network.slice_types = {'eMBB', 'URLLC', 'mMTC'};
    config.network.slice_indices = struct('eMBB', 1, 'URLLC', 2, 'mMTC', 3);
    
    %% Slice-Specific Parameters
    % Resource demands for each slice (PRBs)
    % Based on 3GPP specifications and research literature
    config.network.demands = [60, 100, 50];  % [eMBB, URLLC, mMTC]
    
    % Priority levels (higher number = higher priority)
    % URLLC has highest priority due to latency requirements
    config.network.priorities = [2, 3, 1];  % [eMBB, URLLC, mMTC]
    
    % Reserved resources per slice (minimum guaranteed allocation)
    config.network.reserved = [20, 10, 5];  % [eMBB, URLLC, mMTC]
    
    % Reserve percentages for priority-aware allocation
    config.network.reserve_percent = [0.3, 0.3, 0.2];  % [eMBB, URLLC, mMTC]
    
    %% QoS Requirements (Based on 3GPP Standards)
    config.qos = struct();
    
    % Latency requirements (milliseconds)
    config.qos.latency_thresholds = [10, 1, 100];  % [eMBB, URLLC, mMTC]
    
    % Throughput requirements (Mbps)
    config.qos.throughput_requirements = [100, 10, 1];  % [eMBB, URLLC, mMTC]
    
    % Reliability requirements (percentage)
    config.qos.reliability_requirements = [99.9, 99.999, 99.9];  % [eMBB, URLLC, mMTC]
    
    %% Performance Evaluation Weights
    config.performance = struct();
    
    % Multi-objective optimization weights
    config.performance.weights = struct();
    config.performance.weights.throughput = 0.6;    % eMBB focus
    config.performance.weights.latency = 0.3;       % URLLC focus  
    config.performance.weights.utilization = 0.1;   % mMTC focus
    
    % Reward function weights for Q-Learning
    config.performance.reward_weights = struct();
    config.performance.reward_weights.throughput = 0.7;
    config.performance.reward_weights.latency = 0.3;
    
    % Global latency threshold for reward calculation
    config.performance.latency_threshold = 50;  % ms
    
    %% Validation
    validate_network_config(config);
end

function validate_network_config(config)
    % VALIDATE_NETWORK_CONFIG - Validates configuration parameters
    %
    % Ensures all configuration parameters are within valid ranges
    % and consistent with each other.
    
    % Check basic constraints
    assert(config.network.total_spectrum > 0, 'Total spectrum must be positive');
    assert(config.network.num_slices == 3, 'Must have exactly 3 slices for eMBB, URLLC, mMTC');
    assert(length(config.network.demands) == 3, 'Must have 3 demand values');
    assert(length(config.network.priorities) == 3, 'Must have 3 priority values');
    
    % Check demand constraints
    assert(all(config.network.demands >= 0), 'All demands must be non-negative');
    assert(sum(config.network.demands) <= config.network.total_spectrum * 2, ...
           'Total demands should not exceed 2x total spectrum');
    
    % Check priority constraints
    assert(all(config.network.priorities >= 1), 'All priorities must be >= 1');
    assert(length(unique(config.network.priorities)) == 3, 'Priorities must be unique');
    
    % Check reserve constraints
    assert(all(config.network.reserve_percent >= 0) && all(config.network.reserve_percent <= 1), ...
           'Reserve percentages must be between 0 and 1');
    assert(sum(config.network.reserve_percent) <= 1, 'Total reserve percentage cannot exceed 100%');
    
    % Check QoS constraints
    assert(all(config.qos.latency_thresholds > 0), 'Latency thresholds must be positive');
    assert(all(config.qos.throughput_requirements > 0), 'Throughput requirements must be positive');
    assert(all(config.qos.reliability_requirements >= 0) && all(config.qos.reliability_requirements <= 100), ...
           'Reliability requirements must be between 0 and 100');
    
    % Check performance weights
    weights = config.performance.weights;
    total_weight = weights.throughput + weights.latency + weights.utilization;
    assert(abs(total_weight - 1.0) < 1e-6, 'Performance weights must sum to 1.0');
    
    reward_weights = config.performance.reward_weights;
    total_reward_weight = reward_weights.throughput + reward_weights.latency;
    assert(abs(total_reward_weight - 1.0) < 1e-6, 'Reward weights must sum to 1.0');
    
    fprintf('✓ Network configuration validated successfully\n');
end 