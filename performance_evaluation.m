function [comparison_results, statistical_analysis] = performance_evaluation(config, algorithms, num_runs)
    % PERFORMANCE_EVALUATION - Comprehensive performance evaluation framework
    %
    % This function provides a research-grade framework for evaluating and
    % comparing optimization algorithms with statistical analysis, confidence
    % intervals, and comprehensive performance metrics.
    %
    % Inputs:
    %   config: Configuration structure from master_config()
    %   algorithms: Cell array of algorithm names to compare
    %   num_runs: Number of independent runs for statistical significance
    %
    % Returns:
    %   comparison_results: Structure containing comparison results
    %   statistical_analysis: Structure containing statistical analysis
    %
    % Usage:
    %   [results, stats] = performance_evaluation(config, {'PSO', 'QL', 'HPQL'}, 30);
    %
    % Author: [Your Name]
    % Date: [Current Date]
    % Institution: [Your Institution]
    % Research: 5G Network Slicing Resource Allocation Optimization
    
    %% Input Validation
    validate_evaluation_inputs(config, algorithms, num_runs);
    
    %% Initialize Results Structures
    comparison_results = initialize_comparison_results(config, algorithms, num_runs);
    statistical_analysis = initialize_statistical_analysis(algorithms);
    
    %% Run Multiple Independent Experiments
    fprintf('Starting performance evaluation with %d runs per algorithm...\n', num_runs);
    
    for run = 1:num_runs
        fprintf('Run %d/%d...\n', run, num_runs);
        
        % Set different random seed for each run
        config.algorithm.general.random_seed = config.algorithm.general.random_seed + run;
        
        for alg_idx = 1:length(algorithms)
            algorithm = algorithms{alg_idx};
            
            try
                % Run optimization
                [results, metrics] = optimization_interface(config, algorithm);
                
                % Store results
                comparison_results.runs(run).(algorithm) = results;
                comparison_results.metrics(run).(algorithm) = metrics;
                
                % Update running statistics
                update_running_statistics(comparison_results, statistical_analysis, ...
                                       algorithm, results, metrics, run);
                
            catch ME
                warning('Algorithm %s failed in run %d: %s', algorithm, run, ME.message);
                % Store failure information
                comparison_results.runs(run).(algorithm) = struct('failed', true, 'error', ME.message);
            end
        end
    end
    
    %% Perform Statistical Analysis
    fprintf('Performing statistical analysis...\n');
    statistical_analysis = perform_statistical_analysis(comparison_results, algorithms, config);
    
    %% Generate Comprehensive Reports
    fprintf('Generating evaluation reports...\n');
    generate_evaluation_reports(comparison_results, statistical_analysis, config, algorithms);
    
    %% Create Visualization
    create_evaluation_visualizations(comparison_results, statistical_analysis, config, algorithms);
    
    %% Display Summary
    display_evaluation_summary(comparison_results, statistical_analysis, algorithms);
    
    fprintf('Performance evaluation completed successfully!\n');
end

function validate_evaluation_inputs(config, algorithms, num_runs)
    % VALIDATE_EVALUATION_INPUTS - Validates evaluation inputs
    
    % Check configuration
    if ~isstruct(config) || ~isfield(config, 'network')
        error('Invalid configuration structure');
    end
    
    % Check algorithms
    if ~iscell(algorithms) || isempty(algorithms)
        error('Algorithms must be a non-empty cell array');
    end
    
    valid_algorithms = {'PSO', 'QL', 'HPQL', 'PRIORITY', 'FIXED', 'RESERVATION'};
    for i = 1:length(algorithms)
        if ~ismember(upper(algorithms{i}), valid_algorithms)
            error('Invalid algorithm: %s. Valid options: %s', ...
                  algorithms{i}, strjoin(valid_algorithms, ', '));
        end
    end
    
    % Check number of runs
    if ~isnumeric(num_runs) || num_runs < 1
        error('Number of runs must be a positive integer');
    end
    
    if num_runs < 10
        warning('Number of runs (%d) may be insufficient for statistical significance. Recommended: >= 30', num_runs);
    end
end

function comparison_results = initialize_comparison_results(config, algorithms, num_runs)
    % INITIALIZE_COMPARISON_RESULTS - Initializes comparison results structure
    
    comparison_results = struct();
    comparison_results.config = config;
    comparison_results.algorithms = algorithms;
    comparison_results.num_runs = num_runs;
    comparison_results.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    % Initialize run results
    comparison_results.runs = struct();
    comparison_results.metrics = struct();
    
    % Initialize summary statistics
    comparison_results.summary = struct();
    for i = 1:length(algorithms)
        alg = algorithms{i};
        comparison_results.summary.(alg) = struct();
        comparison_results.summary.(alg).fitness = [];
        comparison_results.summary.(alg).execution_time = [];
        comparison_results.summary.(alg).convergence_iterations = [];
        comparison_results.summary.(alg).throughput = [];
        comparison_results.summary.(alg).latency = [];
        comparison_results.summary.(alg).utilization = [];
        comparison_results.summary.(alg).fairness = [];
        comparison_results.summary.(alg).qos_satisfaction = [];
        comparison_results.summary.(alg).success_rate = 0;
    end
end

function statistical_analysis = initialize_statistical_analysis(algorithms)
    % INITIALIZE_STATISTICAL_ANALYSIS - Initializes statistical analysis structure
    
    statistical_analysis = struct();
    statistical_analysis.algorithms = algorithms;
    statistical_analysis.confidence_level = 0.95;
    statistical_analysis.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    % Initialize statistical measures
    for i = 1:length(algorithms)
        alg = algorithms{i};
        statistical_analysis.(alg) = struct();
        statistical_analysis.(alg).fitness = struct();
        statistical_analysis.(alg).execution_time = struct();
        statistical_analysis.(alg).throughput = struct();
        statistical_analysis.(alg).latency = struct();
        statistical_analysis.(alg).utilization = struct();
        statistical_analysis.(alg).fairness = struct();
        statistical_analysis.(alg).qos_satisfaction = struct();
    end
end

function update_running_statistics(comparison_results, statistical_analysis, algorithm, results, metrics, run)
    % UPDATE_RUNNING_STATISTICS - Updates running statistics during evaluation
    
    if isfield(results, 'failed') && results.failed
        return; % Skip failed runs
    end
    
    % Update summary statistics
    comparison_results.summary.(algorithm).fitness = [comparison_results.summary.(algorithm).fitness, results.best_fitness];
    comparison_results.summary.(algorithm).execution_time = [comparison_results.summary.(algorithm).execution_time, results.execution_time];
    
    if ~isempty(results.convergence_iteration)
        comparison_results.summary.(algorithm).convergence_iterations = ...
            [comparison_results.summary.(algorithm).convergence_iterations, results.convergence_iteration];
    end
    
    % Update performance metrics
    total_throughput = sum([metrics.throughput.eMBB, metrics.throughput.URLLC, metrics.throughput.mMTC]);
    avg_latency = mean([metrics.latency.eMBB, metrics.latency.URLLC, metrics.latency.mMTC]);
    
    comparison_results.summary.(algorithm).throughput = [comparison_results.summary.(algorithm).throughput, total_throughput];
    comparison_results.summary.(algorithm).latency = [comparison_results.summary.(algorithm).latency, avg_latency];
    comparison_results.summary.(algorithm).utilization = [comparison_results.summary.(algorithm).utilization, metrics.efficiency.utilization_efficiency];
    comparison_results.summary.(algorithm).fairness = [comparison_results.summary.(algorithm).fairness, metrics.fairness.jains_index];
    comparison_results.summary.(algorithm).qos_satisfaction = [comparison_results.summary.(algorithm).qos_satisfaction, metrics.qos_satisfaction.overall];
    
    % Update success rate
    comparison_results.summary.(algorithm).success_rate = ...
        (length(comparison_results.summary.(algorithm).fitness) / run) * 100;
end

function statistical_analysis = perform_statistical_analysis(comparison_results, algorithms, config)
    % PERFORM_STATISTICAL_ANALYSIS - Performs comprehensive statistical analysis
    
    confidence_level = statistical_analysis.confidence_level;
    
    for i = 1:length(algorithms)
        alg = algorithms{i};
        summary = comparison_results.summary.(alg);
        
        % Calculate statistical measures for each metric
        metrics_list = {'fitness', 'execution_time', 'throughput', 'latency', 'utilization', 'fairness', 'qos_satisfaction'};
        
        for j = 1:length(metrics_list)
            metric = metrics_list{j};
            data = summary.(metric);
            
            if ~isempty(data)
                % Basic statistics
                statistical_analysis.(alg).(metric).mean = mean(data);
                statistical_analysis.(alg).(metric).std = std(data);
                statistical_analysis.(alg).(metric).median = median(data);
                statistical_analysis.(alg).(metric).min = min(data);
                statistical_analysis.(alg).(metric).max = max(data);
                
                % Confidence intervals
                n = length(data);
                t_value = tinv((1 + confidence_level) / 2, n - 1);
                margin_of_error = t_value * statistical_analysis.(alg).(metric).std / sqrt(n);
                statistical_analysis.(alg).(metric).ci_lower = statistical_analysis.(alg).(metric).mean - margin_of_error;
                statistical_analysis.(alg).(metric).ci_upper = statistical_analysis.(alg).(metric).mean + margin_of_error;
                
                % Normality test
                [h, p_value] = lillietest(data);
                statistical_analysis.(alg).(metric).is_normal = h == 0;
                statistical_analysis.(alg).(metric).normality_p_value = p_value;
            end
        end
    end
    
    % Perform pairwise comparisons
    statistical_analysis.pairwise_comparisons = perform_pairwise_comparisons(comparison_results, algorithms);
    
    % Perform ranking analysis
    statistical_analysis.ranking = perform_ranking_analysis(comparison_results, algorithms);
end

function pairwise_comparisons = perform_pairwise_comparisons(comparison_results, algorithms)
    % PERFORM_PAIRWISE_COMPARISONS - Performs pairwise statistical comparisons
    
    pairwise_comparisons = struct();
    metrics_list = {'fitness', 'execution_time', 'throughput', 'latency', 'utilization', 'fairness', 'qos_satisfaction'};
    
    for i = 1:length(algorithms)
        for j = i+1:length(algorithms)
            alg1 = algorithms{i};
            alg2 = algorithms{j};
            comparison_name = sprintf('%s_vs_%s', alg1, alg2);
            
            pairwise_comparisons.(comparison_name) = struct();
            
            for k = 1:length(metrics_list)
                metric = metrics_list{k};
                data1 = comparison_results.summary.(alg1).(metric);
                data2 = comparison_results.summary.(alg2).(metric);
                
                if ~isempty(data1) && ~isempty(data2)
                    % T-test
                    [h, p_value, ~, stats] = ttest2(data1, data2);
                    pairwise_comparisons.(comparison_name).(metric).t_test.h = h;
                    pairwise_comparisons.(comparison_name).(metric).t_test.p_value = p_value;
                    pairwise_comparisons.(comparison_name).(metric).t_test.t_statistic = stats.tstat;
                    
                    % Wilcoxon rank-sum test (non-parametric)
                    [p_value, h, stats] = ranksum(data1, data2);
                    pairwise_comparisons.(comparison_name).(metric).wilcoxon.h = h;
                    pairwise_comparisons.(comparison_name).(metric).wilcoxon.p_value = p_value;
                    pairwise_comparisons.(comparison_name).(metric).wilcoxon.z_statistic = stats.zval;
                    
                    % Effect size (Cohen's d)
                    pooled_std = sqrt(((length(data1) - 1) * var(data1) + (length(data2) - 1) * var(data2)) / ...
                                    (length(data1) + length(data2) - 2));
                    cohens_d = (mean(data1) - mean(data2)) / pooled_std;
                    pairwise_comparisons.(comparison_name).(metric).effect_size = cohens_d;
                end
            end
        end
    end
end

function ranking = perform_ranking_analysis(comparison_results, algorithms)
    % PERFORM_RANKING_ANALYSIS - Performs ranking analysis across algorithms
    
    ranking = struct();
    metrics_list = {'fitness', 'execution_time', 'throughput', 'latency', 'utilization', 'fairness', 'qos_satisfaction'};
    
    % Initialize ranking structure
    for i = 1:length(metrics_list)
        metric = metrics_list{i};
        ranking.(metric) = struct();
        ranking.(metric).algorithm_ranks = [];
        ranking.(metric).algorithm_names = algorithms;
    end
    
    % Calculate ranks for each metric
    for i = 1:length(metrics_list)
        metric = metrics_list{i};
        means = zeros(1, length(algorithms));
        
        for j = 1:length(algorithms)
            alg = algorithms{j};
            data = comparison_results.summary.(alg).(metric);
            if ~isempty(data)
                means(j) = mean(data);
            else
                means(j) = NaN;
            end
        end
        
        % Sort algorithms by performance (higher is better for most metrics)
        if strcmp(metric, 'fitness') || strcmp(metric, 'latency') || strcmp(metric, 'execution_time')
            % Lower is better for these metrics
            [~, sorted_indices] = sort(means);
        else
            % Higher is better for other metrics
            [~, sorted_indices] = sort(means, 'descend');
        end
        
        ranking.(metric).algorithm_ranks = sorted_indices;
        ranking.(metric).mean_values = means;
    end
    
    % Calculate overall ranking
    overall_scores = zeros(1, length(algorithms));
    for i = 1:length(algorithms)
        for j = 1:length(metrics_list)
            metric = metrics_list{j};
            rank = find(ranking.(metric).algorithm_ranks == i);
            if ~isempty(rank)
                overall_scores(i) = overall_scores(i) + rank;
            end
        end
    end
    
    [~, overall_ranking] = sort(overall_scores);
    ranking.overall.algorithm_ranks = overall_ranking;
    ranking.overall.algorithm_names = algorithms;
    ranking.overall.scores = overall_scores;
end

function generate_evaluation_reports(comparison_results, statistical_analysis, config, algorithms)
    % GENERATE_EVALUATION_REPORTS - Generates comprehensive evaluation reports
    
    % Create reports directory
    reports_dir = fullfile(config.output.data_dir, 'evaluation_reports');
    if ~exist(reports_dir, 'dir')
        mkdir(reports_dir);
    end
    
    % Generate detailed report
    report_file = fullfile(reports_dir, sprintf('%s_evaluation_report.txt', config.output.file_prefix));
    fid = fopen(report_file, 'w');
    
    if fid ~= -1
        fprintf(fid, '=== 5G Network Slicing Optimization Evaluation Report ===\n');
        fprintf(fid, 'Timestamp: %s\n', comparison_results.timestamp);
        fprintf(fid, 'Number of Runs: %d\n', comparison_results.num_runs);
        fprintf(fid, 'Algorithms: %s\n', strjoin(algorithms, ', '));
        fprintf(fid, 'Confidence Level: %.1f%%\n', statistical_analysis.confidence_level * 100);
        fprintf(fid, '\n');
        
        % Summary statistics
        fprintf(fid, '--- Summary Statistics ---\n');
        for i = 1:length(algorithms)
            alg = algorithms{i};
            summary = comparison_results.summary.(alg);
            fprintf(fid, '\nAlgorithm: %s\n', alg);
            fprintf(fid, 'Success Rate: %.1f%%\n', summary.success_rate);
            fprintf(fid, 'Mean Fitness: %.4f ± %.4f\n', ...
                    statistical_analysis.(alg).fitness.mean, ...
                    statistical_analysis.(alg).fitness.std);
            fprintf(fid, 'Mean Execution Time: %.4f ± %.4f seconds\n', ...
                    statistical_analysis.(alg).execution_time.mean, ...
                    statistical_analysis.(alg).execution_time.std);
        end
        
        % Statistical comparisons
        fprintf(fid, '\n--- Statistical Comparisons ---\n');
        pairwise_fields = fieldnames(statistical_analysis.pairwise_comparisons);
        for i = 1:length(pairwise_fields)
            comparison = pairwise_fields{i};
            fprintf(fid, '\nComparison: %s\n', comparison);
            
            comp_data = statistical_analysis.pairwise_comparisons.(comparison);
            metrics_list = fieldnames(comp_data);
            
            for j = 1:length(metrics_list)
                metric = metrics_list{j};
                if isstruct(comp_data.(metric))
                    fprintf(fid, '  %s: p-value=%.4f, effect_size=%.4f\n', ...
                            metric, comp_data.(metric).t_test.p_value, ...
                            comp_data.(metric).effect_size);
                end
            end
        end
        
        % Ranking analysis
        fprintf(fid, '\n--- Ranking Analysis ---\n');
        metrics_list = {'fitness', 'throughput', 'latency', 'utilization', 'fairness', 'qos_satisfaction'};
        for i = 1:length(metrics_list)
            metric = metrics_list{i};
            fprintf(fid, '\n%s Ranking:\n', metric);
            ranks = statistical_analysis.ranking.(metric).algorithm_ranks;
            names = statistical_analysis.ranking.(metric).algorithm_names;
            for j = 1:length(ranks)
                fprintf(fid, '  %d. %s\n', j, names{ranks(j)});
            end
        end
        
        fprintf(fid, '\nOverall Ranking:\n');
        overall_ranks = statistical_analysis.ranking.overall.algorithm_ranks;
        overall_names = statistical_analysis.ranking.overall.algorithm_names;
        for i = 1:length(overall_ranks)
            fprintf(fid, '  %d. %s\n', i, overall_names{overall_ranks(i)});
        end
        
        fclose(fid);
        fprintf('Evaluation report generated: %s\n', report_file);
    end
    
    % Save data structures
    data_file = fullfile(reports_dir, sprintf('%s_evaluation_data.mat', config.output.file_prefix));
    save(data_file, 'comparison_results', 'statistical_analysis');
    fprintf('Evaluation data saved: %s\n', data_file);
end

function create_evaluation_visualizations(comparison_results, statistical_analysis, config, algorithms)
    % CREATE_EVALUATION_VISUALIZATIONS - Creates comprehensive visualizations
    
    % Create plots directory
    plots_dir = fullfile(config.output.plots_dir, 'evaluation');
    if ~exist(plots_dir, 'dir')
        mkdir(plots_dir);
    end
    
    % 1. Box plots for key metrics
    create_box_plots(comparison_results, algorithms, plots_dir, config);
    
    % 2. Performance comparison bar charts
    create_performance_bar_charts(statistical_analysis, algorithms, plots_dir, config);
    
    % 3. Convergence analysis
    create_convergence_plots(comparison_results, algorithms, plots_dir, config);
    
    % 4. Statistical significance heatmap
    create_significance_heatmap(statistical_analysis, algorithms, plots_dir, config);
    
    % 5. Ranking visualization
    create_ranking_visualization(statistical_analysis, algorithms, plots_dir, config);
end

function create_box_plots(comparison_results, algorithms, plots_dir, config)
    % CREATE_BOX_PLOTS - Creates box plots for key metrics
    
    metrics_list = {'fitness', 'execution_time', 'throughput', 'latency', 'utilization', 'fairness', 'qos_satisfaction'};
    metric_labels = {'Fitness', 'Execution Time (s)', 'Throughput', 'Latency', 'Utilization', 'Fairness', 'QoS Satisfaction'};
    
    for i = 1:length(metrics_list)
        metric = metrics_list{i};
        label = metric_labels{i};
        
        figure('Visible', 'off');
        
        % Prepare data for box plot
        data = [];
        groups = [];
        
        for j = 1:length(algorithms)
            alg = algorithms{j};
            metric_data = comparison_results.summary.(alg).(metric);
            if ~isempty(metric_data)
                data = [data, metric_data];
                groups = [groups, repmat(j, 1, length(metric_data))];
            end
        end
        
        if ~isempty(data)
            boxplot(data, groups, 'Labels', algorithms);
            title(sprintf('%s Comparison', label));
            ylabel(label);
            xlabel('Algorithm');
            grid on;
            
            % Save plot
            plot_file = fullfile(plots_dir, sprintf('%s_boxplot_%s.png', config.output.file_prefix, metric));
            saveas(gcf, plot_file);
            close(gcf);
        end
    end
end

function create_performance_bar_charts(statistical_analysis, algorithms, plots_dir, config)
    % CREATE_PERFORMANCE_BAR_CHARTS - Creates performance comparison bar charts
    
    metrics_list = {'fitness', 'throughput', 'latency', 'utilization', 'fairness', 'qos_satisfaction'};
    metric_labels = {'Fitness', 'Throughput', 'Latency', 'Utilization', 'Fairness', 'QoS Satisfaction'};
    
    for i = 1:length(metrics_list)
        metric = metrics_list{i};
        label = metric_labels{i};
        
        figure('Visible', 'off');
        
        % Extract means and confidence intervals
        means = zeros(1, length(algorithms));
        ci_lower = zeros(1, length(algorithms));
        ci_upper = zeros(1, length(algorithms));
        
        for j = 1:length(algorithms)
            alg = algorithms{j};
            if isfield(statistical_analysis.(alg), metric)
                means(j) = statistical_analysis.(alg).(metric).mean;
                ci_lower(j) = statistical_analysis.(alg).(metric).ci_lower;
                ci_upper(j) = statistical_analysis.(alg).(metric).ci_upper;
            end
        end
        
        % Create bar chart with error bars
        bar(means);
        hold on;
        errorbar(1:length(algorithms), means, means - ci_lower, ci_upper - means, 'k.', 'LineWidth', 1.5);
        hold off;
        
        title(sprintf('%s Performance Comparison', label));
        ylabel(label);
        xlabel('Algorithm');
        set(gca, 'XTickLabel', algorithms);
        grid on;
        
        % Save plot
        plot_file = fullfile(plots_dir, sprintf('%s_bar_%s.png', config.output.file_prefix, metric));
        saveas(gcf, plot_file);
        close(gcf);
    end
end

function create_convergence_plots(comparison_results, algorithms, plots_dir, config)
    % CREATE_CONVERGENCE_PLOTS - Creates convergence analysis plots
    
    figure('Visible', 'off');
    
    for i = 1:length(algorithms)
        alg = algorithms{i};
        convergence_data = comparison_results.summary.(alg).convergence_iterations;
        
        if ~isempty(convergence_data)
            subplot(2, 2, i);
            histogram(convergence_data, 10);
            title(sprintf('%s Convergence Distribution', alg));
            xlabel('Iterations to Converge');
            ylabel('Frequency');
            grid on;
        end
    end
    
    % Save plot
    plot_file = fullfile(plots_dir, sprintf('%s_convergence_analysis.png', config.output.file_prefix));
    saveas(gcf, plot_file);
    close(gcf);
end

function create_significance_heatmap(statistical_analysis, algorithms, plots_dir, config)
    % CREATE_SIGNIFICANCE_HEATMAP - Creates statistical significance heatmap
    
    metrics_list = {'fitness', 'throughput', 'latency', 'utilization', 'fairness', 'qos_satisfaction'};
    
    % Create significance matrix
    significance_matrix = zeros(length(algorithms), length(algorithms));
    
    for i = 1:length(algorithms)
        for j = 1:length(algorithms)
            if i ~= j
                comparison_name = sprintf('%s_vs_%s', algorithms{i}, algorithms{j});
                if isfield(statistical_analysis.pairwise_comparisons, comparison_name)
                    % Use fitness p-value as overall significance
                    p_value = statistical_analysis.pairwise_comparisons.(comparison_name).fitness.t_test.p_value;
                    significance_matrix(i, j) = p_value;
                end
            end
        end
    end
    
    figure('Visible', 'off');
    imagesc(significance_matrix);
    colorbar;
    title('Statistical Significance Heatmap (p-values)');
    xlabel('Algorithm');
    ylabel('Algorithm');
    set(gca, 'XTickLabel', algorithms, 'YTickLabel', algorithms);
    
    % Save plot
    plot_file = fullfile(plots_dir, sprintf('%s_significance_heatmap.png', config.output.file_prefix));
    saveas(gcf, plot_file);
    close(gcf);
end

function create_ranking_visualization(statistical_analysis, algorithms, plots_dir, config)
    % CREATE_RANKING_VISUALIZATION - Creates ranking visualization
    
    metrics_list = {'fitness', 'throughput', 'latency', 'utilization', 'fairness', 'qos_satisfaction'};
    metric_labels = {'Fitness', 'Throughput', 'Latency', 'Utilization', 'Fairness', 'QoS'};
    
    % Create ranking matrix
    ranking_matrix = zeros(length(algorithms), length(metrics_list));
    
    for i = 1:length(metrics_list)
        metric = metrics_list{i};
        ranks = statistical_analysis.ranking.(metric).algorithm_ranks;
        for j = 1:length(ranks)
            ranking_matrix(ranks(j), i) = j;
        end
    end
    
    figure('Visible', 'off');
    imagesc(ranking_matrix);
    colorbar;
    title('Algorithm Rankings Across Metrics');
    xlabel('Metric');
    ylabel('Algorithm');
    set(gca, 'XTickLabel', metric_labels, 'YTickLabel', algorithms);
    
    % Add rank numbers
    for i = 1:size(ranking_matrix, 1)
        for j = 1:size(ranking_matrix, 2)
            if ranking_matrix(i, j) > 0
                text(j, i, num2str(ranking_matrix(i, j)), 'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
            end
        end
    end
    
    % Save plot
    plot_file = fullfile(plots_dir, sprintf('%s_ranking_visualization.png', config.output.file_prefix));
    saveas(gcf, plot_file);
    close(gcf);
end

function display_evaluation_summary(comparison_results, statistical_analysis, algorithms)
    % DISPLAY_EVALUATION_SUMMARY - Displays evaluation summary
    
    fprintf('\n=== Performance Evaluation Summary ===\n');
    fprintf('Algorithms evaluated: %s\n', strjoin(algorithms, ', '));
    fprintf('Number of runs: %d\n', comparison_results.num_runs);
    fprintf('Confidence level: %.1f%%\n', statistical_analysis.confidence_level * 100);
    fprintf('\n');
    
    % Display success rates
    fprintf('Success Rates:\n');
    for i = 1:length(algorithms)
        alg = algorithms{i};
        success_rate = comparison_results.summary.(alg).success_rate;
        fprintf('  %s: %.1f%%\n', alg, success_rate);
    end
    fprintf('\n');
    
    % Display overall ranking
    fprintf('Overall Ranking:\n');
    overall_ranks = statistical_analysis.ranking.overall.algorithm_ranks;
    overall_names = statistical_analysis.ranking.overall.algorithm_names;
    for i = 1:length(overall_ranks)
        fprintf('  %d. %s\n', i, overall_names{overall_ranks(i)});
    end
    fprintf('\n');
    
    % Display best performing algorithm for each metric
    metrics_list = {'fitness', 'throughput', 'latency', 'utilization', 'fairness', 'qos_satisfaction'};
    metric_labels = {'Fitness', 'Throughput', 'Latency', 'Utilization', 'Fairness', 'QoS Satisfaction'};
    
    fprintf('Best Algorithm by Metric:\n');
    for i = 1:length(metrics_list)
        metric = metrics_list{i};
        ranks = statistical_analysis.ranking.(metric).algorithm_ranks;
        names = statistical_analysis.ranking.(metric).algorithm_names;
        best_alg = names{ranks(1)};
        fprintf('  %s: %s\n', metric_labels{i}, best_alg);
    end
    fprintf('=====================================\n\n');
end 