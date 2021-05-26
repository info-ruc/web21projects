# Generate and combine query templates
 ./data-processor/templatizer.py tiramisu --dir ../dataset/tiramisu-sample/ --output dataset/templates
 ./data-processor/csv-combiner.py --input_dir ../dataset/templates/ --output_dir ../dataset/tiramisu-combined-csv

# Run through clustering algorithm
 ./algorithm/clusterer/online_clustering.py --dir ../dataset/tiramisu-combined-csv/ --rho 0.8
 ./algorithm/clusterer/generate-cluster-coverage.py --project tiramisu --assignment results/online-clustering-results/None-0.8-assignments.pickle --output_csv_dir results/online-clusters/ --output_dir results/cluster-coverage/

# Run forecasting models
 ./algorithm/forecaster/run_sample.sh

# Generate ENSEMBLE and HYBRID results
./algorithm/forecaster/generate_ensemble_hybrid.py results/prediction-results/agg-60/horizon-4320/ar/ results/prediction-results/agg-60/horizon-4320/noencoder-rnn/ results/prediction-results/agg-60/horizon-4320/ensemble false
./algorithm/forecaster/generate_ensemble_hybrid.py results/prediction-results/agg-60/horizon-4320/ensemble results/prediction-results/agg-60/horizon-4320/kr results/prediction-results/agg-60/horizon-4320/hybrid True
