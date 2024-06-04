from analysis import DataLoader

training_csv_files = 'train.csv'
ideal_functions_csv = 'ideal.csv'
test_data_csv = 'test.csv'

# Load data
data_loader = DataLoader()
data_loader.load_training_data(training_csv_files)
data_loader.load_ideal_functions(ideal_functions_csv)
data_loader.load_test_data(test_data_csv)

# Find best fit ideal functions
ideal_finder = IdealFunctionFinder()
best_fits = ideal_finder.find_best_fit()

# Map test data to ideal functions
test_mapper = TestMapper(best_fits)
test_mapper.map_test_data()

# Visualize data
visualizer = Visualizer(best_fits)
visualizer.visualize()