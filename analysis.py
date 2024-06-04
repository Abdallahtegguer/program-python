import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
import logging
#made by abdallah tegguer
# Configure logging
logging.basicConfig(level=logging.INFO)

Base = declarative_base()
metadata = MetaData()

# Define tables
class TrainingData(Base):
    __tablename__ = 'training_data'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)

class IdealFunction(Base):
    __tablename__ = 'ideal_functions'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    # Define up to y50
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)
    y5 = Column(Float)
    y6 = Column(Float)
    y7 = Column(Float)
    y8 = Column(Float)
    y9 = Column(Float)
    y10 = Column(Float)
    y11 = Column(Float)
    y12 = Column(Float)
    y13 = Column(Float)
    y14 = Column(Float)
    y15 = Column(Float)
    y16 = Column(Float)
    y17 = Column(Float)
    y18 = Column(Float)
    y19 = Column(Float)
    y20 = Column(Float)
    y21 = Column(Float)
    y22 = Column(Float)
    y23 = Column(Float)
    y24 = Column(Float)
    y25 = Column(Float)
    y26 = Column(Float)
    y27 = Column(Float)
    y28 = Column(Float)
    y29 = Column(Float)
    y30 = Column(Float)
    y31 = Column(Float)
    y32 = Column(Float)
    y33 = Column(Float)
    y34 = Column(Float)
    y35 = Column(Float)
    y36 = Column(Float)
    y37 = Column(Float)
    y38 = Column(Float)
    y39 = Column(Float)
    y40 = Column(Float)
    y41 = Column(Float)
    y42 = Column(Float)
    y43 = Column(Float)
    y44 = Column(Float)
    y45 = Column(Float)
    y46 = Column(Float)
    y47 = Column(Float)
    y48 = Column(Float)
    y49 = Column(Float)
    y50 = Column(Float)

class TestData(Base):
    __tablename__ = 'test_data'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    delta_y = Column(Float)
    ideal_function_no = Column(Integer)

class DataLoader:
    def __init__(self, db_url='sqlite:///data.db'):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def load_training_data(self, csv_file):
        session = self.Session()
        try:
            df = pd.read_csv(csv_file)
            for index, row in df.iterrows():
                training_data = TrainingData(x=row['x'], y1=row['y1'], y2=row['y2'], y3=row['y3'], y4=row['y4'])
                session.add(training_data)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Error loading training data: {e}")
        finally:
            session.close()

    def load_ideal_functions(self, csv_file):
        session = self.Session()
        try:
            df = pd.read_csv(csv_file)
            for index, row in df.iterrows():
                ideal_function = IdealFunction(
                    x=row['x'], y1=row['y1'], y2=row['y2'], y3=row['y3'], y4=row['y4'], y5=row['y5'], y6=row['y6'],
                    y7=row['y7'], y8=row['y8'], y9=row['y9'], y10=row['y10'], y11=row['y11'], y12=row['y12'],
                    y13=row['y13'], y14=row['y14'], y15=row['y15'], y16=row['y16'], y17=row['y17'], y18=row['y18'],
                    y19=row['y19'], y20=row['y20'], y21=row['y21'], y22=row['y22'], y23=row['y23'], y24=row['y24'],
                    y25=row['y25'], y26=row['y26'], y27=row['y27'], y28=row['y28'], y29=row['y29'], y30=row['y30'],
                    y31=row['y31'], y32=row['y32'], y33=row['y33'], y34=row['y34'], y35=row['y35'], y36=row['y36'],
                    y37=row['y37'], y38=row['y38'], y39=row['y39'], y40=row['y40'], y41=row['y41'], y42=row['y42'],
                    y43=row['y43'], y44=row['y44'], y45=row['y45'], y46=row['y46'], y47=row['y47'], y48=row['y48'],
                    y49=row['y49'], y50=row['y50']
                )
                session.add(ideal_function)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Error loading ideal functions: {e}")
        finally:
            session.close()

    def load_test_data(self, csv_file):
        session = self.Session()
        try:
            df = pd.read_csv(csv_file)
            for index, row in df.iterrows():
                test_data = TestData(x=row['x'], y=row['y'])
                session.add(test_data)
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Error loading test data: {e}")
        finally:
            session.close()

class IdealFunctionFinder:
    def __init__(self, db_url='sqlite:///data.db'):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def find_best_fit(self):
        session = self.Session()
        best_fits = {}
        try:
            training_data = pd.read_sql(session.query(TrainingData).statement, self.engine)
            ideal_functions = pd.read_sql(session.query(IdealFunction).statement, self.engine)
            
            for i in range(1, 5):
                column_name = f'y{i}'
                if column_name not in training_data.columns:
                    logging.error(f"Column {column_name} not found in training data")
                    continue

                training_func = training_data[column_name]
                errors = {}
                for j in range(1, 51):
                    ideal_column_name = f'y{j}'
                    if ideal_column_name not in ideal_functions.columns:
                        logging.error(f"Column {ideal_column_name} not found in ideal functions")
                        continue

                    ideal_func = ideal_functions[ideal_column_name]
                    error = np.sum((training_func - ideal_func) ** 2)
                    errors[j] = error
                if errors:
                    best_fit = min(errors, key=errors.get)
                    best_fits[column_name] = best_fit
            return best_fits if best_fits else None
        except Exception as e:
            logging.error(f"Error finding best fit: {e}")
        finally:
            session.close()

class TestMapper:
    def __init__(self, best_fits, db_url='sqlite:///data.db'):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.best_fits = best_fits

    def map_test_data(self):
        if not self.best_fits:
            logging.error("Best fits are not calculated")
            return

        session = self.Session()
        try:
            test_data = pd.read_sql(session.query(TestData).statement, self.engine)
            ideal_functions = pd.read_sql(session.query(IdealFunction).statement, self.engine)
            
            results = []
            for index, row in test_data.iterrows():
                x = row['x']
                y = row['y']
                best_fit_func_no = None
                min_deviation = float('inf')
                
                for key, value in self.best_fits.items():
                    ideal_y = ideal_functions.loc[ideal_functions['x'] == x, f'y{value}'].values[0]
                    deviation = abs(y - ideal_y)
                    if deviation < min_deviation:
                        min_deviation = deviation
                        best_fit_func_no = value
                
                if min_deviation <= np.sqrt(2) * min_deviation:  # Check if within threshold
                    result = TestData(x=x, y=y, delta_y=min_deviation, ideal_function_no=best_fit_func_no)
                    session.add(result)
                    results.append(result)
            session.commit()
            return results
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Error mapping test data: {e}")
        finally:
            session.close()

class Visualizer:
    def __init__(self, best_fits, db_url='sqlite:///data.db'):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.best_fits = best_fits

    def visualize(self):
        session = self.Session()
        try:
            training_data = pd.read_sql(session.query(TrainingData).statement, self.engine)
            ideal_functions = pd.read_sql(session.query(IdealFunction).statement, self.engine)
            test_data = pd.read_sql(session.query(TestData).statement, self.engine)

            plots = []
            for i in range(1, 5):
                p = figure(title=f'Training Function y{i} and Ideal Function')
                p.line(training_data['x'], training_data[f'y{i}'], legend_label=f'Training y{i}', color='blue')
                best_fit_func_no = self.best_fits[f'y{i}']
                p.line(ideal_functions['x'], ideal_functions[f'y{best_fit_func_no}'], legend_label=f'Ideal y{best_fit_func_no}', color='green')
                mapped_data = test_data[test_data['ideal_function_no'] == best_fit_func_no]
                p.circle(mapped_data['x'], mapped_data['y'], legend_label='Test Data', color='red')
                plots.append(p)

            output_file('visualization.html')
            show(column(*plots))
        except Exception as e:
            logging.error(f"Error visualizing data: {e}")
        finally:
            session.close()

# Paths to your CSV files
training_csv_file = 'train.csv'
ideal_functions_csv = 'ideal.csv'
test_data_csv = 'test.csv'

# Load data
data_loader = DataLoader()
data_loader.load_training_data(training_csv_file)
data_loader.load_ideal_functions(ideal_functions_csv)
data_loader.load_test_data(test_data_csv)

# Find best fit ideal functions
ideal_finder = IdealFunctionFinder()
best_fits = ideal_finder.find_best_fit()

if best_fits:
    # Map test data to ideal functions
    test_mapper = TestMapper(best_fits)
    test_mapper.map_test_data()

    # Visualize data
    visualizer = Visualizer(best_fits)
    visualizer.visualize()
else:
    logging.error("No best fits found. Cannot proceed with mapping and visualization.")