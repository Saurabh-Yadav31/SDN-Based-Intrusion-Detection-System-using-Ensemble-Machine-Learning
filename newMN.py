import cmd
import os
import shutil
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
import joblib
import subprocess
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import numpy as np
import threading
import time
import configparser

GREEN = '\033[92m'
RESET = '\033[0m'

class Dataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.datasets = self.load_datasets()
        self.selected_dataset = None
        self.selected_dataset_name = None

    def load_datasets(self):
        datasets = {}
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".csv"):
                key = os.path.splitext(filename)[0]
                path = os.path.join(self.dataset_dir, filename)
                datasets[key] = path
        return datasets

    def import_dataset(self, key, path, move=False):
        if not os.path.exists(path):
            print("File does not exist.")
            return
        if not path.lower().endswith('.csv'):
            print("Only CSV files are supported.")
            return
        print("Importing dataset... Please wait.")
        loading_thread = threading.Thread(
            target=self._import_dataset, args=(key, path, move))
        loading_thread.start()
        while loading_thread.is_alive():
            print(".", end='', flush=True)
            time.sleep(0.1)
        loading_thread.join()
        print("\nDataset '{}' imported successfully!".format(key))

    def _import_dataset(self, key, path, move):
        dest_path = os.path.join(self.dataset_dir, os.path.basename(path))
        if move:
            shutil.move(path, dest_path)
        else:
            shutil.copy(path, dest_path)
        self.datasets[key] = dest_path
        temp_df = pd.read_csv(path)
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        features_info_path = f'./datasets/features_info_{dataset_name}.txt'
        with open(features_info_path, 'w') as f:
            f.write("Features:\n")
            for feature in temp_df.columns:
                f.write(f"{feature}\n")

    def remove_dataset(self, key):
        if key in self.datasets:
            dataset_path = self.datasets[key]
            os.remove(dataset_path)
            del self.datasets[key]
            print(f"Dataset '{key}' removed.")
        else:
            print("Dataset not found.")

    def select_dataset(self, key):
        if key in self.datasets:
            dataset_path = self.datasets[key]
            print("Loading dataset... Please wait.")
            loading_thread = threading.Thread(
                target=self._load_dataset, args=(dataset_path,))
            loading_thread.start()
            while loading_thread.is_alive():
                print(".", end='', flush=True)
                time.sleep(0.1)
            loading_thread.join()
            print("\nDataset '{}' selected.".format(key))
        else:
            print("Dataset not found.")

    def _load_dataset(self, dataset_path):
        self.selected_dataset = pd.read_csv(dataset_path)
        self.selected_dataset_name = os.path.splitext(
            os.path.basename(dataset_path))[0]

    def top_rows(self, num, column_name=None):
        if self.selected_dataset is not None:
            if column_name:
                if column_name in self.selected_dataset.columns:
                    print(self.selected_dataset[[column_name]].head(num))
                else:
                    print(
                        f"Column '{column_name}' not found in the selected dataset.")
            else:
                print(self.selected_dataset.head(num))
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def clear_dataset(self):
        if hasattr(self, 'selected_dataset'):
            delattr(self, 'selected_dataset')
            print("Selected dataset cleared.")
        else:
            print("No dataset selected.")

    def clear_all_datasets(self):
        for filename in os.listdir(self.dataset_dir):
            file_path = os.path.join(self.dataset_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        self.datasets = {}
        if hasattr(self, 'selected_dataset'):
            delattr(self, 'selected_dataset')
        print("All datasets cleared.")

    def get_column_names_and_types(self):
        if self.selected_dataset is not None:
            column_names = self.selected_dataset.columns.tolist()
            column_types = self.selected_dataset.dtypes.tolist()
            return column_names, column_types
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")
            return [], []

    def dataset_summary(self):
        if self.selected_dataset is not None:
            summary = {}
            dataset = self.selected_dataset
            summary['rows'], summary['columns'] = dataset.shape
            summary['data_types'] = dataset.dtypes
            return summary
        else:
            return None

    def remove_outliers(self, threshold):
        if self.selected_dataset is not None:
            try:
                threshold = float(threshold)
                print("Removing outliers... Please wait.")
                loading_thread = threading.Thread(
                    target=self._remove_outliers, args=(threshold,))
                loading_thread.start()
                while loading_thread.is_alive():
                    print(".", end='', flush=True)
                    time.sleep(0.1)
                loading_thread.join()
                print("\nRows containing outliers removed.")
            except ValueError as v:
                print(v, "Invalid threshold. Please provide a numeric value.")
            except Exception as e:
                print("An error occurred:", e)
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def _remove_outliers(self, threshold):
        df = self.selected_dataset.copy()
        try:
            df['flow_id'] = df['flow_id'].str.replace('.', '')
            df['ip_src'] = df['ip_src'].str.replace('.', '')
            df['ip_dst'] = df['ip_dst'].str.replace('.', '')
            df = df.astype("float64")
        except:
            pass
        Q1 = df.quantile(0.25, axis=0)
        Q3 = df.quantile(0.75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_rows = ((df < lower_bound) | (df > upper_bound)).any(axis=1)
        df_cleaned = df[~outlier_rows]
        self.selected_dataset = df_cleaned

    def check_missing_values(self, columns=None):
        if not hasattr(self, 'selected_dataset'):
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")
            return None

        if not columns:
            missing_values = self.selected_dataset.isnull().sum()
            print("Missing values in the entire dataset:")
            if missing_values.sum() == 0:
                print("No missing values found.")
                return {}
            else:
                print(missing_values[missing_values > 0])
                return missing_values[missing_values > 0].to_dict()
        else:
            invalid_columns = [
                col for col in columns if col not in self.selected_dataset.columns]
            if invalid_columns:
                print(f"Invalid columns: {', '.join(invalid_columns)}")
                return None
            else:
                missing_values = self.selected_dataset[columns].isnull().sum()
                print(
                    f"Missing values in specified columns ({', '.join(columns)}):")
                if missing_values.sum() == 0:
                    print("No missing values found in the specified columns.")
                    return {}
                else:
                    print(missing_values[missing_values > 0])
                    return missing_values[missing_values > 0].to_dict()

    def impute_missing_values(self):
        if self.selected_dataset is not None:
            print("Imputing missing values... Please wait.")
            loading_thread = threading.Thread(
                target=self._impute_missing_values)
            loading_thread.start()
            while loading_thread.is_alive():
                print(".", end='', flush=True)
                time.sleep(0.1)
            loading_thread.join()
            print("\nMissing values imputed with mean!")
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def _impute_missing_values(self):
        df = self.selected_dataset.copy()
        try:
            df['flow_id'] = df['flow_id'].str.replace('.', '')
            df['ip_src'] = df['ip_src'].str.replace('.', '')
            df['ip_dst'] = df['ip_dst'].str.replace('.', '')
            df = df.astype("float64")
        except:
            pass
        imputer_numerical = SimpleImputer(strategy='mean')
        df_nd = imputer_numerical.fit_transform(df)
        df = pd.DataFrame(df_nd, columns=df.columns)
        self.selected_dataset = df

    def remove_redundant(self):
        if self.selected_dataset is not None:
            df = self.selected_dataset.copy()
            df.drop_duplicates(inplace=True)
            df = df.loc[:, ~df.columns.duplicated()]
            self.selected_dataset = df
            print("Redundant rows and columns removed.")
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def unique_values(self, column_name):
        if self.selected_dataset is not None:
            if column_name in self.selected_dataset.columns:
                unique_values = self.selected_dataset[column_name].unique()
                print(
                    f"Unique values in column '{column_name}':\n{unique_values}")
            else:
                print(
                    f"Column '{column_name}' not found in the selected dataset.")
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def drop_unique_columns(self):
        if self.selected_dataset is not None:
            df = self.selected_dataset.copy()
            unique_cols = [
                col for col in df.columns if df[col].nunique() == len(df[col])]
            if unique_cols:
                df.drop(columns=unique_cols, inplace=True)
                self.selected_dataset = df
                print("Columns with unique values dropped.")
            else:
                print("No columns with all unique values found in the dataset.")
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def rename_column(self, current_name, new_name):
        if self.selected_dataset is not None:
            if current_name in self.selected_dataset.columns:
                self.selected_dataset.rename(
                    columns={current_name: new_name}, inplace=True)
                print(f"Column '{current_name}' renamed to '{new_name}'.")
            else:
                print(f"Column '{current_name}' not found in the dataset.")
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def remove_columns(self, columns):
        if self.selected_dataset is not None:
            columns_to_remove = [col.strip() for col in columns]
            existing_columns = self.selected_dataset.columns
            non_existing = [
                col for col in columns_to_remove if col not in existing_columns]

            if non_existing:
                print(
                    f"Warning: Columns {', '.join(non_existing)} do not exist in the dataset.")

            columns_to_remove = [
                col for col in columns_to_remove if col in existing_columns]

            if columns_to_remove:
                self.selected_dataset = self.selected_dataset.drop(
                    columns=columns_to_remove)
                print(
                    f"Columns {', '.join(columns_to_remove)} removed successfully.")
            else:
                print("No valid columns to remove.")
        else:
            print(
                "No dataset selected. Use 'selectDataset' command to select a dataset.")

    def map_categorical_to_integer(self, column_name):
        if self.selected_dataset is not None:
            if column_name in self.selected_dataset.columns:
                categories = self.selected_dataset[column_name].unique()
                mapping = {}
                print("Mapping categories to integers:")
                for category in categories:
                    mapping[category] = input(
                        f"Enter integer for category '{category}': ")
                try:
                    self.selected_dataset[column_name] = self.selected_dataset[column_name].map(
                        mapping)
                    print(
                        f"Categorical column '{column_name}' mapped to integers.")
                except ValueError:
                    print("Error mapping categories to integers.")
            else:
                print(f"Column '{column_name}' not found in the dataset.")
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def one_hot_encoding(self, column_name):
        if self.selected_dataset is not None:
            if column_name in self.selected_dataset.columns:
                try:
                    self.selected_dataset = pd.get_dummies(
                        self.selected_dataset, columns=[column_name])
                    print(
                        f"One-hot encoding applied to column '{column_name}'.")
                except Exception as e:
                    print(f"Error performing one-hot encoding: {str(e)}")
            else:
                print(f"Column '{column_name}' not found in the dataset.")
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def rename_dataset(self, new_name):
        if self.selected_dataset is not None:
            old_name = self.selected_dataset_name
            new_dataset_path = os.path.join(
                self.dataset_dir, f"{new_name}.csv")
            old_dataset_path = os.path.join(
                self.dataset_dir, f"{old_name}.csv")
            if os.path.exists(old_dataset_path):
                os.rename(old_dataset_path, new_dataset_path)
            self.datasets[new_name] = new_dataset_path
            del self.datasets[old_name]
            self.selected_dataset_name = new_name
            print(f"Dataset '{old_name}' has been renamed to '{new_name}'.")
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def save_dataset(self, key):
        if self.selected_dataset is not None:
            dataset_path = os.path.join(self.dataset_dir, f"{key}.csv")
            self.selected_dataset.to_csv(dataset_path, index=False)
            features_info_path = os.path.join(
                self.dataset_dir, f"features_info_{key}.txt")
            with open(features_info_path, 'w') as f:
                f.write("Features:\n")
                for feature in self.selected_dataset.columns:
                    f.write(f"{feature}\n")
            print(f"Updated dataset saved as '{key}.csv'.")
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def check_missing_values(self, columns=None):
        if self.selected_dataset is None:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")
            return None

        if columns is None:
            missing_values = self.selected_dataset.isnull().sum()
            print("Missing values in the entire dataset:")
            if missing_values.sum() == 0:
                print("No missing values found.")
                return {}
            else:
                print(missing_values[missing_values > 0])
                return missing_values[missing_values > 0].to_dict()
        else:
            invalid_columns = [
                col for col in columns if col not in self.selected_dataset.columns]
            if invalid_columns:
                print(f"Invalid columns: {', '.join(invalid_columns)}")
                return None
            else:
                missing_values = self.selected_dataset[columns].isnull().sum()
                print(
                    f"Missing values in specified columns ({', '.join(columns)}):")
                if missing_values.sum() == 0:
                    print("No missing values found in the specified columns.")
                    return {}
                else:
                    print(missing_values[missing_values > 0])
                    return missing_values[missing_values > 0].to_dict()

    def select_subset(self, columns):
        if self.selected_dataset is None:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")
            return

        missing_columns = [
            col for col in columns if col not in self.selected_dataset.columns]
        if missing_columns:
            print(
                f"Columns {missing_columns} not found in the selected dataset.")
        else:
            self.selected_dataset = self.selected_dataset[columns]
            print(f"Subset of columns {columns} selected.")

    def add_columns(self, new_column_name, columns_to_add):
        if self.selected_dataset is None:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")
            return

        missing_columns = [
            col for col in columns_to_add if col not in self.selected_dataset.columns]
        if missing_columns:
            print(
                f"Columns {missing_columns} not found in the selected dataset.")
        else:
            self.selected_dataset[new_column_name] = self.selected_dataset[columns_to_add].sum(
                axis=1)
            print(
                f"New column '{new_column_name}' created by adding columns {columns_to_add}.")

    def save_feature_selection(self, dataset_name, selected_features, X_selected, y):
        features_info_path = os.path.join(
            self.dataset_dir, f"features_info_{dataset_name}.txt")
        with open(features_info_path, 'w') as f:
            f.write("Selected Features:\n")
            for feature in selected_features:
                f.write(f"{feature}\n")
        print(f"Selected features info saved as '{features_info_path}'.")

        dataset_path = os.path.join(self.dataset_dir, f"{dataset_name}.csv")
        df_selected = pd.DataFrame(X_selected, columns=selected_features)
        df_selected['label'] = y  # Add the label column
        df_selected.to_csv(dataset_path, index=False)
        print(f"Selected features saved as '{dataset_path}'.")

class Model:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.models = self.load_models()

    def load_models(self):
        models = {}
        for filename in os.listdir(self.models_dir):
            if filename.endswith(".pkl"):
                key = os.path.splitext(filename)[0]
                path = os.path.join(self.models_dir, filename)
                models[key] = path
        return models

    def remove_model(self, key):
        if key in self.models:
            model_path = self.models[key]
            os.remove(model_path)
            del self.models[key]
            print(f"Model '{key}' removed.")
        else:
            print("Model not found.")

    def clear_all_models(self):
        for filename in os.listdir(self.models_dir):
            file_path = os.path.join(self.models_dir, filename)
            os.remove(file_path)
        print("All models deleted.")

    def import_model(self, key, path):
        if not os.path.exists(path):
            print("File does not exist.")
            return
        if not path.lower().endswith('.pkl'):
            print("Only PKL files are supported.")
            return
        dest_path = os.path.join(self.models_dir, os.path.basename(path))
        shutil.copy(path, dest_path)
        print(f"Model '{key}' imported successfully!")
        self.models[key] = dest_path

    def export_model(self, model_key, export_path):
        if model_key not in self.models:
            print("Model not found. Please provide a valid model key.")
            return
        if not os.path.exists(export_path):
            print("Export path does not exist.")
            return
        model_src_path = self.models[model_key]
        model_dest_path = os.path.join(export_path, f"{model_key}.pkl")
        shutil.copy(model_src_path, model_dest_path)
        print(f"Model '{model_key}' exported to '{export_path}'.")

    def list_models(self):
        self.models = self.load_models()
        if self.models:
            print("Available trained models:")
            for key in self.models.keys():
                print(key)
        else:
            print("No trained models found.")


class Topology:
    def __init__(self, topologies_dir):
        self.topologies_dir = topologies_dir
        self.topologies = self.load_topologies()
        
    def send_test_packets(self, attack=False):
        script_content = f"""
    from mininet.net import Mininet
    from mininet.link import TCLink
    from mininet.node import RemoteController
    from linearTopo import MyTopo
    import time

    def main():
        topo = MyTopo()
        net = Mininet(topo=topo, controller=RemoteController, link=TCLink)
        net.start()
        time.sleep(5)

        h1 = net.get('h1')
        h2 = net.get('h2')

        print("Sending packets...")

        if {str(attack)}:
            print(h1.cmd('hping3 -S 10.0.0.2 -p 80 -c 100'))
        else:
            print(h1.cmd('ping -c 10 10.0.0.2'))

        time.sleep(5)
        net.stop()

    if __name__ == "__main__":
        main()
    """

        script_path = os.path.join(os.getcwd(), "send_packets.py")
        with open(script_path, "w") as f:
            f.write(script)

        print("Script written to:", script_path)

        try:
            subprocess.run([
    'gnome-terminal', '--', 'bash', '-c',
    f'sudo python3 {script_path}; echo "Press Enter to close..."; read'
])

        except Exception as e:
            print(f"Error executing send_packets script: {e}")


    def load_topologies(self):
        topologies = {}
        for filename in os.listdir(self.topologies_dir):
            if filename.endswith(".py"):
                key = os.path.splitext(filename)[0]
                path = os.path.join(self.topologies_dir, filename)
                topologies[key] = path
        return topologies

    def import_topology(self, key, path):
        if not os.path.exists(path):
            print("File does not exist.")
            return
        if not path.lower().endswith('.py'):
            print("Only Python files (.py) are supported.")
            return
        dest_path = os.path.join(self.topologies_dir, os.path.basename(path))
        shutil.copy(path, dest_path)
        print(f"Topology '{key}' imported successfully!")
        self.topologies[key] = dest_path

    def remove_topology(self, key):
        if key in self.topologies:
            topology_path = self.topologies[key]
            os.remove(topology_path)
            del self.topologies[key]
            print(f"Topology '{key}' removed.")
        else:
            print("Topology not found.")

    def start_topology(self, topology_key):
        if topology_key in self.topologies:
            topology_path = self.topologies[topology_key]
            try:
                subprocess.run(['gnome-terminal', '--', 'sudo',
                               'python3', topology_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing the topology: {e}")
        else:
            print("Topology doesn't exist!")

    def clear_mininet(self):
        try:
            subprocess.run(['sudo', 'mn', '-c'], check=True)
            print("Mininet environment cleared.")
        except subprocess.CalledProcessError as e:
            print(f"Error clearing Mininet environment: {e}")

    def describe_topology(self, topology_name):
        if topology_name not in self.topologies:
            return f"Topology '{topology_name}' not found."

        topology_path = self.topologies[topology_name]
        try:
            with open(topology_path, 'r') as file:
                content = file.read()

            # Extract the class docstring
            class_def_start = content.find("class")
            if class_def_start == -1:
                return f"No class definition found in {topology_name}"

            docstring_start = content.find('"""', class_def_start)
            if docstring_start == -1:
                return f"No docstring found for the topology class in {topology_name}"

            docstring_end = content.find('"""', docstring_start + 3)
            if docstring_end == -1:
                return f"Docstring not properly closed in {topology_name}"

            docstring = content[docstring_start+3:docstring_end].strip()
            return f"\nTopology Description for '{topology_name}':\n{docstring}"
        except Exception as e:
            return f"An error occurred while reading the topology description: {str(e)}"
            
    def send_test_packets(self, attack=False):
        print("Sending packets...")
        script = """
        from mininet.net import Mininet
        from mininet.cli import CLI
        from mininet.link import TCLink
        from mininet.node import RemoteController
        from linearTopo import MyTopo

        def send():
            topo = MyTopo()
            net = Mininet(topo=topo, controller=RemoteController, link=TCLink)
            net.start()

            h1, h2 = net.get('h1'), net.get('h2')
            print(h1.cmd('ping -c 3 10.0.0.2'))

            if {attack}:
                print(h1.cmd('hping3 -S 10.0.0.2 -p 80 -c 100'))
            else:
                print(h1.cmd('ping -c 10 10.0.0.2'))

            net.stop()

        if __name__ == "__main__":
            send()
        """.replace("{attack}", "True" if attack else "False")

        with open("send_packets.py", "w") as f:
            f.write(script)
        
        subprocess.run(['gnome-terminal', '--', 'sudo', 'python3', 'send_packets.py'])
    
class Ryu:
    def __init__(self):
        self.config_file_path = "./ryu-scripts/configurations.conf"

    def start_ryu(self):
        try:
            ryu_script_path = "./ryu-scripts/simple_switch_13.py"
            subprocess.Popen(
                ["gnome-terminal", "--", "ryu-manager", ryu_script_path])
            print("Ryu controller started.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def start_ryu_ids(self, model_name):
        try:
            ryu_script_path = "./ryu-scripts/predictionapp.py"
            config = configparser.ConfigParser()
            config.read(self.config_file_path)
            if 'DEFAULT' not in config:
                config['DEFAULT'] = {}
            config['DEFAULT']['model'] = model_name
            with open(self.config_file_path, "w") as config_file:
                config.write(config_file)
            subprocess.Popen(["gnome-terminal", "--", "ryu-manager",
                             ryu_script_path, "--config-file", self.config_file_path])
            print("Ryu controller IDS started with model:", model_name)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def clear_ryu_buffer(self):
        try:
            file_path = "./ryu-scripts/PredictFlowStatsfile.csv"
            with open(file_path, 'w') as f:
                f.truncate(0)
            print(f"Content of '{file_path}' cleared.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def set_overwrite_interval(self, overwrite_interval):
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file_path)
            if 'DEFAULT' not in config:
                config['DEFAULT'] = {}
            config['DEFAULT']['overwrite_interval'] = str(overwrite_interval)
            with open(self.config_file_path, "w") as config_file:
                config.write(config_file)
            print("Overwrite interval set to:", overwrite_interval)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def set_prediction_delay(self, prediction_delay):
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file_path)
            if 'DEFAULT' not in config:
                config['DEFAULT'] = {}
            config['DEFAULT']['prediction_delay'] = str(prediction_delay)
            with open(self.config_file_path, "w") as config_file:
                config.write(config_file)
            print("Prediction delay set to:", prediction_delay)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def get_prediction_delay(self):
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file_path)
            return config['DEFAULT'].get('prediction_delay', 'Not set')
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def get_overwrite_interval(self):
        try:
            config = configparser.ConfigParser()
            config.read(self.config_file_path)
            return config['DEFAULT'].get('overwrite_interval', 'Not set')
        except Exception as e:
            return f"An error occurred: {str(e)}"

# ... (Dataset, Model, Topology, Ryu classes)

class MachineLearning:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def train_and_evaluate(self, classifier, X, y, split_ratio, model_name):
        print(f"Training {model_name} model... Please wait.")
        self.training_error = None
        loading_thread = threading.Thread(target=self._train_and_evaluate_model, args=(
            classifier, X, y, split_ratio))
        loading_thread.start()
        while loading_thread.is_alive():
            print(".", end='', flush=True)
            time.sleep(0.1)
        loading_thread.join()
        print(f"\n{model_name} model training and evaluation completed.")

        if hasattr(self, 'cm') and hasattr(self, 'acc'):
            print("Confusion Matrix:\n", self.cm)
            print("Accuracy Score: {:.2f}%".format(self.acc * 100))

            save_model = input("Do you want to save the trained model? (yes/no): ").strip().lower()
            if save_model == 'yes':
                model_file = os.path.join(self.models_dir, f"{model_name}.pkl")
                joblib.dump(self.classifier, model_file)
                print(f"Trained model saved as '{model_file}'.")

                features_info_path = os.path.join(os.path.dirname(
                    self.models_dir), "datasets", f"features_info_{model_name}.txt")
                with open(features_info_path, 'w') as f:
                    f.write("Features:\n")
                    for feature in X.columns:
                        f.write(f"{feature}\n")
                print(f"Dataset features (excluding target variable) saved as '{features_info_path}'.")
            else:
                print("Model not saved.")
        else:
            print("Training failed. See error details above.")

    def _train_and_evaluate_model(self, classifier, X, y, split_ratio):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=split_ratio, random_state=0)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            self.cm = confusion_matrix(y_test, y_pred)
            self.acc = accuracy_score(y_test, y_pred)
            self.classifier = classifier
        except Exception as e:
            print(f"Error during training: {e}")
            self.training_error = e

    def train_xgboost(self, X, y, split_ratio, dataset_name):
        model_name = f"{dataset_name}-xgboost"
        classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
        print("Training XGBoost model...")
        self.train_and_evaluate(classifier, X, y, split_ratio, model_name)

    # ... (other methods like train_catboost, train_adaboost, train_lightgbm remain unchanged)
    def train_catboost(self, X, y, split_ratio, dataset_name):
        model_name = f"{dataset_name}-catboost"
        classifier = CatBoostClassifier(verbose=0, random_state=0)
        print("Training CatBoost model...")
        self.train_and_evaluate(classifier, X, y, split_ratio, model_name)

    def train_adaboost(self, X, y, split_ratio, dataset_name):
        model_name = f"{dataset_name}-adaboost"
        classifier = AdaBoostClassifier(random_state=0)
        print("Training AdaBoost model...")
        self.train_and_evaluate(classifier, X, y, split_ratio, model_name)

    def train_lightgbm(self, X, y, split_ratio, dataset_name):
        model_name = f"{dataset_name}-lightgbm"
        classifier = LGBMClassifier(random_state=0)
        print("Training LightGBM model...")
        self.train_and_evaluate(classifier, X, y, split_ratio, model_name)

    def feature_selection(self, dataset, num_features):
        print("Performing feature selection... Please wait.")
        loading_thread = threading.Thread(
            target=self._feature_selection, args=(dataset, num_features))
        loading_thread.start()
        while loading_thread.is_alive():
            print(".", end='', flush=True)
            time.sleep(0.1)
        loading_thread.join()
        print("\nFeature selection completed.")
        return self.selected_features, self.X_selected, self.y

    def _feature_selection(self, dataset, num_features):
        if dataset.selected_dataset is None:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")
            return

        df = dataset.selected_dataset.copy()

        target = input('Enter the name of target variable: ')
        if target not in df.columns:
            print(f"Error: '{target}' is not a column in the dataset.")
            return

        try:
            for col in ['flow_id', 'ip_src', 'ip_dst']:
                if col in df.columns:
                    df[col] = df[col].str.replace('.', '')

            df = df.apply(pd.to_numeric, errors='ignore')
        except Exception as e:
            print(f"Warning: Error in data preprocessing - {str(e)}")

        X = df.drop(columns=[target])
        y = df[target]

        imputer_numerical = SimpleImputer(strategy='mean')
        X_nd = imputer_numerical.fit_transform(X)
        X = pd.DataFrame(X_nd, columns=X.columns)

        selector = SelectKBest(
            score_func=self.pearson_corr_score, k=num_features)
        X_selected = selector.fit_transform(X, y)

        selected_feature_indices = selector.get_support(indices=True)
        self.selected_features = X.columns[selected_feature_indices].tolist()
        self.X_selected = X_selected
        self.y = y

    def pearson_corr_score(self, X, y):
        scores = []
        for feature in X.T:
            if np.all(feature == feature[0]):
                scores.append(0)
            else:
                corr = pearsonr(feature, y)[0]
                scores.append(abs(corr))
        return np.nan_to_num(scores)

class MininetIDS(cmd.Cmd):
    intro = "Welcome to Mininet-IDS!"
    prompt = f"{GREEN}mininet-ids> {RESET}"

    def __init__(self):
        super().__init__()
        self.dataset_dir = "datasets"
        self.models_dir = "models"
        self.topologies_dir = "topologies"
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        if not os.path.exists(self.topologies_dir):
            os.makedirs(self.topologies_dir)
        self.dataset = Dataset(self.dataset_dir)
        self.model = Model(self.models_dir)
        self.topology = Topology(self.topologies_dir)
        self.ryu = Ryu()
        self.ml = MachineLearning(self.models_dir)
        
    def do_sendPackets(self, arg):
        """
        Sends test packets from h1 to h2. Optionally specify 'attack' to simulate attacks.

        Usage: sendPackets [attack]
        """
        attack = arg.strip().lower() == 'attack'
        self.topology.send_test_packets(attack)

    def do_exit(self, arg):
        """
        Exits the Mininet-IDS CLI.

        Usage: exit
        """
        return True

    def do_importDataset(self, args):
        """
        Imports a dataset into the system.

        Usage: importDataset <key> <path> [move]

        Parameters:
        - key: A unique identifier for the dataset
        - path: The file path of the dataset to import
        - move (optional): If specified, moves the file instead of copying

        The dataset name shouldn't contain any dash (-)
        The dataset should be in CSV format.
        """
        parts = args.split()
        if len(parts) < 2:
            print("Usage: importDataset <key> <path> optional: [move]")
            return
        key, path = parts[:2]
        move = len(parts) >= 3 and parts[2].lower() == 'move'
        self.dataset.import_dataset(key, path, move)

    def do_listDatasets(self, args):
        """
        Lists all available datasets in the system.

        Usage: listDatasets
        """
        if self.dataset.datasets:
            print("Available datasets:")
            for key in self.dataset.datasets.keys():
                print(key)
        else:
            print("No datasets imported yet.")

    def do_removeDataset(self, args):
        """
        Removes a specific dataset from the system.

        Usage: removeDataset <key>

        Parameters:
        - key: The identifier of the dataset to remove
        """
        if not args:
            print("Please provide the key of the dataset to remove.")
            return
        self.dataset.remove_dataset(args.strip())

    def do_selectDataset(self, args):
        """
        Selects a dataset for further operations.

        Usage: selectDataset <key>

        Parameters:
        - key: The identifier of the dataset to select
        """
        if not args:
            print("Please provide the name of the dataset to select.")
            return
        self.dataset.select_dataset(args.strip())

    def do_topRows(self, line):
        """
        Displays the first n entries of the selected dataset or a specified column.

        Usage: 
        - topRows <num>
        - topRows <num> <column_name>

        Parameters:
        - num: Number of rows to display
        - column_name (optional): Name of the specific column to display

        Prerequisite: A dataset must be selected.
        """
        args = line.split()
        if len(args) == 1:
            self.dataset.top_rows(int(args[0]))
        elif len(args) == 2:
            self.dataset.top_rows(int(args[0]), args[1])
        else:
            print("Invalid arguments. Use 'topRows num' or 'topRows num column_name'.")

    def do_clearDataset(self, args):
        """
        Clears the currently selected dataset from memory.

        Usage: clearDataset

        Prerequisite: A dataset must be selected.
        """
        self.dataset.clear_dataset()

    def do_clearAllDatasets(self, args):
        """
        Clears all datasets from the system.

        Usage: clearAllDatasets
        """
        self.dataset.clear_all_datasets()

    def do_listCols(self, args):
        """
        Lists all column names and their data types in the selected dataset.

        Usage: listCols

        Prerequisite: A dataset must be selected.
        """
        column_names, column_types = self.dataset.get_column_names_and_types()
        if column_names:
            print("Column names and their data types:\n")
            for col, col_type in zip(column_names, column_types):
                print(f"{col}: {col_type}")

    def do_clrscr(self, args):
        """
        Clears the screen.

        Usage: clrscr
        """
        os.system('clear')

    def do_datasetSummary(self, args):
        """
        Provides a summary of the selected dataset, including number of rows and columns.

        Usage: datasetSummary

        Prerequisite: A dataset must be selected.
        """
        summary = self.dataset.dataset_summary()
        if summary:
            print("Dataset Summary:")
            for key, value in summary.items():
                print(f"{key}: {value}")
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def do_removeOutliers(self, threshold):
        """
        Removes outliers from the selected dataset based on the given threshold.

        Usage: removeOutliers <threshold>

        Parameters:
        - threshold: The threshold value for identifying outliers

        Prerequisite: A dataset must be selected.
        """
        self.dataset.remove_outliers(threshold)

    def do_checkMissingValues(self, args):
        """
        Checks for missing values in the selected dataset or specified columns.

        Usage: 
        - checkMissingValues
        - checkMissingValues <column_name1> [<column_name2> ...]

        Parameters:
        - column_name (optional): Names of specific columns to check

        Prerequisite: A dataset must be selected.
        """
        if not args:
            self.dataset.check_missing_values()
        else:
            columns = args.split()
            self.dataset.check_missing_values(columns)

    def do_imputeMissingValues(self, args):
        """
        Imputes missing values in the selected dataset using mean strategy.

        Usage: imputeMissingValues

        Prerequisite: A dataset must be selected.
        """
        self.dataset.impute_missing_values()

    def do_removeRedundant(self, arg):
        """
        Removes redundant rows and columns from the selected dataset.

        Usage: removeRedundant

        Prerequisite: A dataset must be selected.
        """
        self.dataset.remove_redundant()

    def do_trainXGBoost(self, args):
        """
        Trains an XGBoost model on the selected dataset.

        Usage: trainXGBoost <test_size> <target_variable>

        Parameters:
        - test_size: Proportion of the dataset to include in the test split (e.g., 0.2 for 20%)
        - target_variable: Name of the column to use as the target variable

        Prerequisite: A dataset must be selected.
        """
        try:
            split_ratio, target_variable = args.split()
            split_ratio = float(split_ratio)
            if self.dataset.selected_dataset is None:
                print("No dataset selected. Use 'selectDataset' command to select a dataset.")
                return
            
            # Prepare features (X) and target (y)
            X = self.dataset.selected_dataset.drop(columns=[target_variable]).copy()
            y = self.dataset.selected_dataset[target_variable].copy()

            # Convert target variable to numeric
            if y.dtype == object or y.dtype == 'string':
                try:
                    y = pd.to_numeric(y, errors='raise')
                except ValueError:
                    unique_values = y.unique()
                    label_map = {val: idx for idx, val in enumerate(unique_values)}
                    y = y.map(label_map)
                    print(f"Mapped target variable '{target_variable}' values to integers: {label_map}")

            # Preprocess feature columns (X) to ensure all are numeric
            for column in X.columns:
                if X[column].dtype == object or X[column].dtype == 'string':
                    try:
                        # Attempt to convert to numeric (e.g., '6' -> 6, '17' -> 17)
                        X[column] = pd.to_numeric(X[column], errors='raise')
                    except ValueError:
                        # If conversion fails (e.g., 'TCP', 'UDP'), map to integers
                        unique_values = X[column].unique()
                        feature_map = {val: idx for idx, val in enumerate(unique_values)}
                        X[column] = X[column].map(feature_map)
                        print(f"Mapped feature '{column}' values to integers: {feature_map}")

            self.ml.train_xgboost(X, y, split_ratio, self.dataset.selected_dataset_name)
        except ValueError as e:
            print(f"Error: {e}")
            print("Usage: trainXGBoost <test_size e.g. 0.1 for 10%> <target_variable>")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    def do_trainCatBoost(self, args):
        try:
            split_ratio, target_variable = args.split()
            split_ratio = float(split_ratio)
            if self.dataset.selected_dataset is None:
                print("No dataset selected. Use 'selectDataset' command to select a dataset.")
                return
            X = self.dataset.selected_dataset.drop(columns=[target_variable]).copy()
            y = self.dataset.selected_dataset[target_variable].copy()
            if y.dtype == object or y.dtype == 'string':
                try:
                    y = pd.to_numeric(y, errors='raise')
                except ValueError:
                    unique_values = y.unique()
                    label_map = {val: idx for idx, val in enumerate(unique_values)}
                    y = y.map(label_map)
                    print(f"Mapped target variable '{target_variable}' values to integers: {label_map}")
            for column in X.columns:
                if X[column].dtype == object or X[column].dtype == 'string':
                    try:
                        X[column] = pd.to_numeric(X[column], errors='raise')
                    except ValueError:
                        unique_values = X[column].unique()
                        feature_map = {val: idx for idx, val in enumerate(unique_values)}
                        X[column] = X[column].map(feature_map)
                        print(f"Mapped feature '{column}' values to integers: {feature_map}")
            self.ml.train_catboost(X, y, split_ratio, self.dataset.selected_dataset_name)
        except ValueError as e:
            print(f"Error: {e}")
            print("Usage: trainCatBoost <test_size e.g. 0.1 for 10%> <target_variable>")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    def do_trainAdaBoost(self, args):
        try:
            split_ratio, target_variable = args.split()
            split_ratio = float(split_ratio)
            if self.dataset.selected_dataset is None:
                print("No dataset selected. Use 'selectDataset' command to select a dataset.")
                return
            X = self.dataset.selected_dataset.drop(columns=[target_variable]).copy()
            y = self.dataset.selected_dataset[target_variable].copy()
            if y.dtype == object or y.dtype == 'string':
                try:
                    y = pd.to_numeric(y, errors='raise')
                except ValueError:
                    unique_values = y.unique()
                    label_map = {val: idx for idx, val in enumerate(unique_values)}
                    y = y.map(label_map)
                    print(f"Mapped target variable '{target_variable}' values to integers: {label_map}")
            for column in X.columns:
                if X[column].dtype == object or X[column].dtype == 'string':
                    try:
                        X[column] = pd.to_numeric(X[column], errors='raise')
                    except ValueError:
                        unique_values = X[column].unique()
                        feature_map = {val: idx for idx, val in enumerate(unique_values)}
                        X[column] = X[column].map(feature_map)
                        print(f"Mapped feature '{column}' values to integers: {feature_map}")
            self.ml.train_adaboost(X, y, split_ratio, self.dataset.selected_dataset_name)
        except ValueError as e:
            print(f"Error: {e}")
            print("Usage: trainAdaBoost <test_size e.g. 0.1 for 10%> <target_variable>")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
    def do_trainLightGBM(self, args):
        try:
            split_ratio, target_variable = args.split()
            split_ratio = float(split_ratio)
            if self.dataset.selected_dataset is None:
                print("No dataset selected. Use 'selectDataset' command to select a dataset.")
                return
            X = self.dataset.selected_dataset.drop(columns=[target_variable]).copy()
            y = self.dataset.selected_dataset[target_variable].copy()
            if y.dtype == object or y.dtype == 'string':
                try:
                    y = pd.to_numeric(y, errors='raise')
                except ValueError:
                    unique_values = y.unique()
                    label_map = {val: idx for idx, val in enumerate(unique_values)}
                    y = y.map(label_map)
                    print(f"Mapped target variable '{target_variable}' values to integers: {label_map}")
            for column in X.columns:
                if X[column].dtype == object or X[column].dtype == 'string':
                    try:
                        X[column] = pd.to_numeric(X[column], errors='raise')
                    except ValueError:
                        unique_values = X[column].unique()
                        feature_map = {val: idx for idx, val in enumerate(unique_values)}
                        X[column] = X[column].map(feature_map)
                        print(f"Mapped feature '{column}' values to integers: {feature_map}")
            self.ml.train_lightgbm(X, y, split_ratio, self.dataset.selected_dataset_name)
        except ValueError as e:
            print(f"Error: {e}")
            print("Usage: trainLightGBM <test_size e.g. 0.1 for 10%> <target_variable>")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
    def do_clearAllModels(self, args):
        """
        Deletes all saved machine learning models.

        Usage: clearAllModels
        """
        self.model.clear_all_models()

    def do_importModel(self, args):
        """
        Imports a machine learning model into the system.

        Usage: importModel <key> <path>

        Parameters:
        - key: A unique identifier for the model
        - path: The file path of the model to import

        The model should be in PKL format.
        """
        parts = args.split()
        if len(parts) < 2:
            print("Usage: importModel <key> <path>")
            return
        key, path = parts[:2]
        self.model.import_model(key, path)

    def do_exportModel(self, args):
        """
        Exports a specified model to a given path.

        Usage: exportModel <model_key> <export_path>

        Parameters:
        - model_key: The identifier of the model to export
        - export_path: The path where the model should be exported
        """
        parts = args.split()
        if len(parts) < 2:
            print("Usage: exportModel <model_key> <export_path>")
            return
        model_key, export_path = parts[:2]
        self.model.export_model(model_key, export_path)

    def do_listModels(self, args):
        """
        Lists all available trained machine learning models.

        Usage: listModels
        """
        self.model.list_models()

    def do_printSelectedDataset(self, args):
        """
        Prints the name of the currently selected dataset.

        Usage: printSelectedDataset

        Prerequisite: A dataset must be selected.
        """
        if self.dataset.selected_dataset is not None:
            print("Selected Dataset:")
            print(self.dataset.selected_dataset_name)
        else:
            print("No dataset selected. Use 'selectDataset' command to select a dataset.")

    def do_selectSubset(self, args):
        """
        Selects a subset of columns from the current dataset and makes it the new selected dataset.

        Usage: selectSubset <column1,column2,...>

        Parameters:
        - column1,column2,...: Comma-separated list of column names to include in the subset

        Prerequisite: A dataset must be selected.
        """
        if not args:
            print("Please provide the column names to select a subset.")
            return
        columns = [col.strip() for col in args.split(',')]
        self.dataset.select_subset(columns)

    def do_addColumns(self, args):
        """
        Creates a new column by adding the values of specified existing columns.

        Usage: addColumns <new_column_name> <column1,column2,...>

        Parameters:
        - new_column_name: Name for the new column
        - column1,column2,...: Comma-separated list of columns to add together

        Prerequisite: A dataset must be selected.
        """
        parts = args.split()
        if len(parts) < 2:
            print("Please provide the new column name and the columns to add.")
            return
        new_column_name = parts[0]
        columns_to_add = [col.strip() for col in parts[1].split(',')]
        self.dataset.add_columns(new_column_name, columns_to_add)

    def do_removeColumn(self, args):
        """
        Removes specified column(s) from the selected dataset.

        Usage: removeColumn <column_name1>,<column_name2>,...

        Parameters:
        - column_names: Comma-separated list of column names to remove

        Prerequisite: A dataset must be selected.
        """
        if not args:
            print(
                "Please provide the name(s) of the column(s) to remove, separated by commas.")
            return
        columns = [col.strip() for col in args.split(',')]
        self.dataset.remove_columns(columns)

    def do_saveDataset(self, args):
        """
        Saves the current state of the selected dataset.

        Usage: saveDataset <key>

        Parameters:
        - key: A unique identifier to save the dataset under

        Dataset name shouldn't contain any dash (-)
        Prerequisite: A dataset must be selected.
        """
        if not args:
            print(
                "Please provide the key to save the updated dataset. Don't include dash (-) in name.")
            return
        self.dataset.save_dataset(args.strip())

    def do_importTopology(self, args):
        """
        Imports a network topology into the system.

        Usage: importTopology <key> <path>

        Parameters:
        - key: A unique identifier for the topology
        - path: The file path of the topology to import

        The topology should be a Python script (.py file).
        """
        parts = args.split()
        if len(parts) < 2:
            print("Usage: importTopology <key> <path>")
            return
        key, path = parts[:2]
        self.topology.import_topology(key, path)

    def do_listTopologies(self, args):
        """
        Lists all available network topologies in the system.

        Usage: listTopologies
        """
        if self.topology.topologies:
            print("Available topologies:")
            for key in self.topology.topologies.keys():
                print(key)
        else:
            print("No topologies imported yet.")

    def do_removeTopology(self, args):
        """
        Removes a specific network topology from the system.

        Usage: removeTopology <key>

        Parameters:
        - key: The identifier of the topology to remove
        """
        if not args:
            print("Please provide the key of the topology to remove.")
            return
        self.topology.remove_topology(args.strip())

    def do_describeTopology(self, args):
        """
        Describes the structure of a selected topology.
        Usage: describeTopology <topology_name>
        """
        if not args:
            print("Please provide the name of the topology to describe.")
            return

        topology_name = args.strip()
        description = self.topology.describe_topology(topology_name)
        print(description)

    def do_startTopology(self, args):
        """
        Starts a specific network topology in the system.

        Usage: startTopology <key>

        Parameters:
        - key: The identifier of the topology to start

        This command will launch the specified topology in a new terminal window using Mininet.
        """
        if not args:
            print("Please provide the name of the topology to start.")
            return
        self.topology.start_topology(args.strip())

    def do_clearmn(self, arg):
        """
        Clears the Mininet environment.

        Usage: clearmn
        """
        self.topology.clear_mininet()

    def do_startRyu(self, args):
        """
        Starts the Ryu controller.

        Usage: startRyu
        """
        self.ryu.start_ryu()

    def do_startRyuIDS(self, arg):
        """
        Starts the Ryu controller with the IDS (Intrusion Detection System) application.

        Usage: startRyuIDS <model_name>

        Parameters:
        - model_name: Name of the machine learning model to use for IDS
        """
        if not arg:
            print("Please provide the model name as an argument.")
            return
        self.ryu.start_ryu_ids(arg.strip())

    def do_clearRyuBuffer(self, arg):
        """
        Clears the data buffer of the Ryu IDS application.

        Usage: clearRyuBuffer
        """
        self.ryu.clear_ryu_buffer()

    def do_uniqueValues(self, args):
        """
        Displays the unique values in a specified column of the selected dataset.

        Usage: uniqueValues <column_name>

        Parameters:
        - column_name: Name of the column to check for unique values

        Prerequisite: A dataset must be selected.
        """
        if not args:
            print("Please provide the column name to get unique values from.")
            return
        self.dataset.unique_values(args.strip())

    def do_dropUniqueColumns(self, args):
        """
        Drops columns with all unique values from the selected dataset.

        Usage: dropUniqueColumns

        Prerequisite: A dataset must be selected.
        """
        self.dataset.drop_unique_columns()

    def do_featureSelection(self, args):
        """
        Performs feature selection on the selected dataset.

        Usage: featureSelection <number_of_features>

        Parameters:
        - number_of_features: Number of top features to select

        Prerequisite: A dataset must be selected.
        """
        if not args:
            print(
                "Provide number of features to select.\nUsage: featureSelection <NumberOfFeaturesToSelect>")
            return
        try:
            num_features = int(args)
            selected_features, X_selected, y = self.ml.feature_selection(
                self.dataset, num_features)
            print("\nSelected Features after Feature Selection:")
            print(selected_features)

            save_dataset = input(
                "Do you want to save the selected features as a dataset? (yes/no): ").strip().lower()
            if save_dataset == 'yes':
                dataset_name = input("Enter a name for the dataset: ").strip()
                self.dataset.save_feature_selection(
                    dataset_name, selected_features, X_selected, y)
        except ValueError:
            print(
                "Invalid input. Please provide an integer value for the number of features to select.")
        except Exception as e:
            print("An error occurred during feature selection:", e)

    def do_renameColumn(self, args):
        """
        Renames a column in the selected dataset.

        Usage: renameColumn <current_name> <new_name>

        Parameters:
        - current_name: Current name of the column
        - new_name: New name for the column

        Prerequisite: A dataset must be selected.
        """
        parts = args.split()
        if len(parts) != 2:
            print("Please provide the current and new names of the column to rename.")
            return
        current_name, new_name = parts
        self.dataset.rename_column(current_name, new_name)

    def do_mapCategoricalToInteger(self, args):
        """
        Maps categories in a categorical column to integers.

        Usage: mapCategoricalToInteger <column_name>

        Parameters:
        - column_name: Name of the categorical column to map

        Prerequisite: A dataset must be selected.
        """
        if not args:
            print("Please provide the name of the categorical column to map to integers.")
            return
        self.dataset.map_categorical_to_integer(args.strip())

    def do_oneHotEncoding(self, args):
        """
        Performs one-hot encoding on a specified categorical column.

        Usage: oneHotEncoding <column_name>

        Parameters:
        - column_name: Name of the categorical column to encode

        Prerequisite: A dataset must be selected.
        """
        if not args:
            print(
                "Please provide the name of the categorical column to perform one-hot encoding.")
            return
        self.dataset.one_hot_encoding(args.strip())

    def do_showFlowStatsInfo(self, args):
        """
        Displays information about possible flow statistics that can be collected.

        Usage: showFlowStatsInfo
        """
        flow_stats_info = """
        The following flow statistics are collected in Mininet, please make sure your dataset complies with following features. Also ensure that the features names are also same as used follows:
        1. timestamp: The time at which the flow statistics are collected.
        2. datapath_id: The ID of the datapath (switch).
        3. flow_id: A unique identifier for the flow, constructed using source and destination IP addresses, source and destination ports, and the IP protocol.
        4. ip_src: Source IP address.
        5. tp_src: Source transport port (TCP/UDP).
        6. ip_dst: Destination IP address.
        7. tp_dst: Destination transport port (TCP/UDP).
        8. ip_proto: IP protocol (e.g., 6: TCP, 17: UDP, 1: ICMP).
        9. icmp_code: ICMP code (for ICMP traffic, otherwise NaN).
        10. icmp_type: ICMP type (for ICMP traffic, otherwise NaN).
        11. flow_duration_sec: Duration of the flow in seconds.
        12. flow_duration_nsec: Duration of the flow in nanoseconds.
        13. idle_timeout: Idle timeout value for the flow.
        14. hard_timeout: Hard timeout value for the flow.
        15. flags: Flags associated with the flow.
        16. packet_count: Total number of packets in the flow.
        17. byte_count: Total number of bytes in the flow.
        18. packet_count_per_second: Number of packets per second.
        19. packet_count_per_nsecond: Number of packets per nanosecond.
        20. byte_count_per_second: Number of bytes per second.
        21. byte_count_per_nsecond: Number of bytes per nanosecond.
        """
        print(flow_stats_info)

    def do_renameDataset(self, new_name):
        """
        Renames the currently selected dataset.

        Usage: renameDataset <new_name>

        Parameters:
        - new_name: New name for the dataset

        Prerequisite: A dataset must be selected.
        """
        if not new_name:
            print("Please provide a new name for the dataset.")
            return
        self.dataset.rename_dataset(new_name)

    def do_setOverwriteInterval(self, arg):
        """
        Sets the overwrite interval in the Ryu configuration file.

        Usage: setOverwriteInterval <interval>

        Parameters:
        - interval: Integer value for the overwrite interval
        """
        try:
            overwrite_interval = int(arg.strip())
            self.ryu.set_overwrite_interval(overwrite_interval)
        except ValueError:
            print("Please provide a valid integer for the overwrite interval.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def do_setPredictionDelay(self, arg):
        """
        Sets the prediction delay in the Ryu configuration file.

        Usage: setPredictionDelay <delay>

        Parameters:
        - delay: Integer value for the prediction delay
        """
        try:
            prediction_delay = int(arg.strip())
            self.ryu.set_prediction_delay(prediction_delay)
        except ValueError:
            print("Please provide a valid integer for the prediction delay.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def do_getPredictionDelay(self, arg):
        """
        Displays the current prediction delay setting from the Ryu configuration file.

        Usage: getPredictionDelay

        This function shows the current delay between predictions in the IDS system.
        """
        delay = self.ryu.get_prediction_delay()
        print(f"Current prediction delay: {delay}")

    def do_getOverwriteInterval(self, arg):
        """
        Displays the current overwrite interval setting from the Ryu configuration file.

        Usage: getOverwriteInterval

        This function shows the current interval at which data is overwritten in the IDS system.
        """
        interval = self.ryu.get_overwrite_interval()
        print(f"Current overwrite interval: {interval}")

if __name__ == "__main__":
    interface = MininetIDS()
    interface.cmdloop()
