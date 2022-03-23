import argparse
import json
import os
import pathlib
import sys

import pandas as pd
import matplotlib.pyplot as plt


class DataAnalyzer:

    PATH = pathlib.Path(__file__).parent.resolve()

    def __init__(self, **kwargs) -> None:
        self.raw_data: pd.DataFrame = None
        self.api_temp = None
        self.path = kwargs["path"]
        self.data_path = self.PATH / self.path
        self.graph = kwargs["graph"]
        self.save = kwargs["save"]
        self.results_x = []
        self.results_y = []
        self.forecast_temps = []
    
    def start(self):
        # Read file
        self.read_data()
        
        # Clean file
        self.clean_data()
        
        # Start loop for each prediction
        self.predict()
        
        self.graph_data()
        self.save_data()
        
    def read_data(self):
        try:
            for file in os.listdir(str(self.data_path)):
                if file.endswith(".csv"):
                    csv_file = file
        except Exception as e:
            print(e.message)
            print("invalid directory. Directory must contain a csv file of sensor data and a folder named \"predictions\"")
            sys.exit(1)
                
        for file in os.listdir(str(self.data_path / "predictions")):
            if file.endswith(".csv"):
                api_temp_file = file
        
        # Raw data from all sensors
        self.raw_data = pd.read_csv(self.data_path / csv_file, dtype="float64")
        
        # Minute-by-minute data from api
        self.api_temp = pd.read_csv(self.data_path / "predictions" / api_temp_file, dtype="float64")
        
        # # Start time at 0
        # self.raw_data["time"] = self.raw_data["time"] - self.raw_data["time"][0]
        # self.api_temp["time"] = self.api_temp["time"] - self.api_temp["time"][0]
        
        # Make seperate sensor arrays
        self.sensors_data = {}
        for i in self.raw_data:
            if i != "time":
                self.sensors_data[i] = self.raw_data[["time", i]]
        
        # print(self.sensors_data)
        
    def clean_data(self):
        for i in self.sensors_data:
            # Drop NaNs and -196's
            self.sensors_data[i] = self.sensors_data[i][self.sensors_data[i] > -100].dropna()
        # print(self.sensors_data)
        
    def predict(self):
        # Go through each prediction file
        files = os.listdir(str(self.data_path / "predictions"))
        files.sort()
        for file in files:
            if file.endswith(".json"):
                name = int(file[:-5])
                self.forecast_temps.append(self.get_forecast_temps(name))
                # Go through each sensor
                for sensor_id in self.sensors_data:
                    delta, success = self.get_delta(name, self.sensors_data[sensor_id], self.api_temp)
                    if success:
                        for forecast_temp in self.forecast_temps[-1]:
                            a = (forecast_temp[0]-name)/60
                            b = round(forecast_temp[1]+delta, 3)
                            c, d = self.get_temp_at_time(forecast_temp[0], self.sensors_data[sensor_id])
                            # sys.exit(0)
                            if d:
                                self.results_x.append(a)
                                self.results_y.append(b-c)
                    
    def get_delta(self, time: int, array_1: pd.DataFrame, array_2: pd.DataFrame) -> tuple[float, bool]:
        temp_1 = array_1.iloc[(array_1['time']-time).abs().argsort()[:1]]
        temp_2 = array_2.iloc[(array_2['time']-time).abs().argsort()[:1]]
        time_diff = abs(int(temp_1[temp_1.columns[1]]) - int(temp_2[temp_2.columns[1]]))
        if time_diff > 60:
            return None, False
        diff = round(float(temp_1[temp_1.columns[1]]) - float(temp_2[temp_2.columns[1]]), 3)
        return diff, True
    
    def get_temp_at_time(self, time: int, array: pd.DataFrame) -> tuple[float, bool]:
        temp_1 = array.iloc[(array['time']-time).abs().argsort()[:1]]
        time_diff = abs(int(temp_1["time"])-time)
        if time_diff > 60:
            return None, False
        return float(temp_1[temp_1.columns[1]]), True
    
    def get_forecast_temps(self, time: int) -> list[tuple[int,float]]:
        forecast_temps: list[tuple[int,float]] = []
        prediction_file = open(self.data_path / "predictions" / f"{time}.json",'r')
        prediction_dict = json.load(prediction_file)
        for j in range(1,5):
            pred_time = int(prediction_dict['hourly'][j]['dt'])
            pred_temp = round(self.k_to_f(prediction_dict['hourly'][j]['temp']), 3)
            forecast_temps.append((pred_time, pred_temp))
        return forecast_temps
        
    def graph_data(self):
        # Graphs results if specified to
        if self.graph:
            
            # # Plot sensors vs api temp
            # for i in self.sensors_data:
            #     plt.plot(self.sensors_data[i]["time"], self.sensors_data[i][i], color="green")
            # plt.plot(self.api_temp["time"], self.api_temp["temp"], color="orange")
            
            # # Plot results
            # plt.scatter(self.results_x, self.results_y)
            
            # Plot and print stats
            data = {}
            data["time_diff"] = self.results_x
            data["temp_diff"] = self.results_y
            df = pd.DataFrame(data)
            df = df[df["time_diff"] < 20]
            # print(df)
            # df1 = df[df["time"] ]
            plt.hist(df["temp_diff"])
            print(df["temp_diff"].describe())
            
            # Plot api accuracy
            # for name in self.forecast_temps:
            #     a = []
            #     b = []
            #     for forecast in name:
            #         a.append(forecast[0])
            #         b.append(forecast[1])
            #     plt.plot(a,b)
            plt.show()
            
    def save_data(self):
        # Save results if specified to
        if self.save:
            print("save")
            
    def k_to_f(self, temp):
        return (temp - 273.15) * 1.8 + 32




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-f", required=True, help="The path of the folder containing data to analyze")
    parser.add_argument("--save", "-s", action="store_true", help="Save the resulting analysis")
    parser.add_argument("--graph", "-g", action="store_true", help="Graph the resulting analysis")
    parser.add_argument("--num_sensors", "-n", type=int, required=True, help="The number or sensor columns in the specified file")
    args = parser.parse_args()
    
    da = DataAnalyzer(**vars(args))
    da.start()