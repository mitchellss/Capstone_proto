import sys
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("results_3.csv")
plt.scatter(data["epoch"], data["new"])

MEAN = 1
STD = 2
TWENTY_FIVE = 4
SEVENTY_FIVE = 6

epoch = []
mean = []
std = []
twenty_five = []
seventy_five = []
for i in range(10, 200, 5):
    df = data[data["epoch"] == i]["new"]
    # print(df)
    epoch.append(i)
    mean.append(df.describe()[MEAN])
    std.append(df.describe()[STD])
    twenty_five.append(df.describe()[TWENTY_FIVE])
    seventy_five.append(df.describe()[SEVENTY_FIVE])

plt.plot(epoch, mean, color="red")
plt.plot(epoch, twenty_five, color="green")
plt.plot(epoch, seventy_five, color="green")

plt.show()

# data = pd.read_csv("std.csv")
# plt.plot(data["hour"], data["nodelta"])
# plt.plot(data["hour"], data["delta"])
# plt.plot(data["hour"], data["ml_new"])

# plt.legend(["Forecast Model", "Delta-based Model", "ML Model"], fontsize=15)
# plt.xlabel("Prediction Distance (Hours)",fontsize=15)
# plt.ylabel("Error Standard Deviation (Degrees F)",fontsize=15)
# plt.title("Prediction Distance vs Error Standard Deviation For Prediction Models", fontsize=20)

# plt.show()
