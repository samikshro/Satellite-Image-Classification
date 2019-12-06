import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

labels = ["Barren Land", "Trees", "Grassland", "None"]
correct_class = [2178, 0, 623, 3094]
wrong_class = [477, 2014, 1227, 387]

values = [float(correct_class[i]/(wrong_class[i]+correct_class[i])) for i in range(4)]



b1 = plt.bar(labels, values, label= "Correct Classification")
# plt.bar(labels, wrong_values, label = "Wrong Classification")
plt.legend(loc='upper right')
plt.title("Classification for Variance of HSV Values as Features (10000 images)")
plt.ylabel("Percentage of Images")

def truncate(n):
  return float(int(n * 1000) / 1000)

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 0.05*height, 
                str(truncate(values[i])),
                ha='center', va='bottom')

autolabel(b1)

plt.savefig("figure2.pdf")

plt.clf()
plt.cla()
plt.close()
