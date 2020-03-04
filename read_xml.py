import xml.etree.ElementTree as ET

class BBox(object):
	"""docstring for BBox"""
	def __init__(self):
		super(BBox, self).__init__()
		self.xmin = None
		self.ymax = None
		self.xmax = None
		self.ymin = None

	def print_it(self):
		print("xmin: %i"%self.xmin)
		print("ymax: %i"%self.ymax)
		print("xmax: %i"%self.xmax)
		print("ymin: %i"%self.ymin)

class Vehicle(object):
	"""docstring for Vehicle"""
	def __init__(self):
		super(Vehicle, self).__init__()
		self.name = None
		self.bbox = BBox()

	def print_it(self):
		print("_"*30)
		print("Name of vehicle: {}".format(self.name))
		print("-"*30)
		self.bbox.print_it()

def read_content(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    listOfVehicles = []

    filename = root.find('filename').text
    folder = root.find('folder').text
    print "folder: ",folder

    for boxes in root.iter('object'):
        vehicle = Vehicle()

        for name in boxes.findall("name"):
        	vehicle.name = name.text

        for box in boxes.findall("bndbox"):
            vehicle.bbox.ymin = int(box.find("ymin").text)
            vehicle.bbox.xmin = int(box.find("xmin").text)
            vehicle.bbox.ymax = int(box.find("ymax").text)
            vehicle.bbox.xmax = int(box.find("xmax").text)

        listOfVehicles.append(vehicle)

    return filename, listOfVehicles

def get_file_names(trainFilePath):
	filenames = []
	trainFile = open(trainFilePath,"r")
	gt = trainFile.readline()
	while gt != '':
		gt = './IDD_Detection/Annotations/' + gt[:-1] + '.xml'
		filenames.append(gt)
		gt = trainFile.readline()
	trainFile.close()
	return filenames

if __name__ == '__main__':
	trainFilePath = './IDD_Detection/train.txt'
	filenames = get_file_names(trainFilePath)
	name, listOfVehicles = read_content(filenames[1])
	print name
	[listOfVehicles[i].print_it() for i in range(0,len(listOfVehicles))]
