#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rosbag, sys, csv
import rospy
import time
import string
import os #for file management make directory
import shutil #for file management, copy file
import ntpath

#verify correct input arguments: 1 or 2
if (len(sys.argv) > 3):
	print "invalid number of arguments:   " + str(len(sys.argv))
	print "should be 2: 'bag2csv.py' and 'bagName'"
	print "or just 1  : 'bag2csv.py'"
	sys.exit(1)
elif (len(sys.argv) == 3):
	listOfBagFiles = [sys.argv[1]]
	numberOfFiles = "1"
	folder_name = sys.argv[2]
	print "reading only 1 bagfile: " + str(listOfBagFiles[0])
elif (len(sys.argv) == 1):
	listOfBagFiles = [f for f in os.listdir(".") if f[-4:] == ".bag"]	#get list of only bag files in current dir.
	numberOfFiles = str(len(listOfBagFiles))
	print "reading all " + numberOfFiles + " bagfiles in current directory: \n"
	for f in listOfBagFiles:
		print f
	print "\n press ctrl+c in the next 10 seconds to cancel \n"
	time.sleep(10)
else:
	print "bad argument(s): " + str(sys.argv)	#shouldnt really come up
	sys.exit(1)

count = 0
for bagFile in listOfBagFiles:
	count += 1
	print "reading file " + str(count) + " of  " + numberOfFiles + ": " + bagFile
	#access bag
	bag = rosbag.Bag(bagFile)
	bagContents = bag.read_messages()
	bagPathName = bag.filename
	bagPath, bagName = ntpath.split(bagPathName)


	#create a new directory
	folder = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'raw_data', folder_name)

	try:	#else already exists
		os.makedirs(folder)
	except:
		pass
	shutil.copyfile(bagPathName, folder + '/' + bagName)

	#get list of topics from the bag
	listOfTopics = []
	listOfTimes = []
	for topic, msg, t in bagContents:
		if topic not in listOfTopics:
			listOfTopics.append(topic)
		t = t.secs+(t.nsecs*10**(-9))
		t = float(t)
		listOfTimes.append(t)

	# get first time of bag file
	ts = min(listOfTimes)

	for topicName in listOfTopics:
		#Create a new CSV file for each topic
		filename = folder + '/' + string.replace(topicName, '/', '_') + '.csv'
		with open(filename, 'w+') as csvfile:
			filewriter = csv.writer(csvfile, delimiter = ',')
			firstIteration = True	#allows header row
			for subtopic, msg, t in bag.read_messages(topicName):	# for each instant in time that has data for topicName
				#parse data from this instant, which is of the form of multiple lines of "Name: value\n"
				#	- put it in the form of a list of 2-element lists
				t = t.secs+(t.nsecs*10**(-9))
				t = float(t)
				msgString = str(msg)
				msgList = string.split(msgString, '\n')
				instantaneousListOfData = []
				for nameValuePair in msgList:
					splitPair = string.split(nameValuePair, ':')
					for i in range(len(splitPair)):	#should be 0 to 1
						splitPair[i] = string.strip(splitPair[i])
					instantaneousListOfData.append(splitPair)
				#write the first row from the first element of each pair
				if firstIteration:	# header
					headers = ["rosbagTimestamp"]	#first column header
					for pair in instantaneousListOfData:
						headers.append(pair[0])
					filewriter.writerow(headers)
					firstIteration = False
				# write the value from each pair to the file
				t -= ts
				values = [str(t)]	#first column will have rosbag timestamp
				for pair in instantaneousListOfData:
					if len(pair) > 1:
						values.append(pair[1])
				filewriter.writerow(values)
	bag.close()
print "Done reading all " + numberOfFiles + " bag files."

# example running:
# ./extract_topics_from_bag.py ~/.ros/deepak_command_issuing_2020-02-11-23-56-11.bag deepak_command_issuing
