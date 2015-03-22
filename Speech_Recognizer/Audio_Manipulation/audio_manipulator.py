from pydub import AudioSegment
from os import listdir
from os.path import isfile, join, split
from os import walk
import os
import sys

inputdir = r'./OriginalAudioFiles' #Should contain sub-dirs for each student
outputdir = r'./ModifiedAudioFiles' #Destination folder
noisefile = r'Noise3.wav' 

noise_segment = AudioSegment.from_wav(noisefile) 

#Function to get all the wav files given a root directory
def get_files_list(root):
	files_list = []
	for dirpath, dirnames, files in walk(root):
		for name in files:
			if name.lower().endswith(".wav"):
				files_list.append(join(dirpath,name))
	return files_list

#Function to generate audio segment object for each audio file,given the audio files list
def audio_segment_generator(audio_file_list, changeChannel=0):
	audio_segment_list = []
	command = "AudioSegment.from_wav(file)"
	if(changeChannel == 1):
		command += ".set_channels(2)"
	for file in audio_file_list:
		#print "Audio Path "+file
		audio_segment = eval(command)
		audio_segment_list.append(audio_segment)
		print "Audio Segment "+str(len(audio_segment_list))+" prepared."
	return audio_segment_list


#Function which adds noise in the background of each audio file,
#given the audio segment list and the noise segment
def noise_insertor(audio_segment_list, noise_audio_segment):
	modified_audio_segment_list = []
	for original_clip in audio_segment_list:
		#print "Audio Path "+ audio_files[len(modified_audio_segment_list)]
		#print "Number of Channels "+ str(original_clip.channels)
		#print "Frame Rate "+ str(original_clip.frame_rate)
		#print "Length "+ str(len(original_clip))
		modified_audio_segment_list.append(original_clip.overlay(noise_audio_segment, loop=True))
		print "Audio Clip "+str(len(modified_audio_segment_list))+" prepared."
	return modified_audio_segment_list


#Function to generate training files for k-means classifier (with and without noise)
def audio_appendor(audio_segment_list):
	audio_appended = AudioSegment.empty()
	for audio in audio_segment_list:
 		 audio_appended += audio
 	noise_audio_appended = audio_appended.overlay(noise_segment, loop=True)
 	audio_appended.export(join(outputdir,"kmeans_train.wav"), format="wav")
 	noise_audio_appended.export(join(outputdir,"kmeans_train_noise.wav"), format="wav")
 	return 

#Function to dump files in destination directory
def output_generator(audio_file_list,output_list):
	output_subdirs=set() #Stores the sub-directories to be created
	output_files_name = [] #Stores the output filenames
	
	for file in audio_file_list: # Loop to find out dirs to be created and generate output file names
		fcomps = file.split("/")
		output_subdirs.add(fcomps[-2])
		output_files_name.append(fcomps[-2]+"/"+fcomps[-1])	
	

	for subdir in output_subdirs: # Loop to create dirs
		if not os.path.exists(join(outputdir,subdir)):
			os.makedirs(join(outputdir,subdir))

	for index in range(0,len(audio_file_list)): # Loop to write the files
		outfileobj = output_list[index]
		outfilename = output_files_name[index]
		outfilepath = join(outputdir,outfilename)
		outfileobj.export(outfilepath,format="wav")
	
if __name__ == "__main__" :
	
	operation_code=int(input("Enter 0 to Exit.\nEnter 1 to REMOVE METADATA and return the original audio clips.\nEnter 2 to ADD NOISE to the original clips and return them(Metadata removed automatically).\nEnter 3 to generate the 2 training files(with and without noise) for the k-means classifier.\nEnter 4 to REMOVE METADATA, CHANGE THE NUMBER OF CHANNELS TO 2 and return the original clips.\nEnter 5 to ADD NOISE, CHANGE THE NUMBER OF CHANNELS TO 2 and return them.\n"))
	if(operation_code == 0):
		sys.exit()
	else:
		print("Audio Manipulation Begins.")
		audio_files = get_files_list(inputdir)
		if(operation_code == 1):
			audio_objects = audio_segment_generator(audio_files)
			output_generator(audio_files, audio_objects)
		elif(operation_code == 2):
			audio_objects = audio_segment_generator(audio_files)
			modified_audio_objects = noise_insertor(audio_objects, noise_segment)
			output_generator(audio_files, modified_audio_objects)
		elif(operation_code == 3):
			audio_objects = audio_segment_generator(audio_files)
			print("K-Means Training Files Generation Starts")
			audio_appendor(audio_objects)
			print("K-Means Training Files Generation Ends")
		elif(operation_code == 4):
			audio_objects = audio_segment_generator(audio_files,1)
			output_generator(audio_files, audio_objects)
		elif(operation_code == 5):
			audio_objects = audio_segment_generator(audio_files,1)
			modified_audio_objects = noise_insertor(audio_objects, noise_segment)
			output_generator(audio_files, modified_audio_objects)
		print("Audio Manipulation Ends.") 
		print("Check the 'ModifiedAudioFiles' Folder for output")			

