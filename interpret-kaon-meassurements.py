import os
import math
import numpy as np
from collections import defaultdict # to make bins for pn and pt sorting
import re  # to extract numbers from filenames (for numerically ordering the datasets and looping)

######## MAKE SURE THE TXT FILES ARE IN A FOLDER CALLED "output-Sets" 
######## THIS FOLDER SHOULD BE IN THE SAME DIRECTORY AS THE SOURCE CODE FILE

# folder where the files are located
folder_name = 'output-Sets'
current_dir = os.path.dirname(os.path.abspath(__file__))  # get working directory
folder_path = os.path.join(current_dir, folder_name)  # full path to the folder

# folder to store the analyzed data
output_folder_name = 'analysed-data'
output_folder_path = os.path.join(current_dir, output_folder_name)

# create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# hold anti- kaons for each event in all samples
kaons_in_all_events_every_sample = []
antikaons_in_all_events_every_sample = []

# define class to create Kaon particle Object
class KaonParticle: 
    def __init__(self, px, py, pz, particle_id): # constructor for particle properties
        self.px = px
        self.py = py
        self.pz = pz
        self.particle_id = particle_id
        self.total_mom = math.sqrt( px**2 + py**2 + pz**2 ) # total momentum formula
        self.pt = math.sqrt( px**2 + py**2 ) # transverse momentum formula
        self.pn = 0.5 * math.log( (self.total_mom + pz) / (self.total_mom - pz) ) # pseudorapidity formula

    def to_row(self):
        """Returns a list of particle data formatted as a table row.""" # (only when object get printed)
        return [f"{self.particle_id:.0f}",  # Particle ID without decimals
                f"{self.px:.2f}",  # formatted to 2 decimal places
                f"{self.py:.2f}", 
                f"{self.pz:.2f}", 
                f"{self.pt:.2f}", 
                f"{self.pn:.2f}"] 

# returns the bin for pt in increments of 0.5.
def get_pt_bin(pt):
    return (int(pt // 0.5) * 0.5, (int(pt // 0.5) + 1) * 0.5)

# returns the bin for pn in increments of 1.
def get_pn_bin(pn):
    lower_bound = int(math.floor(pn))  # round down to nearest integer
    upper_bound = lower_bound + 1
    return (lower_bound, upper_bound)

def parse_file(file_path, output_file):

    try: # check if file exists
        file = open(file_path, 'r')
    except FileNotFoundError: # return error and empty list if not found
        print(f'File "{file_path}" not found. \nMake sure the file is in a folder "{folder_name}" in the same directory as the source code.')
        return []
    
    # initilize variables
    objects_list = []
    event_num = 0 #store the number of events
    pt_pn_table = defaultdict(lambda: defaultdict(int))  # Defaultdict for counting particles in pt, pn bins

    # variables to count toatal kaon anti-/particles
    kaons_in_this_event = 0
    antikaons_in_this_event = 0
    kaons_in_all_events = []
    antikaons_in_all_events = []

    # open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()  # split the line by spaces

            # if it does not have 4 values its an event
            if len(values) != 4: 
                # append the kaon number on new event and set current to 0
                if event_num != 0:
                    kaons_in_all_events.append(kaons_in_this_event)
                    antikaons_in_all_events.append(antikaons_in_this_event)
                    kaons_in_all_events_every_sample.append(kaons_in_this_event)
                    antikaons_in_all_events_every_sample.append(antikaons_in_this_event)
                    kaons_in_this_event = 0
                    antikaons_in_this_event = 0

                event_num +=1
                continue  # end current iteration if line is event

            px, py, pz, particle_id = map(float, values)  # convert values to floats and assigns each value from list to a variable

            # break current iteration if particle is not a Kaon
            if abs(particle_id) != 321: 
                continue

            if (particle_id > 0): # if its positive 
                kaons_in_this_event+=1
            else: # if its negative
                antikaons_in_this_event+=1

            # create a new KaonParticle object using the class and append it to the list
            obj = KaonParticle(px, py, pz, particle_id)
            objects_list.append(obj)

            # get the pt and pn bins
            pt_bin = get_pt_bin(obj.pt)  # Bin pt in increments of 0.5
            pn_bin = get_pn_bin(obj.pn)  # Bin pn in increments of 1

            # increment the corresponding bin
            pt_pn_table[pt_bin][pn_bin] += 1

    # after the last itteration, append the lists with particle numbers
    kaons_in_all_events.append(kaons_in_this_event)
    antikaons_in_all_events.append(antikaons_in_this_event)
    kaons_in_all_events_every_sample.append(kaons_in_this_event)
    antikaons_in_all_events_every_sample.append(antikaons_in_this_event)

    # write pt and pn ranges
    write_pt_pn_range(pt_pn_table, output_file)

    differance_per_event = np.array(kaons_in_all_events) - np.array(antikaons_in_all_events) #differance between kaon and antikaon for every event
    mean_differance = np.mean(differance_per_event) # mean differents of all events
    std_diviation = np.std(differance_per_event, ddof = 1)


    # pring mean anti-/particles differance and their uncertainty
    with open(output_file, 'a') as f:
        f.write(f"\nFile: {file_path}\n")
        f.write(f"Mean differance between anti- and particle of Kaons: {mean_differance:.10f} +- {std_diviation:.10f}\n\n")


    return objects_list

# function to write the pt and pn bins and their counts to a file
def write_pt_pn_range(pt_pn_table, output_file):
    with open(output_file, 'w') as f:
        f.write("Transverse Momentum (pt) and Pseudorapidity (pn) Bin Counts:\n")
        for pt_bin, pn_bins in sorted(pt_pn_table.items()):
            f.write(f"pt range {pt_bin[0]:.1f} < pt <= {pt_bin[1]:.1f}:\n")
            for pn_bin, count in sorted(pn_bins.items()):
                f.write(f"  pn range {pn_bin[0]} < pn <= {pn_bin[1]}: {count} particles\n")

# function to extract the numerical index from the filename (analyse file after file by name)
def extract_number(file_name):
    # match any digits after "Set" in the filename, such as "output-Set5.txt"
    match = re.search(r"Set(\d+)", file_name)
    if match:
        return int(match.group(1))  # return the number as an integer
    return float('inf')  # if no match, return a very high number to avoid processing such files

# main function to loop over all files in the 'output-Sets' folder
def process_files_in_folder(folder_path):
    # check if the folder exists
    if not os.path.exists(folder_path):
        print(f'Folder "{folder_path}" not found. Make sure it exists in the current directory.')
        return

    # list all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # sort files numerically based on the number in their name
    files.sort(key=extract_number)

    # loop over each file in the folder
    for i, file_name in enumerate(files, start=1):
        file_path = os.path.join(folder_path, file_name)
        
        # process only .txt files
        if file_name.endswith('.txt'):
            print(f"Processing file: {file_name}")

            # create output filename based on the file index
            output_file_name = f"analysed-data-set{i}.txt"
            output_file_path = os.path.join(output_folder_path, output_file_name)
            
            # parse the file and write the results to the output file
            parse_file(file_path, output_file_path)

    # write final differance and std to file
    with open(os.path.join(current_dir, "total-diff-and-std.txt"), 'w') as f:
        differance_per_event_every_sample = np.array(kaons_in_all_events_every_sample) - np.array(antikaons_in_all_events_every_sample) #differance between kaon and antikaon for every event
        mean_differance_every_sample = np.mean(differance_per_event_every_sample) # mean differents of all events
        std_diviation_every_sample = np.std(differance_per_event_every_sample, ddof=1) #ddof=1 for sample std
        f.write("Total average differance of antiparticles vs particles over all events of all smaples for Kaons:\n")
        f.write(f"{mean_differance_every_sample:.10f} +- {std_diviation_every_sample:.10f}")
    

# run the function to process all files in the 'output-Sets' folder
process_files_in_folder(folder_path)