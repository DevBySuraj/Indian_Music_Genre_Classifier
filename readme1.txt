

data labels: 
Sufi/Ghazal (Focus on Vocals/Harmonium)

Bhangra/Folk (Focus on heavy Dhol/Percussion)

Bollywood Pop (Modern produced sound)

Classical/Raga (Sitar, Sarod, Tabla focus)

Indie Rock (Electric guitars and drums)

This dataset contains 1000 audio samples of Indian music, categorized into five genres: Sufi/Ghazal, Bhangra/Folk, Bollywood Pop, Classical/Raga, and Indie Rock. Each sample is labeled according to its genre, allowing for analysis and classification based on musical features such as instrumentation, rhythm, and vocal style. The dataset is ideal for tasks such as genre classification, music recommendation systems, and audio feature extraction.



now what to do:
Revised Methodology Checklist
Step 1: Download High-Quality (320kbps) Hindi MP3s.
Step 2: Convert to .wav and run noisereduce.
Step 3: Use Spleeter to separate the "Music" and "Vocals."
Step 4: Generate Mel-Spectrograms for both parts.
Step 5: Save as a custom NumPy dataset (similar to what you did for GTZAN).



11/3/26

Step 2: Convert to .wav and run noisereduce.


implemented the funciton which converts the mp3 to wav and reduces the noise using the noise reduce libraray.
1. making the input and output directories
2. iterating through the mp3 files in the input directory
3. converting each mp3 file to wav format using pydub  
4. applying noise reduction to the wav file using the noisereduce library
5. saving the processed wav file to the output directory

the output file is reduced now the background music is lowered and the vocals are more hearable clearly
i think when i will use the masking then there could be a problem if i use these reduced files as the instruments parts is lowered, Quality is decreased
will just convert to wav not reduce the noise if there is found a problem in the future

this step completed on 11/3//25 with git push - 11:42 pm 



12/3/2026
step 3:
using the ai tools to separate the music(instruments,drums, beats etc) and vocals


Installing the spleeter tool(pre trained model) from the github document
## https://github.com/deezer/spleeter?tab=readme-ov-file
1. installing dependencies ffmpeg and libsndfile by conda
2. # install spleeter with pip

numpy version mismatch while using spleeter
spleeter separate -o audio_output audio_example.mp3  //command

solution:
downgrade the version

# 1. Uninstall the current NumPy
pip uninstall numpy -y

# 2. Install the compatible version
pip install "numpy<2"

# 3. Re-install Spleeter (to ensure all dependencies align)
pip install spleeter --upgrade


ffprobe error: ffprobe.exe exists int the ffmpeg folder installation

Gemini said
This error is a classic "DLL Hell" conflict. It basically means that when Spleeter
 tried to use FFmpeg (specifically ffprobe.exe), it accidentally grabbed a version 
 of a library file (gdk_pixbuf-2.0-0.dll) that is buried inside your local Conda 
 environment instead of using the clean, global version of FFmpeg you installed earlier.

now i want the spleeter to use the global real ffmpeg .

ffprobe.exe exists int the ffmpeg folder installation setting it's path to this session

set PATH=C:\FFmpeg\ffmpeg-8.0.1-full_build\bin;%PATH%  

this worked now ffmpeg is using it's own installation not the conda one(which was wrong) - temporary fix
but for permanent change i have to change the path in the environment variables

permanent change in the env variable moving the ffmpeg to the tops


using the 2 , 4 , 5 stem models
https://github.com/deezer/spleeter/wiki/2.-Getting-started#separate-sources all models

commands:
spleeter separate --help
spleeter separate -o audio_output audio_example.mp3 
spleeter separate -o audio_output -p spleeter:4stems audio_example.mp3
spleeter separate -o audio_output -p spleeter:5stems audio_example.mp3
spleeter separate -o audio_output -p spleeter:4stems-16kHz audio_example.mp3 


for the first time all the models will be download but after that the separation will be faster

now making the python function which will convert all the files into 4 stem slots - 8:23pm


9:51 pm
now making the python function which will convert all the files into 4 stem slots

Spleeter to separate your Hindi songs into 4 stems (Vocals, Drums, Bass, Other),
 you are essentially applying Time-Frequency Masking at scale.

"Implemented Time-Frequency Soft-Masking via Spleeter to decompose complex Hindi audio mixtures into
 constituent stems, effectively isolating rhythmic (Dhol/Tabla) and melodic (Sitar/Harmonium) features
  for enhanced CNN training accuracy."

Masking:
it is like applying a filter on the whole songs for each(stem/feature). lets's say vocal here, then it will apply vocal mask
on the whole audio, it will take that pixel/Frequency and make it 1(add) and make it 0(block) where it didn't find, at last we will
have our whole mask(vocal stem), the same goes for each mask here

The "Security Guard" Analogy
Imagine a spectrogram is a crowded room. Each "stem" (vocal, drums, etc.) has its own "security guard" (the mask).

The Vocal Mask: It looks at every single pixel of sound. If it "sees" the wavy, melodic shape of a human voice, it
 opens the gate (Value = 1). If it sees a sharp drum hit or a guitar string, it closes the gate (Value = 0).

The Drum Mask: At the exact same time, another mask is doing the opposite. It blocks the smooth vocal waves and only 
allows the sharp, vertical "strikes" of the Dhol or Tabla to pass through.

here soft masking is used:
Soft masking: like gradient color
both vocal and instruments will be there but in some ratio, if it is 0 or 1(binary) then it will be like robot

Soft Masking: The "Hindi Music" Special
In Hindi music, a singer might hit the exact same pitch as a flute. If the mask was a simple "On/Off" switch (Binary Mask), it would sound choppy.

Instead, Spleeter uses Soft Masking. If a specific frequency contains both a Sitar and a Voice:

The Vocal Mask might be 0.7 (70% allowed).

The Other Mask might be 0.3 (30% allowed).

This ensures that the "Music Slot" and the "Vocal Slot" sound natural rather than robotic.




#
conversion errors of ffmpeg
{ the wav conversion error

1. The "Clean" Conda Solution (Highest Success Rate)
Since the environment is insisting on using its own tools, give it a version of FFmpeg that
actually works. This is the professional way to handle dependency conflicts in Data Science.

conda install -c conda-forge ffmpeg -y
fixes the broken dll files in the project's ffmpeg(env)

The "Pydub" Method (Why it failed)
Pydub is a "wrapper." When you call AudioSegment.from_file(), it doesn't actually read the audio itself.
 Instead, it starts a secret background conversation:

Pydub asks FFmpeg: "Hey, tell me about bp01.mp3. Give me the info in JSON format."

FFmpeg tries to look at the file, but because of the library conflicts or path issues in your environment,
 it crashes or sends back an empty message.

Pydub tries to read that empty message as a JSON object. Since it expects a { but gets nothing, it throws
 that Expecting value: line 1 column 1 error.

Essentially, Pydub was trying to "read the label" on the bottle before opening it, and the label was missing.


The Last solution which i applied:
The "Librosa" Method (Why it works)
Librosa takes a much more direct approach:

Librosa tells FFmpeg: "Don't talk to me; just stream the raw audio bits directly into this variable (y)."

It treats the audio as a NumPy array (just a big list of numbers representing sound waves).

Because Librosa doesn't ask for a JSON report, it never encounters the "missing label" problem. It just
 grabs the liquid (the audio) out of the bottle.

}




13/3/26 6:48 pm

explicitly telling the path# --- STEP 1: FORCE PATH AT THE TOP ---
# This ensures that even background processes see your clean FFmpeg
ffmpeg_bin = r'C:\FFmpeg\ffmpeg-8.0.1-full_build\bin'
os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]

worked succesfully


running the source_separation function which will separate all 500 songs in 4 stems

starting time - 7:04 pm

source_separation done for all 500 files now- total 2000 files - 20m 48 seconds, 7:25pm

## Feature Engineering
#Step 4: chunk divide with overlapping

# Waveshow with the corresponding audio for each stem for one file


#chunk diving and overlapping and waveshow for single file

# critical loop holes in the chunks above process:
1. Inconsistent input problem
-- as the last chunk will be shorted than all the chunk
-- Cnn expects all mel's or input to of same size
-- will give value error when training the Data

my solution- will make all mel's to same shape at the end
also - can make all chunk of same size also
but i will make them same size here

2. Silent slot issues
in the vocal, drums, or other stem in the chunk or in the whole audio
-- if there is no wave(no sound) model can learn silence means this genre

solutin -- we have to remove those chunks that are too low or silent
        -- Add an energy threshold check (RMS energy) to skip chunks that are too quiet.
done this step also
  
3. Data Leakage (Strategic)
This won't break the code, but it will ruin your research.

The Problem: If you chunk a song into 60 pieces and randomly shuffle them into Training 
and Testing sets, the model will "memorize" the song.

The Effect: You'll get 99% accuracy in the lab, but 20% accuracy in the real world. 
This is called Data Leakage.

The Fix: Always split your data by Song ID, not by Chunk ID.


Applied: on one file
##  Optimized chunking and overlapping 
#### 1. all chunks same length
#### 2. removal of silence 

#### at last shows perfect_chunks and at the last shows skipped chunks


### calculating the mel spectrogram for chunks
-- mel spectrogram show from the previous perfect_chunks by making a function for a 
single file

Final step 
# The Final Preprocessing "Master Plan"
# You have all the ingredients! Now you need to automate the "Factory Line." Since you have 5 genres, you need to save these spectrograms along with their Labels.

# The best way to do this for 500 songs is:

# Iterate through your hindi_stems.

# Chunk the audio (using your logic).

# Convert to Mel-spectrograms.

# Save as a NumPy array (.npy) and a separate labels array.

done unitl here pushing to git - 11:27 pm


19/3/2026 - Final data Preprocessing

preparing the dataset in mel's and their respectie labels in numpy array for each stem

example - 
1. vocal stem
-- vocal_features and vocal_labels

made a load and preprocess function which takes, data dir and classes
1. loop through all folder and find one stem(eg vocal)
2. goes through file and chunk it, convert to mel, resize and appned to data with labels
3. then saves the np array for both data and labels
4. repeating the whole task for each stems

at the end confirming the no of sample with thier shapes by loading the npy files

#
One Small Adjustment for Drums
For percussion, the timing is everything. Since you are using target_shape=(150, 150),
 the code is taking a 4-second drum loop and "squeezing" it into 150 pixels of width.

My Recommendation:
Keep the threshold at 0.01 for now, but if you notice it's skipping too many chunks, you 
can lower it to 0.005. Drums sometimes have very quiet but important ghost notes that define the genre.

#The "Normalization" Warning
Since you’ve moved to the training phase, remember that Mel-spectrograms are usually in Decibels 
(-80 to 0). Neural networks struggle with negative numbers and large scales.

#
The "Data Imbalance" Check
Before training, you should check if one genre (e.g., Bollypop) has 5,000 samples while another (e.g., Sufi) 
only has 500. If the counts per label are wildly different, the model will become "biased" towards the majority genre.

#What’s the Game Plan?
You have four separate datasets. You have two choices now:

Option A: Train one "Super Model" on the OTHER stem (since it has the most data and instrumental detail).

Option B: Train a Vocal model and a Drum model separately and see which one is more accurate.