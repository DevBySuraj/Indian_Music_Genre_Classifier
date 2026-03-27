# Indian_Music_Genre_Classifier





# Step 1: Coverting all the mp3 files to wav format and applying noise reduction to them.

Filtering and Noise Reduction: by use of pydub, noisereduce, soundfile
1. making a funciton which do this operation and saves all the process files in the new directory cleaned_songs_wav.


# Step 2: Source separaton by the use of spleeter

Separating all the audio files into 4 stems by the use of spleeter
made a function which do this operation(ffmpeg error remains needs to be fixed)
and makes 2000 files from 500 files by source separtion


# Step 3: Feature Engineering

Diving the audio files(all stems) in chunks of 4 second with 2 second overlap(waveshow)
1. considering only same length chunks
2. neglecting silent chunks for all stems
3. making mel's for chunks
4. saving all mel's with same shape for each chunk with it's label of the genre
5. now the data set is fully prepared with mel's and corresponding label to give it to the the CNN


# Step 4: Final dataset prepare for each stem (multi input CNN)
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

# Step 5: Dataset 3 way split and Normalization

Splitted the data in 3 way split with shuffling and stratfiy
1. training data 80%
2. validation data 10%
3. Test Data 10%

Normalization of data from decibel(-80, 0) to (0 - 1) on train data only for all splitof(X)


# Step 6: CNN building and Training

1. selecting the model
2. adding the layers 
3. compiling
4. training

-- cpu training 
-- gpu training
-- cuda and cDNN required
-- version mismatch of tf, spleeter - common ground tf 2.9 for both 
5. model save





    Japanese version
Music Source Separation tool with pre-trained models / ISMIR2019 extended abstract


## Reference:
Deezer Research - Source Separation Engine Story - deezer.io blog post:
    [English version](https://deezer.io/releasing-spleeter-deezer-r-d-source-separation-engine-2b88985e797e)
    [Japanese version](http://dzr.fm/splitterjp)
[Music Source Separation tool with pre-trained models / ISMIR2019 extended abstract](http://archives.ismir.net/ismir2019/latebreaking/000036.pdf)


@article{spleeter2020,
  doi = {10.21105/joss.02154},
  url = {https://doi.org/10.21105/joss.02154},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {50},
  pages = {2154},
  author = {Romain Hennequin and Anis Khlif and Felix Voituret and Manuel Moussallam},
  title = {Spleeter: a fast and efficient music source separation tool with pre-trained models},
  journal = {Journal of Open Source Software},
  note = {Deezer Research}
}