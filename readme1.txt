

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

