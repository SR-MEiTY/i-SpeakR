<pre>
# i-SpeakR
Speaker recognition toolkit for Indian languages
------------------------------------------------


How to run the toolkit?
-----------------------
1. Download the toolkit
2. Install Anaconda in your system. <a href="https://docs.anaconda.com/anaconda/install/">Refer here</a>
3. Run from terminal:
	$ cd [path to tookit]
4. Create a new virtual environment using the environment.yml file. <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file">Refer here</a>
5. Activate the new environment
6. Open run_program.sh and edit the values for various parameters. 
   The details about the parameters can be checked by running the 
   following command in the terminal:
	$ python main.py -h
7. Execute the following command in the terminal to run the toolkit:
	$ ./run_program.sh
8. Provide any inputs the programs asks for during the execution.



Coding conventions followed:
---------------------------
    Local variable:   Should have all lower-case alphabets [a-z]
                      Should start with an alphabet
                      Can include numerals [0-9]
                      Can include special characters [@, _, $]
                      Should end with an underscore "_"
    
    Global variables: Should have all upper-case alphabets [A-Z]
                      Should start with an alphabet
                      Can include numerals [0-9]
                      Can include special characters [@, _, $]
                      Should NOT end with an underscore "_"
    
    Function names:   Should have all lower-case alphabets [a-z]
                      Should follow snake case. [this_is_a_snake_case]
                      Should not include numerals [0-9]
                      Can start with an underscore "_"
    
    Class names:      Can have a combination of lower-case [a-z] and upper-case 
                      alphabets [A-Z]
                      Should follow Pascal case - ThisIsPascalCase
                      Words must start with an upper-case alphabet
                      Consecutive words must be concatenated
                      Should not include numerals [0-9] or special characters
                      
    Commenting:       All functions must include Doc Strings
                      All significant line of code must have an inline comment
                      All variable declarations must be commented




Structures for various IDs:
---------------------------

    Feature filename structure:
        [Feature folder]/[Split-ID].npy
            
    Split-ID structure:
        [Utterance-ID]_[Chop Size]_[Split count formatted as a 3-digit number]

    Utterance-ID structure:
        "infer" mode:
            [DEV/ENR/TEST]_[Speaker-ID]_[File Name]
        "specify" mode:
            [Speaker-ID]_[File Name]




Data input format:
------------------
There are two options to provide data.
    1. Basic: Provide the path to the root folder containing data. The data is
    expected to be stored in a specific format, as shown below:
        root/...
               |--DEV/-.
               |       |..[SpeakerID-1]/-.
               |       |                 |--{wav files}
                       |--[..]
               |       |                  
               |       |..[SpeakerID-N]/-.
               |                         |--{wav files}
               |        
               |--ENR/-.        
               |       |..[SpeakerID-1]/-.
               |       |                 |--{wav files}
                       |--[..]
               |       |                   
               |       |..[SpeakerID-N]/-.
               |                         |..{wav files}
               |
               |--TEST/-.        
                        |--[SpeakerID-1]/-.
                        |                 |--{wav files}
                        |--[..]
                        |                   
                        |--[SpeakerID-N]/-.
                                          |--{wav files}
                        
    2. Advanced: This option for advanced users. User needs to provide path 
    containing csv files. The csv filename for development should start with 
    “DEV”, csv filename for enrollment should start with “ENR”, and the csv 
    filename for testing should start with “TEST”. The csv files need to have 
    the following fields:
        1. utterance_id: Unique ID for each utterances listed in the csv file 
        (required)
        2. speaker_id: Speaker ID corresponding to each utterance in the csv 
        file (required)
        3. wav_path: Path to the wav file for each utterance in the csv file 
        (required)
        4. gender: Gender of the speaker for each utterance (optional)
        5. sensor: Sensor used to record each utterance (optional)
        6. language: Language used for each utterance (optional)

</pre>
