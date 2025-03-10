# Notes

Frustrating how unbelievably convoluted the files are and with sparse documentation. I'll need to take some notes on the structure of the massive dataset.

Just to reiterate (to myself), the goal is to given a particular image, I can get the associated fMRI scan(s).

- Here is an example of flattening all the data into one dataframe https://huggingface.co/datasets/clane9/NSD-Flat
- There are sessions, runs, and trials.
- There are max 40 sessions where a person is being scanned (once a week for ~ a year)
- Each session is broken up into runs
- Each run is broken up into trials
- Each trial is showing an image for 3 seconds and a rest pause for 1 second 