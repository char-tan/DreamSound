# DreamSound - Audio DeepDream

https://user-images.githubusercontent.com/76977995/133623797-abfdad05-c1b7-43e6-9082-95c9b9164077.mp4

This project is a fork of a [Pytorch DeepDream](https://github.com/gordicaleksa/pytorch-deepdream) implementation by [Aleksa Gordić](https://github.com/gordicaleksa) where I have applied the DeepDream algorithm to audio. VGG19 models were trained to classify mel spectrograms from two datasets; UrbanSounds8k and free-music-archive (small). These trained models are then used to "dream" audio features onto input mel spectrograms; like the original DeepDream this creates some interesting outputs! 

**WARNING!** This code can produce some horrible / loud sounds, turn the volume down before listening to an output for the first time. 

**Note:** due to git lfs not supporting public forks, I have a separate [repo](https://github.com/char-tan/DreamSoundModels) where the required model files can be found.

## Examples

examples/ contains some example outputs, including the parameters that were used in each case. 

**Note:** For completeness, I haven't cropped any of these wav files. Some of the ouroboros files end up "overcooked" (bad screeching sound).

## Usage

Important parameters

- input = the initial input to the model, either "noise" (default), "sine" or "path to wav file".
- layers_to_use = the activation layers to use, gradient ascent is done on the mean of these (see model.py for the list of available layers, "iterate" iterates backwards across the layers for ouroboros output).
- target_class_index = if "output" in layers_to_use, the output class to target / measure loss against.
- create_ouroboros = passes output into input repeatedly to create a sequence of progressively more processed clips.

**python dreamsound.py --input="sine" --layers_to_use "relu1_1" "relu5_2" "output" --target_class_index=4**

There is a tendency for ouroboros audio to eventually become "overcooked" (bad screeching sound), decreasing the number of iterations per clip can increase the length of ouroboros before this occurs. 

Please explore the other parameters and see how they affect the output! I would love to hear clips generated with this project, I need to figure out a good way to receive them.

## Target Classes

UrbanSounds8k - 10 classes

0. air_conditioner 
1. car_horn
2. children_playing
3. dog_bark
4. drilling
5. engine_idling
6. gun_shot
7.  jackhammer
8. siren
9. street_music

free-music-archive small - 8 classes

0. hip_hop
1. pop
2. folk
3. experimental
4. rock
5. international
6. electronic
7. instrumental

## Acknowledgements

I developed this code from https://github.com/gordicaleksa/pytorch-deepdream, many thanks to gordicaleksa for the fantastic DeepDream implementation!

The code I used to train the models was heavily inspired by the following series: https://towardsdatascience.com/urban-sound-classification-part-1-99137c6335f9

UrbandSounds8k dataset: https://urbansounddataset.weebly.com/

Free music archive dataset: https://github.com/mdeff/fma





