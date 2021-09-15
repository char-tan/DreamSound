import os
import argparse
import random
import soundfile as sf
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import *
from utils import *

def gradient_ascent(opt, model, input_tensor, iteration):
    "one gradient ascent iteration"

    activations = model(input_tensor) # model returns dict of activations

    # calculate losses over activations
    losses = []
    for layer in opt.layers_to_use:

        # MSE with zero vector
        if not layer == "output":
            target_vector = torch.zeros_like(activations[layer])
            loss_component = torch.nn.MSELoss(reduction='mean')(activations[layer], target_vector)

        # CrossEntropyLoss with target class
        elif layer == "output": 
            target_vector = torch.tensor([opt.target_class_index], device="cuda")
            loss_component = torch.nn.CrossEntropyLoss()(activations[layer], target_vector)
            
        losses.append(loss_component)

    loss = torch.mean(torch.stack(losses))
    loss.backward()

    # smooth gradients with gaussian kernels
    grad = input_tensor.grad.detach()
    sigma = ((iteration + 1) / opt.num_iterations) * 2.0 + opt.smoothing_coefficient
    smooth_grad = CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well

    # normalise gradients
    smooth_grad = (smooth_grad - smooth_grad.mean()) / smooth_grad.std()

    # update image with gradient ascent
    input_tensor.data += opt.lr * smooth_grad

    # clear gradients
    input_tensor.grad.data.zero_()

    # clamp input
    input_tensor.data = torch.clamp(input_tensor, min=-3, max=3)

def deep_dream_spectrogram(opt, input_spec, model):
    "applies DeemDream to single spectrogram"

    # convert to tensor and add batch / chan dims
    input_spec = torch.from_numpy(input_spec).unsqueeze(0).unsqueeze(0).float().cuda()

    input_spec = standardise(input_spec, opt.mean, opt.std) # standardise with stats of training data

    base_shape = input_spec.shape  # save initial height and width

    unprocessed = input_spec.clone() # for detail recovery

    # iterate over sizes in pyramid (small to big)
    for pyramid_level in range(opt.pyramid_size):

        new_shape = get_new_shape(opt, base_shape, pyramid_level) # get new shape
        input_spec = F.interpolate(input_spec, size=(new_shape[0], new_shape[1])) # resize to new shape
        
        if pyramid_level != 0 and opt.replace_detail:
            "help from: https://keras.io/examples/generative/deep_dream/"
            # replace detail lost by interpolation
            upsampled_shrunk_unprocessed = F.interpolate(shrunk_unprocessed, size=(new_shape[0], new_shape[1])) # upsample previous size to current size
            same_size_unprocessed = F.interpolate(unprocessed, size=(new_shape[0], new_shape[1])) # downsample original to current size
            lost_detail = same_size_unprocessed - upsampled_shrunk_unprocessed # lost detail is difference
            
            input_spec = input_spec + lost_detail * opt.detail_factor # factor can be used to add less detail

        # roll patch horizontally
        pyramid_roll_max = get_pyramid_roll_max(opt, base_shape, pyramid_level) # get max roll for pyramid level
        pyramid_roll = torch.randint(pyramid_roll_max, size=(1,)).item() # get random roll value in range
        input_spec = input_spec.roll(pyramid_roll, dims=-1)

        input_spec.requires_grad_()

        # gradient ascent iterations
        for iteration in range(opt.num_iterations):
            gradient_ascent(opt, model, input_spec, iteration)

        input_spec = input_spec.detach()

        # unroll patch
        input_spec = input_spec.roll(-pyramid_roll, dims=-1)

        shrunk_unprocessed = F.interpolate(unprocessed, size=(new_shape[0], new_shape[1])) # for replacing detail at next size

    input_spec = destandardise(input_spec, opt.mean, opt.std) # destandardise with stats of training data

    return input_spec.cpu().detach().numpy()[0][0]

def dream_sound_single(opt, input_wav, model):
    "single clip dream"

    # convert input to spectrogram
    input_spec = wav_to_mel_dB(input_wav)

    print("dreaming single spectrogram")
    dream_spec = deep_dream_spectrogram(opt, input_spec, model)

    print("converting to wav")
    dream_wav = mel_dB_to_wav(dream_spec)

    print("writing to file")
    sf.write(opt.save_path+"/dream.wav", dream_wav, samplerate=opt.sample_rate)

    print("plotting")
    fig, ax = plt.subplots(4)
    plot_mel_dB(input_spec, fig, ax, 0)
    plot_mel_dB(dream_spec, fig, ax, 1)
    plot_wav(input_wav, fig, ax, 2)
    plot_wav(dream_wav, fig, ax, 3)
    plt.show()

def dream_sound_ouroboros(opt, input_wav, model):
    "repeatedly feeds output spectrogram back into input"

    print(opt.layers_to_use)

    if opt.layers_to_use == ["iterate"]:
        iterate_flag = True
        print("iterating over layers")
    else:
        iterate_flag = False

    transform_output_length = input_wav.shape[0]

    # convert input to spectrogram
    input_spec = wav_to_mel_dB(input_wav)

    print("dreaming ouroboros")

    dream_specs = [input_spec]
    dream_wavs = [input_wav]

    # iterate over length of ouroboros
    for i in range(opt.ouroboros_length):

        if iterate_flag:
            # iterate backwards over layers in model
            opt.layers_to_use = [opt.layers[-(i+1)]]
            print("maximising",opt.layers_to_use)

        print(f'ouroboros iteration {i+1}.')

        # generate dream for this iteration
        dream_spec = deep_dream_spectrogram(opt, input_spec, model)  
        dream_specs.append(dream_spec)

        # convert to wav
        dream_wav = mel_dB_to_wav(dream_spec)
        dream_wavs.append(dream_wav)

        if not opt.ouroboros_transform == "none":
            input_wav = ouroboros_transform(opt, dream_wav, transform_output_length) # apply transformation to dream wav
            input_spec = wav_to_mel_dB(input_wav) # generate spec from transformed dream wav
        else:
            input_spec = dream_spec # use mel that hasn't been converted to and from audio

    # fade between oouroboros clips
    if opt.crossfade:
        clips = [dream_wavs[0]]
        clip_length = dream_wavs[-1].size # quick fix for audio files shorted by wav -> mel -> wav

        fade_in = np.linspace(0.0, 1.0, num = clip_length) # amplitude envelope vectors
        fade_out = np.linspace(1.0, 0.0, num = clip_length)

        for i in range(len(dream_wavs) - 1):
            fade_out_clip = dream_wavs[i][:clip_length] * fade_out
            fade_in_clip = dream_wavs[i+1][:clip_length] * fade_in
            clips.append(fade_out_clip + fade_in_clip)

        dream_ouroboros = np.concatenate(clips)

    # no fading
    else:
        dream_ouroboros = np.concatenate(dream_wavs)

    print("writing to file")
    sf.write(opt.save_path+"/dream_ouroboros.wav", dream_ouroboros, samplerate=opt.sample_rate)

    print("plotting")
    fig, ax = plt.subplots(opt.ouroboros_length+1)
    for i, dream_spec in enumerate(dream_specs):
        plot_mel_dB(dream_spec, fig, ax, i)
    plt.show()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input / output params
    parser.add_argument("--input", type=str, help="input wav for dreaming, or 'noise' / 'sine' ", default="noise")
    parser.add_argument("--clip_length", type=int, help="length of noise / sine clip to use", default=51200)
    parser.add_argument("--sine_freq", type=int, help="frequency of sine wave", default=250)
    parser.add_argument("--sample_rate", type=int, help="sample rate to use", default=22050)

    # model params
    parser.add_argument("--model", type=str, help="model to use", default="fma-small", choices=["UrbanSounds8k", "fma-small"])
    parser.add_argument("--layers_to_use", type=str, nargs='+', help="Layer whose activations we should maximize while dreaming", default=["relu5_4"])
    parser.add_argument("--target_class_index", type=int, help="class of final output layer to target, -1 for random, no effect if 'output' not in layers_to_use", default=-1)

    # dreaming params
    parser.add_argument("--pyramid_size", type=int, help="number of levels in a pyramid", default=3)
    parser.add_argument("--pyramid_ratio", type=float, help="ratio of sizes in the pyramid", default=2)
    parser.add_argument("--replace_detail", type=bool, help="to replace detail lost by interpolation", default=True)
    parser.add_argument("--detail_factor", type=float, help="multiplying factor to add less detail", default=1.0)
    parser.add_argument("--pyramid_roll", type=int, help="max horizontal rolling each iteration (scales with pyramid)", default=20)
    parser.add_argument("--num_iterations", type=int, help="number of gradient ascent iterations per pyramid level", default=10)
    parser.add_argument("--lr", type=float, help="learning rate i.e. step size in gradient ascent", default=0.02)
    parser.add_argument("--smoothing_coefficient", type=float, help='directly controls standard deviation for gradient smoothing', default=0.5)

    # ourobors dreaming params
    parser.add_argument("--create_ouroboros", type=bool, help="create ouroboros audio", default=False)
    parser.add_argument("--ouroboros_length", type=int, help="number of clips in ouroboros audio", default=10)
    parser.add_argument("--ouroboros_transform", type=str, help="transform to apply between each dream iteration", default="none", choices=["none", "time_stretch", "resample"])
    parser.add_argument("--ouroboros_stretch", type=float, help="factor for 'stretch' transform", default=0.95)
    parser.add_argument("--ouroboros_sr_offset", type=int, help="sample rate offset for 'resample' transform", default=500)
    parser.add_argument("--crossfade", type=bool, help="fade between ouroboros clips", default=True)

    opt = parser.parse_args()

    # classes / mean / std of models
    if opt.model == "UrbanSounds8k":
        opt.num_classes = 10
        opt.mean = -23.5891
        opt.std = 16.3979
    elif opt.model == "fma-small":
        opt.num_classes = 8
        opt.mean = -24.1847 
        opt.std = 16.8729 

    if opt.layers_to_use == ["iterate"]:
        # list of all layers in model
        opt.layers = ["conv1_1", "relu1_1",  "conv1_2", "relu1_2", "mp1",
                "conv2_1", "relu2_1",  "conv2_2", "relu2_2", "mp2",
                "conv3_1", "relu3_1",  "conv3_2", "relu3_2", "conv3_3", "relu3_3",  "conv3_4", "relu3_4", "mp3",
                "conv4_1", "relu4_1",  "conv4_2", "relu4_2", "conv4_3", "relu4_3",  "conv4_4", "relu4_4", "mp4",
                "conv5_1", "relu5_1",  "conv5_2", "relu5_2", "conv5_3", "relu5_3",  "conv5_4", "relu5_4", "mp5",
                "avgpool", "linear1", "reluC_1", "dropout1", "linear2", "reluC_2", "dropout2", "output"]
        opt.ouroboros_iterations = len(opt.layers)
 
    # initalise model
    model = Vgg19Activations("models/"+opt.model+".pth", opt.layers_to_use, opt.num_classes).cuda()

    # random target class
    if opt.target_class_index == -1:
        opt.target_class_index = random.randint(0, opt.num_classes)

    # folder prep
    os.makedirs("outputs", exist_ok=True)

    for i in range(1000):
        opt.save_path = "outputs/"+str(i)
        try:
            os.makedirs(opt.save_path, exist_ok=False)
            break
        except FileExistsError:
            pass

    # write params to file
    with open (opt.save_path + "/params.txt", "w") as f:
        f.write(str(vars(opt)))

    # generate / load input
    if opt.input == "noise":
        rng = np.random.default_rng()
        input_wav = rng.random(opt.clip_length)
        input_wav = input_wav * 2 - 1
    elif opt.input == "sine":
        sine = np.arange(0, opt.clip_length) / opt.sample_rate
        input_wav = np.sin(sine*2*np.pi*opt.sine_freq)
    else:
        input_wav, sr = librosa.load(opt.input, sr=opt.sample_rate)

    input_wav = librosa.util.normalize(input_wav)


    if not opt.create_ouroboros:
        # create a single DeepDream audio
        dreamed_spec = dream_sound_single(opt, input_wav, model)  
    else:  
        # create ouroboros audio (feeding neural network's output to it's input)
        dream_sound_ouroboros(opt, input_wav, model)
