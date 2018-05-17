import os
import sys
import getopt
from streams import InputStream, OutputStream
from worker import Worker
from utils import out, err, warn
from visuserver import VisualizationServer
import json
import signal
from functools import partial
from collections import OrderedDict


def main(filename, file_dir, arguments):

    # let's print some stuff to debug the provided argument :)
    out()
    out('[Environment]')
    out('- Working dir: ' + os.getcwd())
    out('- Script dir: ' + file_dir)
    out('- Script: ' + filename)
    out('- Arguments: ' + str(arguments))
    out()

    # declaring basic options
    input_file = ''
    input_folder = ''
    output_folder = ''
    action = ''
    frames = -1
    repetitions = -1
    w = -1
    h = -1
    fps = -1
    force_gray = False
    visualization_port = 0

    # parameters that do not depend on the layer number
    layers = 1  # number of layers
    blur = True
    step_size = -1
    resume = 0
    all_black = False
    save_scores_only = False
    check_params = True
    grad = False
    step_adapt = False
    grad_order2 = False
    rk = 0

    # parameters that are defined layer-wise
    c_eps1 = {0: 10}  # layer-convergence threshold on the norm of the first derivative of q (squared norm)
    c_eps2 = {0: 10}  # layer-convergence threshold on the norm of the second derivative of q (squared norm)
    c_eps3 = {0: 10}  # layer-convergence threshold on the norm of the third derivative of q (squared norm)
    c_frames = {0: 10000}  # layer-convergence threshold on the number of processed frames
    c_frames_min = {0: 100}  # minimum number of frames processed by each layer
    features = {0: 4}  # number of features
    filter_size = {0: 3}  # "edge" size of a filter, i.e., the area is filter_size x filter_size
    k = {0: 0.5}  # weight of the norm of "q"
    alpha = {0: 0.5}  # weight of the norm of the second derivative q^(2)
    beta = {0: 1.0}  # weight of the norm of the first derivative q^(1)
    gamma = {0: 2.0}  # weight of the mixed-product first-second derivatives q^(2)q^(1)
    theta = {0: 1.0}  # exp(theta x t)
    lambdaM = {0: 1.0}  # motion constraint
    lambdaC = {0: 2.0}  # conditional entropy
    lambdaE = {0: 1.0}  # entropy
    eps1 = {0: 100000.0}  # bound on the norm of the first derivative q^(1)
    eps2 = {0: 100000.0}  # bound on the norm of the second derivative q^(2)
    eps3 = {0: 100000.0}  # bound on the norm of the third derivative q^(3)
    eta = {0: 0.25}
    rho = {0: 0.0}
    init_q = {0: 0.1}
    gew = {0: 1.0}  # weight to the last measurement of the entropy term (historical moving average)
    day_only = {0: False}
    init_fixed = {0: False}

    # command line arguments and their description (yes, I should have made a dictionary...)
    accepted_options = ["port=", "resume=", "run=", "out=", "res=", "fps=", "frames=",
                        "gray=", "f=", "m=", "init_q=", "theta=", "alpha=", "beta=",
                        "k=", "gamma=", "lambdaC=", "lambdaE=", "lambdaM=",
                        "rep=", "eps1=", "eps2=", "eps3=", "eta=",
                        "step_size=", "all_black=", "init_fixed=", "check_params=",
                        "grad=", "rho=", "day_only=",
                        "save_scores_only=", "gew=", "rk=", "step_adapt=", "blur=", "grad_order2=",
                        "layers=", "c_eps1=", "c_eps2=", "c_eps3=", "c_frames=", "c_frames_min="]
    description = ["port of the visualization service",
                   "resume an experiment from the output folder (binary flag - "
                   "if set to 2 also clear the output folder and stats",
                   "the video file (also folders of frames are supported)",
                   "the path of the output folder (it will be created/cleared)", "input resolution (example: 240x120)",
                   "frames per second", "maximum number of frames to consider", "force gray scale (binary flag)",
                   "edge of a filter (example for a 3x3 filter: 3)", "number of features",
                   "maximum absolute value of initial components of q",
                   "exp(theta x t)", "weight of the norm of the second derivative of q",
                   "weight of the norm of the first derivative of q", "weight of the norm of q",
                   "weight of the mixed-product first-second derivatives", "weight of the conditional entropy",
                   "weight of the entropy", "weight of the motion constraint",
                   "number of video repetitions",
                   "bound on the norm of the first derivative of q (squared norm)",
                   "bound on the norm of the second derivative of q (squared norm)",
                   "bound on the norm of the third derivative of q (squared norm)",
                   "update factor for parameter rho",
                   "size of the step in the Euler's method (or learning rate in the gradient-based update)",
                   "force input to be all black",
                   "initialize all the components of q to init_q",
                   "check validity of the alpha, beta, gamma, theta, and k parameters (binary flag)",
                   "switch to online stochastic gradient (binary flag)",
                   "blurring factor",
                   "force the system to work only in day-mode",
                   "do not save output data, with the exception of the scalar scores (binary flag)",
                   "weight to give to the entropy term in the moving-average-based estimate",
                   "use the Runge-Kutta method (binary flag) - not implemented yet",
                   "use adaptive step size (gradient-based optimization only)",
                   "use Gaussian blurring (binary flag)",
                   "simulates Gradient-based updates by means of a second-order diff. eq. (binary flag)",
                   "number of layers",
                   "layer-convergence threshold on the norm of the first derivative of q (squared norm)",
                   "layer-convergence threshold on the norm of the second derivative of q (squared norm)",
                   "layer-convergence threshold on the norm of the third derivative of q (squared norm)",
                   "layer-convergence threshold on the number of processed frames",
                   "minimum number of frames processed by each layer"]

    # processing the input arguments in order to detect the layer ID of the arguments (when provided)
    if arguments is not None and len(arguments) > 0:

        # finding arguments that are about layer-related options
        i = 0
        layer_opts = []
        for a in arguments:
            layer_opt = -1
            if a[0] == '-':
                u = 0
                while a[-(u + 1)].isdigit():
                    u = u + 1
                if u > 0 and a[-(u + 1)] == '_':
                    layer_opt = int(a[-(u):])
                    arguments[i] = a[:-(u + 1)]
                layer_opts.append(layer_opt)
            i = i + 1

        # parsing options
        print(arguments)
        try:
            opts, args = getopt.getopt(arguments, "", accepted_options)
        except getopt.GetoptError:
            usage(filename, accepted_options, description)
            sys.exit(2)
    else:
        usage(filename, accepted_options, description)
        sys.exit(2)

    # processing the input arguments, extracting their values and pairing them with the layer ID
    try:
        # handling the validated options (putting back the layer-related info)
        i = 0
        for opt, arg in opts:
            if opt == '--run':
                if os.path.isdir(arg):
                    input_folder = os.path.abspath(arg)
                    if input_folder.endswith(os.sep):
                        input_folder = input_folder[:-1]
                elif os.path.isfile(arg):
                    input_file = os.path.abspath(arg)
                elif arg == '0':
                    input_file = arg
                else:
                    raise IOError("Cannot open: " + arg)
                action = 'run'
            elif opt == '--out':
                output_folder = os.path.abspath(arg)
                if output_folder.endswith(os.sep):
                    output_folder = output_folder[:-1]
            elif opt == '--res':
                wh = arg.split('x')
                w = int(wh[0])
                h = int(wh[1])
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--gray':
                if int(arg) > 0:
                    force_gray = True
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--fps':
                fps = float(arg)
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--frames':
                frames = int(arg)
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--rep':
                repetitions = int(arg)
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--m':
                features[max(layer_opts[i],0)] = int(arg)
            elif opt == '--f':
                filter_size[max(layer_opts[i],0)] = int(arg)
            elif opt == '--init_q':
                init_q[max(layer_opts[i],0)] = float(arg)
            elif opt == '--theta':
                theta[max(layer_opts[i],0)] = float(arg)
            elif opt == '--alpha':
                alpha[max(layer_opts[i],0)] = float(arg)
            elif opt == '--beta':
                beta[max(layer_opts[i],0)] = float(arg)
            elif opt == '--eps1':
                eps1[max(layer_opts[i],0)] = float(arg)
            elif opt == '--eps2':
                eps2[max(layer_opts[i],0)] = float(arg)
            elif opt == '--eps3':
                eps3[max(layer_opts[i],0)] = float(arg)
            elif opt == '--eta':
                eta[max(layer_opts[i],0)] = float(arg)
            elif opt == '--k':
                k[max(layer_opts[i],0)] = float(arg)
            elif opt == '--gamma':
                gamma[max(layer_opts[i],0)] = float(arg)
            elif opt == '--lambdaC':
                lambdaC[max(layer_opts[i],0)] = float(arg)
            elif opt == '--lambdaM':
                lambdaM[max(layer_opts[i],0)] = float(arg)
            elif opt == '--lambdaE':
                lambdaE[max(layer_opts[i],0)] = float(arg)
            elif opt == '--step_size':
                step_size = float(arg)
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--resume':
                resume = int(arg)
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--port':
                visualization_port = int(arg)
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--all_black':
                all_black = int(arg) > 0
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--init_fixed':
                init_fixed[max(layer_opts[i],0)] = int(arg) > 0
            elif opt == '--check_params':
                check_params = int(arg) > 0
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--rho':
                rho[max(layer_opts[i],0)] = float(arg)
            elif opt == '--grad':
                grad = int(arg) > 0
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--day_only':
                day_only[max(layer_opts[i],0)] = int(arg) > 0
            elif opt == '--save_scores_only':
                save_scores_only = int(arg) > 0
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--gew':
                gew[max(layer_opts[i],0)] = float(arg)
            elif opt == '--rk':
                rk = int(arg) > 0
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--step_adapt':
                step_adapt = int(arg) > 0
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--blur':
                blur = int(arg) > 0
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--grad_order2':
                grad_order2 = int(arg) > 0
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            elif opt == '--c_eps1':
                c_eps1[max(layer_opts[i],0)] = float(arg)
            elif opt == '--c_eps2':
                c_eps2[max(layer_opts[i],0)] = float(arg)
            elif opt == '--c_eps3':
                c_eps3[max(layer_opts[i],0)] = float(arg)
            elif opt == '--c_frames':
                c_frames[max(layer_opts[i],0)] = int(arg)
            elif opt == '--c_frames_min':
                c_frames_min[max(layer_opts[i],0)] = int(arg)
            elif opt == '--layers':
                layers = int(arg)
                if layer_opts[i] >= 0:
                    raise ValueError("This option cannot be defined layer-wise: " + opt)
            i = i + 1
    except (ValueError, IOError) as e:
        err(e)
        sys.exit(1)

    # default options are the ones defined in the first layer
    n = {0: -1}
    for i in range(0,layers):
        if i not in c_eps1:
            c_eps1[i] = c_eps1[0]
        if i not in c_eps2:
            c_eps2[i] = c_eps2[0]
        if i not in c_eps3:
            c_eps3[i] = c_eps3[0]
        if i not in c_frames:
            c_frames[i] = c_frames[0]
        if i not in c_frames_min:
            c_frames_min[i] = c_frames_min[0]
        if i not in features:
            features[i] = features[0]
        if i not in filter_size:
            filter_size[i] = filter_size[0]
        if i not in k:
            k[i] = k[0]
        if i not in alpha:
            alpha[i] = alpha[0]
        if i not in beta:
            beta[i] = beta[0]
        if i not in gamma:
            gamma[i] = gamma[0]
        if i not in theta:
            theta[i] = theta[0]
        if i not in lambdaM:
            lambdaM[i] = lambdaM[0]
        if i not in lambdaC:
            lambdaC[i] = lambdaC[0]
        if i not in lambdaE:
            lambdaE[i] = lambdaE[0]
        if i not in eps1:
            eps1[i] = eps1[0]
        if i not in eps2:
            eps2[i] = eps2[0]
        if i not in eps3:
            eps3[i] = eps3[0]
        if i not in eta:
            eta[i] = eta[0]
        if i not in rho:
            rho[i] = rho[0]
        if i not in init_q:
            init_q[i] = init_q[0]
        if i not in gew:
            gew[i] = gew[0]
        if i not in day_only:
            day_only[i] = day_only[0]
        if i not in init_fixed:
            init_fixed[i] = init_fixed[0]
        if i > 0:
            n[i] = features[i-1]

    # sorting by layer index
    c_eps1 = OrderedDict(sorted(c_eps1.items()))
    c_eps2 = OrderedDict(sorted(c_eps2.items()))
    c_eps3 = OrderedDict(sorted(c_eps3.items()))
    c_frames = OrderedDict(sorted(c_frames.items()))
    c_frames_min = OrderedDict(sorted(c_frames_min.items()))
    features = OrderedDict(sorted(features.items()))
    filter_size = OrderedDict(sorted(filter_size.items()))
    k = OrderedDict(sorted(k.items()))
    alpha = OrderedDict(sorted(alpha.items()))
    beta = OrderedDict(sorted(beta.items()))
    gamma = OrderedDict(sorted(gamma.items()))
    theta = OrderedDict(sorted(theta.items()))
    lambdaM = OrderedDict(sorted(lambdaM.items()))
    lambdaC = OrderedDict(sorted(lambdaC.items()))
    lambdaE = OrderedDict(sorted(lambdaE.items()))
    eps1 = OrderedDict(sorted(eps1.items()))
    eps2 = OrderedDict(sorted(eps2.items()))
    eps3 = OrderedDict(sorted(eps3.items()))
    eta = OrderedDict(sorted(eta.items()))
    rho = OrderedDict(sorted(rho.items()))
    init_q = OrderedDict(sorted(init_q.items()))
    gew = OrderedDict(sorted(gew.items()))
    day_only = OrderedDict(sorted(day_only.items()))
    init_fixed = OrderedDict(sorted(init_fixed.items()))

    # the output folder must be provided!
    if len(output_folder) == 0:
        usage(filename, accepted_options, description)
        sys.exit(2)

    # fixing/forcing options
    if repetitions <= 0:
        repetitions = 1
    input_is_video = input_file is not None and len(input_file) > 0

    # printing parsed options
    out('[Parsed Video Options]')
    out('- Action: ' + action)
    out('- Input: ' + (input_file or input_folder))
    out('- Output: ' + output_folder)
    out('- ForceGray: ' + ("yes" if force_gray else "no"))
    out('- Repetitions: ' + str(repetitions))
    out('- Frames: ' + ("empty" if frames <= 0 else str(frames)))
    out()

    # opening the input stream
    input_stream = InputStream(input_file or input_folder, input_is_video, blur)

    # checking option-related errors (given the information acquired from the input stream)
    try:
        if grad_order2 and grad:
            raise ValueError('You cannot activate both grad and grad_order2!')

        if fps <= 0:
            if step_size < 0.0:
                fps = input_stream.fps
            else:
                if step_size > 0.0:
                    fps = 1.0 / step_size
                else:
                    fps = input_stream.fps

        if frames != -1 and input_stream.frames < frames:
            raise ValueError('Video frames: ' + str(input_stream.frames) + ', requested frames: ' + str(frames))
        if frames <= 0:
            if fps != input_stream.fps:
                v_len = float(input_stream.frames) / float(input_stream.fps)
                frames = int(v_len * fps)
            else:
                frames = input_stream.frames

        if w <= 0:
            w = input_stream.w
        if h <= 0:
            h = input_stream.h
        if force_gray:
            c = 1
        else:
            c = input_stream.c

        if step_size < 0.0:
            step_size = 1.0 / fps
    except ValueError as e:
        err(e)
        sys.exit(1)
    tot_frames = frames * repetitions

    # printing input stream information
    out('[Input Stream Info]')
    out('- Width: ' + str(input_stream.w))
    out('- Height: ' + str(input_stream.h))
    out('- Channels: ' + str(input_stream.c))
    out('- FPS: ' + ("empty" if float(input_stream.fps) <= 0.0 else str(input_stream.fps)))
    out('- Frames: ' + ("empty" if input_stream.frames < 0 else str(input_stream.frames)))
    out()

    # setting up the output stream (folder)
    output_stream = OutputStream(output_folder, resume == 0)

    # opening a visualization service (random port)
    visualization_server = None
    if not save_scores_only:
        visualization_server = VisualizationServer(visualization_port,
                                                   os.path.abspath(file_dir + os.sep + "web"), output_folder)
        out('[Visualization Server]')
        out('- IP: ' + str(visualization_server.ip))
        out('- Port: ' + str(visualization_server.port))
        out('- Data Root: ' + os.path.abspath(output_folder))
        out("- URL: http://" + str(visualization_server.ip) + ":" + str(visualization_server.port))
        out()

    # setting up algorithm options
    n[0] = c
    options = {'root': os.path.abspath(output_folder),
               'm': features,
               'n': n,
               'f': filter_size,
               'init_q': init_q,
               'theta': theta,
               'alpha': alpha,
               'beta': beta,
               'k': k,
               'gamma': gamma,
               'lambdaC': lambdaC,
               'lambdaM': lambdaM,
               'lambdaE': lambdaE,
               'eps1': eps1,
               'eps2': eps2,
               'eps3': eps3,
               'eta': eta,
               'step_size': step_size,
               'all_black': all_black,
               'init_fixed': init_fixed,
               'check_params': check_params,
               'rho': rho,
               'day_only': day_only,
               'save_scores_only': save_scores_only,
               'gew': gew,
               'rk': rk,
               'grad': grad,
               'step_adapt': step_adapt,
               'blur': blur,
               'grad_order2': grad_order2,
               'layers': layers,
               'c_eps1': c_eps1,
               'c_eps2': c_eps2,
               'c_eps3': c_eps3,
               'c_frames': c_frames,
               'c_frames_min': c_frames_min}

    out('[Algorithm Options]')
    out(json.dumps(options, indent=3))
    out()

    # saving options to the output folder
    output_stream.save_option("resolution", str(w) + "x" + str(h))
    output_stream.save_option("w", str(w))
    output_stream.save_option("h", str(h))
    output_stream.save_option("fps", str(fps))
    output_stream.save_option("real_video_frames", str(input_stream.frames))
    output_stream.save_option("repetitions", str(repetitions))
    output_stream.save_option("frames", str(tot_frames))   # processed frames

    for k, v in options.items():
        output_stream.save_option(k, v)

    if not save_scores_only:
        output_stream.save_option("ip", str(visualization_server.ip))
        output_stream.save_option("port", str(visualization_server.port))
        output_stream.save_option("url", "http://" + str(visualization_server.ip) + ":"
                                  + str(visualization_server.port))

    # interruption (CTRL+C) halder
    def interruption(status_array, signal, frame):
        status_array[0] = False

    # creating the real worker
    status = [True]
    try:
        worker = Worker(input_stream, output_stream, w, h, fps, frames, force_gray, repetitions, options,
                        resume > 0, resume > 1)
        signal.signal(signal.SIGINT, partial(interruption, status))
    except ValueError as e:
        err(e)
        if not save_scores_only:
            visualization_server.close()
        sys.exit(1)

    # main loop
    out("Operations are now starting...")

    while status[0]:
        try:
            status_new, frame_time, frame_time_no_save_io, frame_time_no_io = worker.run_step()
            status[0] = status[0] * status_new
        except ValueError as e:
            try:
                worker.close()
            except (ValueError, IOError):
                pass
            err("Error while processing frame " + str(int(worker.steps)+1) + "/" + str(tot_frames))
            err(e)
            if not save_scores_only:
                visualization_server.close()
            sys.exit(1)

        if status_new:
            out("Processed frame " + str(int(worker.steps)) + "/" + str(tot_frames) +
                " [frame_time: " + "{0:.3f}".format(frame_time) + " (" +
                "{0:.3f}".format(frame_time_no_save_io) + "), avg_fps: " + "{0:.3f}".format(worker.measured_fps) + "]")
    worker.save()
    worker.close()
    out("Done! (model saved)")

    # fixing errors in the frame count procedure
    if worker.steps != tot_frames:
        output_stream.save_option("frames", str(worker.steps))

    # quit the visualization service
    if not save_scores_only:
        visualization_server.close()


def usage(filename, accepted_options, description):
    out("Usage:")
    out()
    out(filename + ' --run <file/folder/0> --out <folder> [options]')
    out()
    out("where 'file' is video file, and '0' indicates the local web-cam")
    out()
    out("Full list of parameters and options:")
    i = 0
    for k in accepted_options:
        if i < len(description):
            if description[i] != "none":
                tab = (18 - len(k[:-1])) * " "
                out("   --" + k[:-1] + ":" + tab + description[i])
        else:
            out("   --" + k[:-1])
        i = i + 1
    out()


if __name__ == "__main__":
    main(os.path.basename(sys.argv[0]), os.path.dirname(os.path.abspath(sys.argv[0])), sys.argv[1:])
